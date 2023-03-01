import os
import copy
import argparse
import wandb

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import optax
import equinox as eqx

from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=int, 
                    default=0, help="GPU identifier.")

parser.add_argument("--num_cores", type=int, 
                    default=8, help="Number of cores for MCTS.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=2500, help="Number of runs on random data.")

parser.add_argument("--num_actions", type=int, 
                    default=11, help="Number of actions.")

parser.add_argument("--num_simulations", type=int, 
                    default=25, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=128, help="Learning batchsize.")

parser.add_argument("--regularization", type=float, 
                    default=1e-2, help="Contribution of L2 regularization.")

parser.add_argument("--lr", type=float, 
                    default=2e-4, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=4, help="Number input variables.")

parser.add_argument("--num_outputs", type=int, 
                    default=4, help="Number of output variables.")

args = parser.parse_args()

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(args.num_cores)
cpu_devices = jax.devices("cpu")
gpu_devices = jax.devices("gpu")
print("cpu", cpu_devices)
print("gpu", gpu_devices)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" # increases performance enormously!
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import graphax
from graphax import VertexGame, make_vertex_game_state
from graphax.core import forward, reverse
from graphax.examples import construct_random, \
							construct_Helmholtz, \
							construct_LIF

from alphagrad.utils import A0_loss, get_masked_logits, preprocess_data, postprocess_data
from alphagrad.data import VertexGameGenerator, \
							make_recurrent_fn, \
							make_environment_interaction
from alphagrad.modelzoo import CNNModel, TransformerModel
from alphagrad.differentiate import differentiate

wandb.init("Vertex_AlphaZero")
wandb.run.name = args.name
wandb.config = vars(args)

NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs
NUM_GAMES = 256
SHAPE = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES+NUM_OUTPUTS)

key = jrand.PRNGKey(args.seed)
edges, INFO = construct_Helmholtz() # construct_LIF() # 
state = make_vertex_game_state(INFO, edges)
env = VertexGame(state)
INFO = graphax.GraphInfo(4, 11, 4, 128)

nn_key, key = jrand.split(key, 2)
subkeys = jrand.split(nn_key, 4)
MODEL = TransformerModel(INFO, 128, 64, 7, 3, 6, key=key)




batched_step = jax.vmap(env.step)
batched_reset = jax.vmap(env.reset)
batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0, None))


recurrent_fn = make_recurrent_fn(MODEL, 
                                INFO, 
                                batched_step, 
                                batched_get_masked_logits)


env_interaction = make_environment_interaction(INFO, 
                                            	args.num_simulations,
                                                recurrent_fn,
                                                batched_step,
                                                batched_one_hot,
												temperature=0)


game_generator = VertexGameGenerator(NUM_GAMES, INFO, key)
optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(MODEL, eqx.is_inexact_array))

### needed to reassemble data
num_i = INFO.num_inputs
num_v = INFO.num_intermediates
num_o = INFO.num_outputs
edges_shape = (num_i+num_v, num_v+num_o)
obs_idx = jnp.prod(jnp.array(edges_shape))
policy_idx = obs_idx + num_v
reward_idx = policy_idx + 1
split_idxs = (obs_idx, policy_idx, reward_idx)


import time
def train_agent(data, network, opt_state, key):
    # data movement is a bottleneck
	data = data.reshape(-1, *data.shape[2:])
	data = postprocess_data(data, -2)

	obs, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	batchsize = args.batchsize*num_v
	search_policy = search_policy.reshape(batchsize, num_v)
	search_value = search_value.reshape(batchsize, 1)
	obs = obs.reshape(batchsize, *edges_shape)

	key, subkey = jrand.split(key, 2)
	subkeys = jrand.split(subkey, obs.shape[0])
	loss, grads = eqx.filter_value_and_grad(A0_loss)(network, 
                                                	search_policy, 
                                                	search_value, 
                                                	obs,
													args.regularization,
                                                	subkeys)
	updates, opt_state = optim.update(grads, opt_state)
	network = eqx.apply_updates(network, updates)
	return loss, network, opt_state


edges, info = construct_Helmholtz()
helmholtz_game = make_vertex_game_state(info, edges)

gkey, key = jrand.split(key)
edges, info = construct_random(gkey, INFO, fraction=.35)
print(edges)
random_game = make_vertex_game_state(info, edges)


forward_edges = copy.deepcopy(edges)
_, ops = forward(forward_edges, info)
print("forward-mode:", ops)


reverse_edges = copy.deepcopy(edges)
_, ops = reverse(reverse_edges, info)
print("reverse-mode:", ops)


pbar = tqdm(range(args.episodes))
rewards = []
for e in pbar:
	data_key, env_key, train_key, key = jrand.split(key, 4)
	batch_games = game_generator(args.batchsize, data_key)

	### BEGIN Code for parallelization
	start_time = time.time()
	init_carry = preprocess_data(batch_games, args.num_cores, env_key)
	data = env_interaction(MODEL, init_carry)
	print("mcts", time.time() - start_time)
	### END code for parallelization

	start_time = time.time()
	loss, MODEL, opt_state = eqx.filter_jit(train_agent, device=gpu_devices[args.gpu], donate="all")(data, MODEL, opt_state, train_key)	
	print("train", time.time() - start_time)

	start_time = time.time()
	rew = differentiate(MODEL, env_interaction, key, random_game, helmholtz_game)
	print("diff", time.time() - start_time)
	wandb.log({"loss": loss.tolist(), "# computations": rew[0].tolist()})
	pbar.set_description(f"loss: {loss}, return: {rew}")

