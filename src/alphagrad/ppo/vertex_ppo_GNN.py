"""
PPO implementation with insights from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import os
import argparse
from functools import partial, reduce

import numpy as np

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import jraph
from jraph import GraphsTuple

from tqdm import tqdm
import wandb

import distrax
import optax
import equinox as eqx

from alphagrad.config import setup_experiment
from alphagrad.experiments import make_benchmark_scores
from alphagrad.vertexgame import step, step_sparse
from alphagrad.utils import symlog, symexp, entropy, explained_variance
from alphagrad.transformer.models import PPOModel, PPOModelGNN

from alphagrad.GNN.graph_utils import graph_sparsify

DEBUG = 0

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--task", type=str,
                    default="RoeFlux_1d", help="Name of the task to run.")

parser.add_argument("--gpus", type=str, 
                    default="0", help="GPU ID's to use for training.")

parser.add_argument("--seed", type=int, 
                    default="250197", help="Random seed.")

parser.add_argument("--config_path", type=str, 
                    default=os.path.join(os.getcwd() + "/src/alphagrad/ppo/", "config") if DEBUG else os.path.join(os.getcwd(), "config"), 
                    help="Path to the directory containing the configuration files.")

parser.add_argument("--wandb", type=str,
                    default="run", help="Wandb mode.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
key = jrand.PRNGKey(args.seed)
model_key, init_key, key = jrand.split(key, 3)


config, graph, graph_shape, task_fn = setup_experiment(args.task, args.config_path)
sparse_graph = graph_sparsify(graph)
mM_order, scores = make_benchmark_scores(graph)

sparse_graph = GraphsTuple(
        nodes=jnp.array([sparse_graph.nodes]), # Node features are masks for elimination. (0 not eliminated, 1 eliminated).
        edges=jnp.array([sparse_graph.edges]),
        senders=jnp.array([sparse_graph.senders]),
        receivers=jnp.array([sparse_graph.receivers]),
        n_node=jnp.array([sparse_graph.n_node]),
        n_edge=jnp.array([sparse_graph.n_edge]),
        globals=sparse_graph.globals
    )

parameters = config["hyperparameters"]
ENTROPY_WEIGHT = parameters["entropy_weight"]
VALUE_WEIGHT = parameters["value_weight"]
EPISODES = parameters["episodes"]
NUM_ENVS = parameters["num_envs"]
LR = parameters["lr"]

GAE_LAMBDA = parameters["ppo"]["gae_lambda"]
EPS = parameters["ppo"]["clip_param"]
MINIBATCHES = parameters["ppo"]["num_minibatches"]

ROLLOUT_LENGTH = parameters["ppo"]["rollout_length"]
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = graph.shape[-1] # ROLLOUT_LENGTH # TODO fix this
MINIBATCHSIZE = NUM_ENVS*ROLLOUT_LENGTH//MINIBATCHES

model = PPOModelGNN(
    edge_sparsity_embedding_size=4,
    init_edge_feature_shape=5,
    init_node_feature_shape=1,
    edge_feature_shapes=[32, 64, 64, 64],
    node_feature_shapes=[32, 64, 64, 64],
    num_nodes=sparse_graph.n_node[0],
    key=key
)
"""
Edge feature shape: 5
Node feature shape: 1
Embedded edge sparsity shape: 4
"""

init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))

def init_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    get_biases = lambda m: [x.bias
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x) and x.bias is not None]
    weights = get_weights(model)
    biases = get_biases(model)
    new_weights = [init_fn(subkey, weight.shape)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_biases = [jnp.zeros_like(bias) for bias in biases]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model

# Initialization could help with performance
model = init_weight(model, init_fn, init_key)

run_config = {"seed": args.seed,
                "entropy_weight": ENTROPY_WEIGHT, 
                "value_weight": VALUE_WEIGHT, 
                "lr": LR,
                "episodes": EPISODES, 
                "num_envs": NUM_ENVS, 
                "gae_lambda": GAE_LAMBDA, 
                "eps": EPS, 
                "minibatches": MINIBATCHES, 
                "minibatchsize": MINIBATCHSIZE, 
                "obs_shape": OBS_SHAPE, 
                "num_actions": NUM_ACTIONS, 
                "rollout_length": ROLLOUT_LENGTH, 
                "fwd_fmas": scores[0], 
                "rev_fmas": scores[1], 
                "out_fmas": scores[2]}

wandb.init(project="AlphaGrad")
wandb.run.name = "PPO_" + args.task + "_" + args.name


# Function that normalized the reward
def reward_normalization_fn(reward):
    return symlog(reward) # reward / 364. # 


def inverse_reward_normalization_fn(reward):
    return symexp(reward) # reward * 364. # 


# Definition of some RL metrics for diagnostics
def get_num_clipping_triggers(ratio):
    _ratio = jnp.where(ratio <= 1.+EPS, ratio, 0.)
    _ratio = jnp.where(ratio >= 1.-EPS, 1., 0.)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0))
def get_log_probs_and_value(network, state, action):
    prob_dist, value = network(state)
    log_prob = jnp.log(prob_dist[action] + 1e-7)

    return log_prob, prob_dist, value, entropy(prob_dist)


@jax.jit
@jax.vmap
def get_returns(trajectories):
    rewards = trajectories[2]
    dones = trajectories[3]
    discounts = trajectories[8]
    inputs = jnp.stack([rewards, dones, discounts]).T
    inputs = jnp.squeeze(inputs)

    def loop_fn(episodic_return, traj):
        reward = traj[0]
        done = traj[1]
        discount = traj[2]
        # Simplest advantage estimate
        # The advantage estimate has to be done with the states and actions 
        # sampled from the old policy due to the importance sampling formulation
        # of PPO
        done = 1. - done
        episodic_return = reward + discount*episodic_return*done
        return episodic_return, episodic_return
    
    _, output = lax.scan(loop_fn, 0., inputs[::-1])
    return output[::-1]


# Calculates advantages using generalized advantage estimation
@jax.jit
@jax.vmap
def get_advantages(trajectories):
    rewards = trajectories[2] # (98,1)
    dones = trajectories[3] # (98,1)
    values = trajectories[4] # (98,1)
    next_values = trajectories[6] # (98,1)
    discounts = trajectories[8] # (98,1)
    inputs = jnp.stack([rewards, dones, values, next_values, discounts]).T 
    inputs = jnp.squeeze(inputs) # (98, 5)

    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward = traj[0]
        done = traj[1]
        value = inverse_reward_normalization_fn(traj[2])
        next_value = inverse_reward_normalization_fn(traj[3])
        discount = traj[4]
        # Simplest advantage estimate
        # The advantage estimate has to be done with the states and actions 
        # sampled from the old policy due to the importance sampling formulation
        # of PPO
        done = 1. - done
        episodic_return = reward + discount*episodic_return*done
        delta = reward + next_value*discount*done - value
        advantage = delta + discount*GAE_LAMBDA*lastgaelam*done
        estim_return = advantage + value
        
        next_carry = (episodic_return, advantage)
        new_sample = jnp.array([episodic_return, estim_return, advantage])
            
        return next_carry, new_sample
    
    _, output = lax.scan(loop_fn, (0., 0.), inputs[::-1])
    output = output[::-1]
    split_outs = jnp.split(output, indices_or_sections=3, axis=-1)

    return (*trajectories, *split_outs)
    
    
@jax.jit
def shuffle_and_batch(trajectories, key):
    """
    Shuffles and batches trajectory data for mini-batch training in PPO.

    This function performs three main operations:
    1. Reshapes the trajectory data into a flat array
    2. Randomly shuffles the data along the first axis
    3. Splits the data into minibatches

    Args:
        trajectories: A JAX tree containing trajectory data. Expected to contain:
            - state information (nodes, edges, etc.)
            - actions
            - rewards
            - other trajectory-related data
        key: A JAX PRNGKey for random shuffling

    Returns:
        trajectories: A JAX tree with the same structure as input, but reshaped into
                     MINIBATCHES number of batches, each of size (NUM_ENVS*ROLLOUT_LENGTH//MINIBATCHES)

    Notes:
        - The function assumes global constants NUM_ENVS, ROLLOUT_LENGTH, and MINIBATCHES are defined
        - Uses JAX's tree_map to handle nested structure of trajectory data
        - The function is JIT-compiled for better performance
    """
    size = NUM_ENVS*ROLLOUT_LENGTH//MINIBATCHES
    state = trajectories[0]

    lead_axis_dimension = state.nodes.shape[0] * state.nodes.shape[1]
    shuffled_indices = jrand.permutation(key, jnp.arange(lead_axis_dimension))

    trajectories = jax.tree_map(lambda x: x.reshape(lead_axis_dimension, *x.shape[2:]), trajectories)
    trajectories = jax.tree_map(lambda x: jnp.take(x, shuffled_indices, axis=0), trajectories)
    trajectories = jax.tree_map(lambda x: x.reshape(MINIBATCHES, size, *x.shape[1:]), trajectories)

    return trajectories


def init_carry_sparse(keys):
    # graphs_seq = [sparse_graph for _ in range(len(keys))]
    # graphs_batch1 = jraph.batch(graphs_seq)
    graphs_batch = jax.tree_map(lambda x: jnp.tile(x, (32, *[1 for _ in range(len(x.shape[1:]))])), sparse_graph)
    
    return graphs_batch


# Implementation of the RL algorithm
@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, None, 0, 0))
def rollout_fn(network, rollout_length, init_carry, key):
    keys = jrand.split(key, rollout_length)
    def step_fn(state, key):
        net_key, next_net_key, act_key = jrand.split(key, 3)
        # mask = 1. - state.at[1, 0, :].get()
        mask = 1. - jnp.squeeze(state.nodes)

        prob_dist, state_value_estimate = network(state)
                
        distribution = distrax.Categorical(probs=prob_dist)
        action = distribution.sample(seed=act_key)

        next_state, reward, done = step_sparse(state, action)
        discount = 1.

        next_logits, next_state_value_estimate = network(next_state)
        
        new_sample = (
            state, 
            jnp.array([action]), 
            jnp.array([reward]), 
            jnp.array([done]), 
            state_value_estimate,
            next_state, 
            next_state_value_estimate, 
            prob_dist, 
            jnp.array([discount]))
            
        return next_state, (new_sample)
    scan_out = lax.scan(step_fn, init_carry, keys)
    
    return scan_out


def loss(network, trajectories, keys):
    state = trajectories[0]
    actions = jnp.squeeze(trajectories[1])
    actions = jnp.int32(actions)
    
    rewards = jnp.squeeze(trajectories[2])
    next_state = trajectories[5]
    
    old_prob_dist = trajectories[7]
    discounts = jnp.squeeze(trajectories[8])
    episodic_returns = jnp.squeeze(trajectories[9])
    returns = jnp.squeeze(trajectories[10])
    advantages = jnp.squeeze(trajectories[11])
    
    log_probs, prob_dist, values, entropies = get_log_probs_and_value(network, state, actions)
    _, _, next_values, _ = get_log_probs_and_value(network, next_state, actions)
    norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-7)
    
    # Losses
    old_log_probs = jax.vmap(lambda dist, a: jnp.log(dist[a] + 1e-7))(old_prob_dist, actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    
    num_triggers = get_num_clipping_triggers(ratio)
    trigger_ratio = num_triggers / len(ratio)
    
    clipping_objective = jnp.minimum(ratio*norm_adv, jnp.clip(ratio, 1.-EPS, 1.+EPS)*norm_adv)
    ppo_loss = jnp.mean(-clipping_objective)
    entropy_loss = jnp.mean(entropies)
    value_loss = jnp.mean((values - reward_normalization_fn(returns))**2)
    
    # Metrics
    dV = returns - rewards - discounts*inverse_reward_normalization_fn(next_values) # assess fit quality
    fit_quality = jnp.mean(jnp.abs(dV))
    explained_var = explained_variance(advantages, returns)
    kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), old_prob_dist))
    total_loss = ppo_loss
    total_loss += VALUE_WEIGHT*value_loss
    total_loss -= ENTROPY_WEIGHT*entropy_loss
    return total_loss, [kl_div, entropy_loss, fit_quality, explained_var,ppo_loss, 
                        VALUE_WEIGHT*value_loss, ENTROPY_WEIGHT*entropy_loss, 
                        total_loss, trigger_ratio]
    

@eqx.filter_jit
def train_agent(network, opt_state, trajectories, keys):  
    grads, metrics = eqx.filter_grad(loss, has_aux=True)(network, trajectories, keys)  
    updates, opt_state = optim.update(grads, opt_state)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, metrics


@eqx.filter_jit
def test_agent(network, rollout_length, keys):
    env_carry = init_carry_sparse(keys)
    _, trajectories = rollout_fn(network, rollout_length, env_carry, keys)
    returns = get_returns(trajectories)
    best_return = jnp.max(returns[:, 0], axis=-1)
    idx = jnp.argmax(returns[:, 0], axis=-1)
    best_act_seq = jnp.squeeze(trajectories[1])[idx, :]
    return best_return, best_act_seq, returns[:, 0]


# Define optimizer
schedule = optax.cosine_decay_schedule(LR, EPISODES, 0.) # Works better with scheduler
optim = optax.chain(optax.adam(schedule, b1=.9, eps=1e-7), 
                    optax.clip_by_global_norm(.5))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


# Training loop
pbar = tqdm(range(EPISODES))
samplecounts = 0

env_keys = jrand.split(key, NUM_ENVS)
env_carry = init_carry_sparse(env_keys)
print("Scores:", scores)
best_global_return = jnp.max(-jnp.array(scores))
best_global_act_seq = None

elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

for episode in pbar:
    subkey, key = jrand.split(key, 2)
    keys = jrand.split(key, NUM_ENVS)  
    env_carry = jax.jit(init_carry_sparse)(keys)
    
    env_carry, trajectories = rollout_fn(model, ROLLOUT_LENGTH, env_carry, keys)

    trajectories = get_advantages(trajectories)

    batches = shuffle_and_batch(trajectories, subkey)
    
    # We perform multiple descent steps on a subset of the same trajectory sample
    # This severely increases data efficiency
    # Furthermore, PPO utilizes the 'done' property to continue already running
    # environments
    for i in range(MINIBATCHES):
        subkeys = jrand.split(key, MINIBATCHSIZE)
        batch_ = jax.tree_map(lambda x: x[i], batches)
        model, opt_state, metrics = train_agent(model, opt_state, batch_, subkeys)   
    samplecounts += NUM_ENVS*ROLLOUT_LENGTH
    
    kl_div, policy_entropy, fit_quality, explained_var, ppo_loss, value_loss, entropy_loss, total_loss, clipping_trigger_ratio = metrics
    
    test_keys = jrand.split(key, NUM_ENVS)
    best_return, best_act_seq, returns = test_agent(model, 98, test_keys)
    
    if best_return > best_global_return:
        best_global_return = best_return
        best_global_act_seq = best_act_seq
        print(f"New best return: {best_return}")
        vertex_elimination_order = [int(i) for i in best_act_seq]
        print(f"New best action sequence: {vertex_elimination_order}")
        elim_order_table.add_data(episode, best_return, np.array(best_act_seq))
    
    # Tracking different RL metrics
    wandb.log({"best_return": best_return,
               "mean_return": jnp.mean(returns),
                "KL divergence": kl_div,
                "entropy evolution": policy_entropy,
                "value function fit quality": fit_quality,
                "explained variance": explained_var,
                "sample count": samplecounts,
                "ppo loss": ppo_loss,
                "value loss": value_loss,
                "entropy loss": entropy_loss,
                "total loss": total_loss,
                "clipping trigger ratio": clipping_trigger_ratio})
        
    pbar.set_description(f"entropy: {policy_entropy:.4f}, best_return: {best_return}, mean_return: {jnp.mean(returns)}, fit_quality: {fit_quality:.2f}, expl_var: {explained_var:.4}, kl_div: {kl_div:.4f}")
        
wandb.log({"Elimination order": elim_order_table})
vertex_elimination_order = [int(i) for i in best_act_seq]
print(f"Best vertex elimination sequence after {EPISODES} episodes is {vertex_elimination_order} with {best_global_return} multiplications.")

