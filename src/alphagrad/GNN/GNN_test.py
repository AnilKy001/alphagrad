from typing import Tuple
import time
import unittest
from parameterized import parameterized
import inspect

import equinox as eqx
import optax

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn

from alphagrad.vertexgame import make_graph
from alphagrad.vertexgame import reverse as dense_reverse
from alphagrad.vertexgame.transforms import embed as dense_embed

from graphax import jacve
from graphax.sparse.utils import count_muls_jaxpr
from graphax.examples import RobotArm_6DOF

from alphagrad.GNN.graph_network import EdgeGATNetwork
from alphagrad.GNN.graph_utils import graph_sparsify, reverse, add_self_edges_fn

from alphagrad.transformer.models import PPOModelGNN

# jax.config.update("jax_disable_jit", True)

key_ = jrand.PRNGKey(42)
key1, key2 , key_normal= jrand.split(key_, 3)

F = RobotArm_6DOF

sig = inspect.signature(F)
num_params = len(sig.parameters)

xs = [jnp.zeros((1,))]*num_params

args = range(len(xs))
dense_graph = make_graph(F, *xs)

sparse_graph = graph_sparsify(dense_graph)
sparse_graph = add_self_edges_fn(sparse_graph)

model = PPOModelGNN(4, 5, 1, [16, 16], [16, 16], sparse_graph.n_node, key1)


target_dist = jax.nn.softmax(jrand.normal(key_normal, (120,)))

def loss_fn(model):
    prob_dist, value = model(sparse_graph)
    loss = jnp.sum(prob_dist - target_dist)
    loss += value[0]
    return loss

for i in range(100):
    grads = eqx.filter_grad(loss_fn)(model)
    print(jax.tree_util.tree_leaves(grads))

print("Done")