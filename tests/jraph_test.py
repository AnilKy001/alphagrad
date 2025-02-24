from typing import Tuple
import time
import unittest
from parameterized import parameterized
import inspect

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
from graphax.examples import (RobotArm_6DOF, RoeFlux_1d, f, Perceptron, 
                            Simple, Lighthouse, Hole, Helmholtz)

from alphagrad.GNN.graph_network import EdgeGATNetwork
from alphagrad.GNN.graph_utils import graph_sparsify, reverse, add_self_edges_fn

jax.config.update("jax_disable_jit", True)

# F = RobotArm_6DOF

test_funcs = [RobotArm_6DOF, RoeFlux_1d, f, Perceptron, Simple, Lighthouse, Hole]

def func_test(F):

    print("Function:", F.__name__)

    sig = inspect.signature(F)
    num_params = len(sig.parameters)

    xs = [jnp.zeros((1,))]*num_params

    args = range(len(xs))
    dense_graph = make_graph(F, *xs)

    jaxpr = jax.make_jaxpr(jacve(F, order="rev", argnums=args, count_ops=True))(*xs)
    deriv_jaxpr = jax.make_jaxpr(jacve(F, order="rev", argnums=args, count_ops=True))(*xs)

    jacobian, aux = jax.jit(jacve(F, order="rev", argnums=args, count_ops=True, ))(*xs)
    print("num muls:", aux["num_muls"], "num_adds:", aux["num_adds"])
    print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

    sparse_graph = graph_sparsify(dense_graph)
    sparse_graph = add_self_edges_fn(sparse_graph)

    # TODO the sparse version of vertex elimination does not yield the correct number
    # of multiplications and additions

    start = time.time()
    out_graph, out = jax.jit(reverse)(sparse_graph)
    end = time.time()
    sparse_num_muls = int(out_graph.globals[0][0])
    # print("jraph reverse time jit", end-start, out_graph.globals)

    # print([(int(i), int(j)) for i, j in zip(out[0], out[1])])

    # key = jrand.PRNGKey(123)
    # print("embedding takes time")
    # dense_graph = old_embed(key, dense_graph, [20, 150, 20])
    start = time.time()
    out_graph, nops = jax.jit(dense_reverse)(dense_graph)
    end = time.time()
    dense_num_muls = nops
    # print("alphagrad time jit", end-start, nops)


    key = jrand.PRNGKey(42)
    edge_gat_net = EdgeGATNetwork(4, (32, 32, 32), (16, 16, 16), key)

    sparse_graph = sparse_graph._replace(
        nodes=jnp.astype(sparse_graph.nodes[:, jnp.newaxis], jnp.int32),
        edges=jnp.astype(sparse_graph.edges, jnp.int32),
        senders=jnp.astype(sparse_graph.senders, jnp.int32),
        receivers=jnp.astype(sparse_graph.receivers, jnp.int32),
    )

    logits = edge_gat_net(sparse_graph)
    # print(jnn.softmax(logits))

    print("Sparse num muls: ", sparse_num_muls, "Dense num muls:", dense_num_muls)

    return sparse_num_muls, dense_num_muls


class testMathOperations(unittest.TestCase):

    @parameterized.expand([(F,) for F in test_funcs])
    def test_num_muls(self, F):
        sparse_num_muls, dense_mul_muls = func_test(F)
        self.assertEqual(sparse_num_muls, dense_mul_muls)

if __name__ == "__main__":
    unittest.main()