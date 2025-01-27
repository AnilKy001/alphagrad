import jax
import jraph as jr
import jax.numpy as jnp
from typing import Callable


def graph_sparsify(dense_graph) -> jr.GraphsTuple:

    # Each vertex has an integer as its feature. 
    # Integer is the mask of the vertex for elimination.
    # 0 means that the vertex is not eliminated.
    # 1 means that the vertex is eliminated.
    out_nodes = jnp.where(dense_graph[1, 0, :] > 0, 1, 0)
    num_input_nodes = dense_graph.shape[1] - dense_graph.shape[2] - 1
    in_nodes = jnp.ones(num_input_nodes)
    node_features = jnp.concatenate([in_nodes, out_nodes])

    edge_positions = jnp.nonzero(dense_graph[1, 1:])
    edge_features = []
    for i, j in zip(edge_positions[0], edge_positions[1]):
        edge_features.append(dense_graph[:, i+1, j])
    edge_features = jnp.stack(edge_features)

    sparse_graph = jr.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        n_node=len(node_features),
        n_edge=len(edge_positions[0]),
        globals=jnp.array([[0.]])
    )

    return sparse_graph

# Embeds the edge sparsity type.
def apply_edge_sparsity_embedding(sparsity_embedding_fn: Callable, graph: jr.GraphsTuple) -> jr.GraphsTuple:
    sparsity_embeddings = jax.vmap(sparsity_embedding_fn)(11 + graph.edges[:, 0])
    edges_with_embedded_sparsity = jnp.concatenate((sparsity_embeddings, graph.edges[:, 1:]), axis=-1)
    return graph._replace(edges=edges_with_embedded_sparsity)

# Embeds whether a node is an input/output node or not.
def apply_node_op_type_embedding(op_type_embedding_fn: Callable, graph: jr.GraphsTuple) -> jr.GraphsTuple:
    nodes_with_embedded_op_type = jax.vmap(op_type_embedding_fn)(graph.nodes[:, 0])
    return graph._replace(nodes=nodes_with_embedded_op_type)