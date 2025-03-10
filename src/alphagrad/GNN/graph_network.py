import jax
import jraph as jr
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import equinox as eqx
from equinox import AbstractVar
from typing import Callable, Sequence
from abc import ABC, abstractmethod
from chex import PRNGKey

from alphagrad.GNN.graph_utils import apply_edge_sparsity_embedding, apply_node_op_type_embedding

# Implementation of EdgeGATLayer:
class EdgeGATLayer(eqx.Module):
    edge_update: eqx.nn.Linear
    node_update: eqx.nn.Linear
    attention_logit: eqx.nn.Linear
    # attention_logit: eqx.nn.Linear

    """
    1) For each edge, compute attention scores 
    """

    def __init__(self, edge_feature_size, node_feature_size, transformed_edge_feature_size, transformed_node_feature_size, keys: jrand.PRNGKey):
        super().__init__()

        edge_key, node_key, attn_key = jrand.split(keys, 3)
        aggregated_edge_feature_size = edge_feature_size + 2 * node_feature_size # since we are concatenating the sender and receiver node features with current edge feature.
        aggregated_node_feature_size = node_feature_size + 2 * transformed_edge_feature_size
        attention_logit_in_feature_size = 2 * node_feature_size
        self.edge_update = eqx.nn.Linear(aggregated_edge_feature_size, transformed_edge_feature_size, key=edge_key)
        self.node_update = eqx.nn.Linear(aggregated_node_feature_size, transformed_node_feature_size, key=node_key)
        # self.attention_logit = lambda k, v:  jnp.dot(k, v) / jnp.max(jnp.array([1., jnp.sqrt(len(k))]))
        self.attention_logit = eqx.nn.Linear(attention_logit_in_feature_size, 1, key=attn_key)
    
    """
    The following functions are executed in the order:
        1) update_edge_fn
        2) attention_logit_fn
        3) attention_reduce_fn
        4) update_node_fn
    """

    # Preprocess the edge features to create new feature embeddings. 
    # These will be used to calculate the attention scores.
    def update_edge_fn(self, edge_features, sender_node_features, receiver_node_features, global_features=None):
        aggregated_features = jnp.concatenate((edge_features, sender_node_features, receiver_node_features), axis=-1) # batch of aggregated features
        # vectorize for each edge:
        out_fn = jax.vmap(lambda x: self.edge_update(x))
        out = out_fn(aggregated_features)
        updated_edge_features = jnn.relu(out)

        return updated_edge_features
    
    def update_node_fn(self, current_node_features, updated_outgoing_edge_features, updated_incoming_edge_features, global_features=None):
        aggregated_features = jnp.concatenate((current_node_features, updated_outgoing_edge_features, updated_incoming_edge_features), axis=-1) # batch of aggregated features
        # vectorize for each node:
        out_fn = jax.vmap(lambda x: self.node_update(x))
        out = out_fn(aggregated_features)
        updated_node_features = jnn.relu(out)

        return updated_node_features
    
    def attention_logit_fn(self, updated_edge_features, sender_node_features, receiver_node_features, global_features=None):
        # Attention scores for edges are computed using the nodes connected to that edge. 
        # If node i features are more important to node j features, the edge between i and j 
        # will have a higher attention score.
        features = jnp.concatenate((sender_node_features, receiver_node_features), axis=-1)
        logit_fn = jax.vmap(lambda x: self.attention_logit(x))
        attention_logits = logit_fn(features)

        return attention_logits
    
    
    def attention_reduce_fn(self, edge_features, attention_weights):
        # Output is 
        weighted_edge_features = jax.vmap(lambda k, v: k * v, in_axes=(0, 0))(edge_features, attention_weights)

        return weighted_edge_features


    def __call__(self, graph: jr.GraphsTuple, key: PRNGKey) -> jr.GraphsTuple:
        
        return jr.GraphNetGAT(
            update_edge_fn=self.update_edge_fn,
            update_node_fn=self.update_node_fn,
            attention_logit_fn=self.attention_logit_fn,
            attention_reduce_fn=self.attention_reduce_fn
        )(graph)
        


class EdgeGATNetwork(eqx.Module):
    sparsity_embedding: eqx.nn.Embedding
    gat_layers: eqx.nn.Sequential

    def __init__(
            self,
            edge_sparsity_embedding_size: int,
            init_edge_feature_shape: int,
            init_node_feature_shape: int,
            edge_feature_shapes: Sequence[int],
            node_feature_shapes: Sequence[int],
            key: jrand.PRNGKey
    ):
        
        super().__init__()

        embed_key_1, gat_key, out_key = jrand.split(key, 3)

        self.sparsity_embedding = eqx.nn.Embedding(21, edge_sparsity_embedding_size, key=embed_key_1)
        gat_keys = jrand.split(gat_key, len(edge_feature_shapes)+1)
        first_layer_edge_feature_shape = edge_sparsity_embedding_size + init_edge_feature_shape - 1

        self.gat_layers = eqx.nn.Sequential([
            EdgeGATLayer(first_layer_edge_feature_shape, init_node_feature_shape, edge_feature_shapes[0], node_feature_shapes[0], gat_keys[0]),
            *[EdgeGATLayer(edge_feature_shapes[i], node_feature_shapes[i], edge_feature_shapes[i+1], node_feature_shapes[i+1], gat_keys[i+1]) 
            for i in range(len(edge_feature_shapes)-1)]
        ])

        return
    
    def apply_sparsity_embeddings(self, graph: jr.GraphsTuple) -> jr.GraphsTuple:
        return apply_edge_sparsity_embedding(self.sparsity_embedding, graph)
    
    @eqx.filter_jit
    def __call__(self, graph: jr.GraphsTuple) -> jnp.ndarray:
        """Forward pass of the GNN.
        Args:
            graph: GraphsTuple containing the graph data.
        Returns:
            nodes: Elimination probabilities for each node.
        """
        if graph.nodes.ndim == 1:
            graph = graph._replace(nodes=jnp.expand_dims(graph.nodes, axis=-1))

        graph = self.apply_sparsity_embeddings(graph)

        graph = self.gat_layers(graph)

        node_features = jnp.squeeze(graph.nodes)

        return node_features
