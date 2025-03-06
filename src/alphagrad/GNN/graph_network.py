import jax
import jraph as jr
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import equinox as eqx
from equinox import AbstractVar
from typing import Callable, Sequence
from abc import ABC, abstractmethod

from alphagrad.GNN.graph_utils import apply_edge_sparsity_embedding, apply_node_op_type_embedding

# Abstract base class for a graph network layer
class graphNetworkLayer(eqx.Module, ABC):

    @abstractmethod
    def layer_fn(self):
        pass

    @abstractmethod
    def __call__(self, graph: jr.GraphsTuple) -> jr.GraphsTuple:
        pass


class graphNetwork(eqx.Module, ABC):
    '''
    Works as an Actor and outputs which node to eliminate next as a mask over all nodes.
    '''

    @abstractmethod
    def __call__(self, graph: jr.GraphsTuple) -> jnp.ndarray:
        '''
        Args: 
            graph (jr.GraphsTuple): A graph represented as a GraphsTuple object whose nodes will be eliminated.

        Returns:
            jax.numpy.ndarray: Mask of the nodes that will be eliminated.
        '''
        pass


# Implementation of EdgeGATLayer:
class EdgeGATLayer(graphNetworkLayer):
    edge_update: eqx.nn.Linear
    node_update: eqx.nn.Linear
    # attention_logit: eqx.nn.Linear

    """
    1) For each edge, compute attention scores 
    """

    def __init__(self, edge_feature_size, node_feature_size, transformed_edge_feature_size, transformed_node_feature_size, keys: jrand.PRNGKey):

        edge_key, node_key, attn_key = jrand.split(keys, 3)

        aggregated_edge_feature_size = edge_feature_size + 2 * node_feature_size # since we are concatenating the sender and receiver node features with current edge feature.
        aggregated_node_feature_size = node_feature_size + 2 * transformed_edge_feature_size
        # attention_logit_in_feature_size = transformed_edge_feature_size + 2 * transformed_node_feature_size
        self.edge_update = eqx.nn.Linear(aggregated_edge_feature_size, transformed_edge_feature_size, key=edge_key)
        self.node_update = eqx.nn.Linear(aggregated_node_feature_size, transformed_node_feature_size, key=node_key)
        # self.attention_logit = eqx.nn.Linear(attention_logit_in_feature_size, 1, key=attn_key)

    # Preprocess the edge features to create new feature embeddings. 
    # These will be used to calculate the attention scores.
    def update_edge_fn(self, edge_features, sender_node_features, receiver_node_features, global_features):
        aggregated_features = jnp.concatenate((edge_features, sender_node_features, receiver_node_features), axis=-1) # batch of aggregated features
        # vectorize for each edge:
        out = jax.vmap(self.edge_update)(aggregated_features)
        updated_edge_features = jnn.gelu(out)

        return updated_edge_features
    
    def update_node_fn(self, current_node_features, outgoing_edge_features, incoming_edge_features, global_features):
        aggregated_features = jnp.concatenate((current_node_features, outgoing_edge_features, incoming_edge_features), axis=-1) # batch of aggregated features
        # vectorize for each node:
        out = jax.vmap(self.node_update)(aggregated_features)
        updated_node_features = jnn.gelu(out)

        return updated_node_features
    
    def attention_logit_fn(self, updated_edge_features, updated_sender_node_features, updated_receiver_node_features, global_features):
        # aggregated_features = jnp.concatenate((updated_edge_features, updated_sender_node_features, updated_receiver_node_features), axis=-1) # batch of aggregated features
        # vectorize for each edge:
        # out = jax.vmap(self.attention_logit)(aggregated_features)

        # Attention scores for edges are computed using the nodes connected to that edge. 
        # If node i features are more important to node j features, the edge between i and j 
        # will have a higher attention score.

        attention_logits = jax.vmap(lambda k, v:  jnp.dot(k, v) / jnp.sqrt(len(k)), in_axes=(0, 0))(updated_sender_node_features, updated_receiver_node_features)

        return attention_logits
    
    
    def attention_reduce_fn(self, edge_features, attention_weights):
        # Output is 
        weighted_edge_features = jax.vmap(lambda k, v: k * v, in_axes=(0, 0))(edge_features, attention_weights)

        return weighted_edge_features
    
    def layer_fn(self):
        gat_layer = jr.GraphNetGAT(
            update_edge_fn=self.update_edge_fn,
            update_node_fn=self.update_node_fn,
            attention_logit_fn=self.attention_logit_fn,
            attention_reduce_fn=self.attention_reduce_fn
        )

        return gat_layer

    def __call__(self, graph: jr.GraphsTuple) -> jr.GraphsTuple:
        
        return self.layer_fn()(graph)
        


class EdgeGATNetwork(graphNetwork):
    sparsity_embedding: eqx.nn.Embedding
    gat_layers: Sequence[EdgeGATLayer]
    output_layer: eqx.nn.Linear
    
    edge_feature_shapes: Sequence[int] = eqx.field(static=True)
    node_feature_shapes: Sequence[int] = eqx.field(static=True)

    def __init__(
            self,
            edge_sparsity_embedding_size: int,
            init_edge_feature_shape: int,
            init_node_feature_shape: int,
            edge_feature_shapes: Sequence[int],
            node_feature_shapes: Sequence[int],
            key: jrand.PRNGKey
    ):

        self.edge_feature_shapes = edge_feature_shapes
        self.node_feature_shapes = node_feature_shapes

        embed_key_1, embed_key_2, out_key = jrand.split(key, 3)

        self.sparsity_embedding = eqx.nn.Embedding(21, edge_sparsity_embedding_size, key=embed_key_1)
        self.output_layer = eqx.nn.Linear(node_feature_shapes[-1], 1, key=out_key) # to generate the mask

        keys = jrand.split(key, len(edge_feature_shapes)+1)
        self.gat_layers = []

        first_layer_edge_feature_shape = edge_sparsity_embedding_size + init_edge_feature_shape - 1

        self.gat_layers.append(
            EdgeGATLayer(first_layer_edge_feature_shape, init_node_feature_shape, edge_feature_shapes[0], node_feature_shapes[0], keys[0])
        )

        for i in range(1, len(edge_feature_shapes)):
            self.gat_layers.append(EdgeGATLayer(edge_feature_shapes[i-1], node_feature_shapes[i-1], edge_feature_shapes[i], node_feature_shapes[i], keys[i]))

        return
    
    def apply_sparsity_embeddings(self, graph: jr.GraphsTuple) -> jr.GraphsTuple:
        return apply_edge_sparsity_embedding(self.sparsity_embedding, graph)
    
    @jax.jit
    def __call__(self, graph: jr.GraphsTuple) -> jnp.ndarray:


        
        if graph.nodes.ndim == 1:
            graph = graph._replace(nodes=jnp.expand_dims(graph.nodes, axis=-1))
        mask = graph.nodes
        graph = self.apply_sparsity_embeddings(graph)

        for layer_ in self.gat_layers:
            graph = layer_(graph)

        nodes = jax.vmap(self.output_layer)(graph.nodes)
        nodes = jnp.squeeze(nodes)

        return jnp.where(jnp.squeeze(mask) > 0., -jnp.inf, nodes) # Already eliminated nodes are -jnp.inf so that after softmax they have 0 elimination probability.
