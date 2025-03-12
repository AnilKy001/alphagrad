from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey
import equinox as eqx
from equinox import static_field

from jraph import GraphsTuple

from alphagrad.transformer import MLP
from alphagrad.transformer import Encoder
from alphagrad.transformer import PositionalEncoder
from alphagrad.GNN.graph_network import EdgeGATNetwork


class GraphEmbedding(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    
    def __init__(self, 
                graph_shape: Sequence[int],
                embedding_dim: int,
                key: PRNGKey = None,
                **kwargs) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key = jrand.split(key, 3)
        kernel_size, stride = 1, 1
        self.embedding = eqx.nn.Conv2d(num_vo, num_vo, (5, kernel_size), 
                                        stride=(1, stride), key=embed_key)
        conv_size = (num_i+num_vo-kernel_size) // stride+1
        self.projection = jrand.normal(proj_key, (conv_size, embedding_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
    
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        output_mask = graph.at[2, 0, :].get()
        vertex_mask = graph.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(graph.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = graph.at[:, 1:, :].get() #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        return embeddings.T, attn_mask.T


class SequentialTransformer(eqx.Module):
    num_heads: int
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_enc: Encoder
    policy_head: MLP
    value_head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()
    
    def __init__(self, 
                in_dim: int,
                seq_len: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int = 1024,
                num_layers_policy: int = 2,
                policy_ff_dims: Sequence[int] = [512, 256],
                value_ff_dims: Sequence[int] = [1024, 512],
                key: PRNGKey = None) -> None:
        super().__init__()  
        self.num_heads = num_heads      
        e_key, p_key, pe_key, v_key, t_key = jrand.split(key, 5)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(in_dim, seq_len)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=e_key)
        
        self.policy_enc = Encoder(num_layers=num_layers_policy,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=pe_key)
        
        # self.global_token = jrand.normal(t_key, (in_dim, 1))
        # self.global_token_mask_x = jnp.ones((seq_len, 1))
        # self.global_token_mask_y = jnp.ones((1, seq_len+1))
        self.policy_head = MLP(in_dim, 1, policy_ff_dims, key=p_key)
        self.value_head = MLP(in_dim, 1, value_ff_dims, key=v_key)
        
        
    def __call__(self, xs: Array, mask: Array = None, key: PRNGKey = None) -> Array:
        e_key, p_key = jrand.split(key, 2)
            
        # Add global token to input
        # xs = jnp.concatenate((self.global_token, xs), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)
        
        # Transpose inputs for equinox attention mechanism
        xs = self.pos_enc(xs).T

        # Replicate mask and apply encoder
        if mask is not None:
            mask = mask.T
            mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
            xs = self.encoder(xs, mask=mask, key=e_key)
            policy_embedding = xs # self.policy_enc(xs, mask=mask, key=p_key)
        else: 
            xs = self.encoder(xs, mask=None, key=e_key)
            policy_embedding = xs # self.policy_enc(xs, mask=None, key=p_key)
        # global_token_xs = xs[0]
        values = jax.vmap(self.value_head)(xs)
        value = jnp.mean(values)
        
        policy = jax.vmap(self.policy_head)(xs)
        return jnp.concatenate((jnp.array([value]), policy.squeeze()))


class PPOModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer
    
    def __init__(self, 
                graph_shape: Sequence[int],
                embedding_dim: int,
                num_layers: int,
                num_heads: int,
                key: PRNGKey = None,
                **kwargs) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        self.embedding = eqx.nn.Conv2d(num_vo, num_vo, (5, 1), use_bias=False, key=embed_key)
        self.projection = eqx.nn.Linear(num_i+num_vo, embedding_dim, use_bias=False, key=proj_key)
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(embedding_dim,
                                                num_vo, 
                                                num_layers, 
                                                num_heads, 
                                                key=tf_key, 
                                                **kwargs)
    
    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get() # + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        # embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        embeddings = jax.vmap(self.projection, in_axes=0)(embeddings)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)


class PPOModelGNN(eqx.Module):    
    """
    Generates state value estimate and policy logits by using GNNs as policy 
    and value networks.
    """

    GAT_net: eqx.Module
    value_net: eqx.nn.Sequential
    policy_net: eqx.nn.Sequential
    
    def __init__(
            self,
            edge_sparsity_embedding_size: int,
            init_edge_feature_shape: int,
            init_node_feature_shape: int,
            edge_feature_shapes: Sequence[int],
            node_feature_shapes: Sequence[int],
            num_nodes: int,
            key: PRNGKey
    ) -> None:
        super().__init__()
        policy_key, value_key = jrand.split(key, 2)
        pol_keys = jrand.split(policy_key, 5)
        val_keys = jrand.split(value_key, 5)

        self.GAT_net = EdgeGATNetwork(
                edge_sparsity_embedding_size, 
                init_edge_feature_shape, 
                init_node_feature_shape, 
                edge_feature_shapes, 
                node_feature_shapes, 
                policy_key
        )

        out_node_feature_shape = node_feature_shapes[-1]

        ravel_node_features_shape = num_nodes * out_node_feature_shape

        self.policy_net = eqx.nn.Sequential([
            eqx.nn.Linear(out_node_feature_shape, 1024, key=pol_keys[0]),
            eqx.nn.Linear(1024, 512, key=pol_keys[1]),
            eqx.nn.Linear(512, 256, key=pol_keys[2]),
            eqx.nn.Linear(256, 128, key=pol_keys[3]),
            eqx.nn.Linear(128, 1, key=pol_keys[4])
        ])

        self.value_net = eqx.nn.Sequential([
            eqx.nn.Linear(ravel_node_features_shape, 1024, key=val_keys[0]),
            eqx.nn.Linear(1024, 512, key=val_keys[1]),
            eqx.nn.Linear(512, 256, key=val_keys[2]),
            eqx.nn.Linear(256, 128, key=val_keys[3]),
            eqx.nn.Linear(128, 1, key=val_keys[4])
        ])

    
    def __call__(self, sparse_graph: GraphsTuple) -> Array:

        elimination_mask = sparse_graph.nodes
        elimination_mask = 1. - jnp.squeeze(elimination_mask)

        node_features = self.GAT_net(sparse_graph)

        node_elimination_vals = jax.vmap(self.policy_net, in_axes=(0))(node_features)
        node_elimination_vals = jnp.squeeze(node_elimination_vals)
        node_elimination_logits = jnp.where(elimination_mask == 0., -jnp.inf, node_elimination_vals)
        action_prob_dist = jnn.softmax(node_elimination_logits, axis=0)


        ravel_node_features = node_features.ravel()
        state_value_estimate = self.value_net(ravel_node_features)

        return action_prob_dist, state_value_estimate

class AlphaZeroModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer
    
    def __init__(self, 
                graph_shape: Sequence[int],
                embedding_dim: int,
                num_layers: int,
                num_heads: int,
                key: PRNGKey = None,
                **kwargs) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        kernel_size, stride = 3, 2
        self.embedding = eqx.nn.Conv2d(num_vo, num_vo, (5, kernel_size), 
                                        stride=(1, stride), key=embed_key)
        conv_size = (num_i+num_vo-kernel_size) // stride+1
        self.projection = jrand.normal(proj_key, (conv_size, embedding_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(embedding_dim,
                                                num_vo, 
                                                num_layers, 
                                                num_heads, 
                                                key=tf_key, 
                                                **kwargs)
    
    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get() #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)
    

class PolicyNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP
    
    def __init__(self, 
                graph_shape: Sequence[int],
                in_dim: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int = 1024,
                mlp_dims: Sequence[int] = [512, 256],
                key: PRNGKey = None) -> None:
        super().__init__()     
        num_i, num_vo, num_o = graph_shape
        self.num_heads = num_heads
        encoder_key, embed_key, key = jrand.split(key, 3)
        self.embedding = GraphEmbedding(graph_shape, in_dim, key=embed_key)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(in_dim, num_vo)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=encoder_key)

        self.head = MLP(in_dim, 1, mlp_dims, key=key)
        
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:  
        # Embed the input graph
        embeddings, mask = self.embedding(graph)
        
        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T
        
        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)
        
        policy = jax.vmap(self.head)(xs)
        return policy.squeeze()


class ValueNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()
    
    def __init__(self, 
                graph_shape: Sequence[int],
                in_dim: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int = 1024,
                mlp_dims: Sequence[int] = [1024, 512],
                key: PRNGKey = None) -> None:
        super().__init__()    
        num_i, num_vo, num_o = graph_shape  
        self.num_heads = num_heads 
        embedding_key, encoder_key, token_key, key = jrand.split(key, 4)
        self.embedding = GraphEmbedding(graph_shape, in_dim, key=embedding_key)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(in_dim, num_vo)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=encoder_key)
        
        # self.global_token = jrand.normal(token_key, (in_dim, 1))
        # self.global_token_mask_x = jnp.ones((num_vo, 1))
        # self.global_token_mask_y = jnp.ones((1, num_vo+1))
        self.head = MLP(in_dim, 1, mlp_dims, key=key)
        
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        # Embed the input graph
        embeddings, mask = self.embedding(graph)
        
        # Add global token to input
        # embeddings = jnp.concatenate((self.global_token, embeddings), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)
        
        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T
        
        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)
        values = jax.vmap(self.head)(xs)
        
        return jnp.mean(values)
    
    
# TODO test ResNet

