import jax
from jraph import GraphsTuple
import jax.numpy as jnp
import jax.lax as lax

from typing import Callable, Tuple, Sequence
from chex import Array

from alphagrad.vertexgame.core import ADD_SPARSITY_MAP, MUL_SPARSITY_MAP, CONTRACTION_MAP

# The size of these buffers are the main bottleneck of the algorithm.
IN_VAL_BUFFER_SIZE = 10
OUT_VAL_BUFFER_SIZE = 10

NUM_SPARSITY_TYPES = 21
OFFSET = (NUM_SPARSITY_TYPES - 1) // 2

def graph_sparsify(dense_graph: Array) -> GraphsTuple:
    """
    GNNs operate on a sparse graph representation. This representation is 
    a tuple that stores node/edge features and sender/receiver indices.
    This function converts a dense graph representation to the sparse one.
    """

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

    sparse_graph = GraphsTuple(
        nodes=node_features, # Node features are masks for elimination. (0 not eliminated, 1 eliminated).
        edges=edge_features,
        senders=edge_positions[0],
        receivers=edge_positions[1] + num_input_nodes,
        n_node=jnp.array([len(node_features)]),
        n_edge=jnp.array([len(edge_positions[0])]),
        globals=jnp.array([[0.]])
    )

    return sparse_graph


def add_self_edges_fn(graph: GraphsTuple) -> GraphsTuple:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    total_num_nodes = graph.n_node[0]
    graph._replace(receivers = jnp.concatenate((graph.receivers, jnp.arange(total_num_nodes)), axis=0))
    graph._replace(senders = jnp.concatenate((graph.senders, jnp.arange(total_num_nodes)), axis=0))

    return graph

# Embeds the edge sparsity type.
def apply_edge_sparsity_embedding(sparsity_embedding_fn: Callable, graph: GraphsTuple) -> GraphsTuple:
    sparsity_embeddings = jax.vmap(sparsity_embedding_fn)(11 + graph.edges[:, 0])
    edges_with_embedded_sparsity = jnp.concatenate((sparsity_embeddings, graph.edges[:, 1:]), axis=-1)

    return graph._replace(edges=edges_with_embedded_sparsity)

# Embeds whether a node is an input/output node or not.
def apply_node_op_type_embedding(op_type_embedding_fn: Callable, graph: GraphsTuple) -> GraphsTuple:
    nodes_with_embedded_op_type = jax.vmap(op_type_embedding_fn)(graph.nodes[:, 0])

    return graph._replace(nodes=nodes_with_embedded_op_type)



def sparse_mul(in_jac: Array, out_jac: Array) -> Tuple[float, Array]:
    """
    Function that computes the shape of the resulting sparse multiplication of 
    the Jacobian of the incoming edge and the Jacobian of the outgoing edge.
    It also computes the number of necessary multiplications to do so.

    Args:
        in_jac (Array): Sparsity type and Jacobian shape of the incoming edge.
        out_jac (Array): Sparsity type and Jacobian shape of the outgoing edge.

    Returns:
        Tuple: Tuple containing the sparsity type and Jacobian shape of the 
                resulting edge as well as the number of multiplications.
    """
    # Get the sparsity type of the incoming and outgoing edge and compute the
    # sparsity type of the resulting edge
    in_sparsity = in_jac[0].astype(jnp.int32)
    out_sparsity = out_jac[0].astype(jnp.int32)
    res_sparsity = jnp.array([MUL_SPARSITY_MAP[in_sparsity + OFFSET, out_sparsity + OFFSET]])
    # print("in_sparsity: ", in_sparsity)
    # print("out_sparsity: ", out_sparsity)
    # print("res_sparsity: ", res_sparsity)  
    
    # Check how the contraction between incoming and outgoing Jacobian is done
    # and compute the resulting number of multiplications
    contraction_map = CONTRACTION_MAP[:, in_sparsity + OFFSET, out_sparsity + OFFSET]
    factors = jnp.concatenate((out_jac[1:3], jnp.abs(out_jac[3:]), in_jac[3:]))
    masked_factors = lax.cond(jnp.sum(contraction_map) > 0,
                                lambda a: jnp.where(contraction_map > 0, a, 1),
                                lambda a: jnp.zeros_like(a), 
                                factors)

    fmas = jnp.prod(masked_factors)
    return fmas, jnp.concatenate([res_sparsity, out_jac[1:3], in_jac[3:]])


def sparse_add(in_jac: Array, out_jac: Array) -> Array:
    """
    Function that computes the shape of the resulting sparse addition of the
    Jacobian of the incoming edge and the Jacobian of the outgoing edge.

    Args:
        in_jac (Array): Sparse type and Jacobian shape of the incoming edge.
        out_jac (Array): Sparse type and Jacobian shape of the outgoing edge.

    Returns:
        Array: Sparse type and Jacobian shape of the resulting edge.
    """
    in_sparsity = in_jac[0].astype(jnp.int32)
    out_sparsity = out_jac[0].astype(jnp.int32)
    res_sparsity = jnp.array([ADD_SPARSITY_MAP[in_sparsity + OFFSET, out_sparsity + OFFSET]])
    return jnp.concatenate([res_sparsity, in_jac[1:]])
    


def del_and_copy_edge(n: int, 
                        i: int, 
                        pos_buf: Array, 
                        jacs_buf: Array, 
                        edge_conn: Array, 
                        edge_vals: Array) -> Tuple:
    """
    Function that deletes the respective edge at position `i` from `edge_conn`
    and `edge_vals` and copies the edge into the buffers `pos_buf` and `jacs_buf`.
    Furthermore, it deletes the the edge from `edge_conn` and the value of the 
    edge from `edge_vals`.

    Args:
        n (int): Global counter variable that track where we are in the sparse
                representation of our graph.
        i (Array): Global counter variable that tracks the index of the buffer.
        pos_buf (Array): Buffer that stores the connectivity of the edge in question.
        jacs_buf (Array): Buffer that stores the value of the edge in question.
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.

    Returns:
        Tuple: A tuple containing the updated counter variables, buffers and
                state of the computational graph.
    """
    # Fill the position and value buffers with the edge we want to delete
    pos_buf = pos_buf.at[i, :].set(edge_conn[n]) # add edge to delete to buffer
    jacs_buf = jacs_buf.at[i, :].set(edge_vals[n]) # add features of edge to delete to buffer
    
    # Delete the edge from the graph representation
    edge_conn = edge_conn.at[n].set(-1) # delete edge from graph
    edge_vals = edge_vals.at[n].set(0) # delete edge features from graph
    
    return (i+1, pos_buf, jacs_buf, edge_conn, edge_vals)


def cond(condition, true_fn, false_fn, *xs):
    if condition:
        return true_fn(*xs)
    else:
        return false_fn(*xs)
    
    
def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)
        

def get_edges(vertex: int, edge_conn: Array, edge_vals: Array) -> Tuple:    
    """
    Function that iterates through the sparse representation of the computational
    graph and looks for edges connected to a specific vertex.
    For every edge connected to the vertex, the connectivity and the value
    of the respective edge are written to buffers depending on whether the edge
    is ingoing (`in_` prefix) or outgoing (`out_`prefix).

    Args:
        vertex (int): Computational graph vertex we want to eliminate according
                    to the vertex elimination scheme
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
    Returns:
        Tuple: A tuple containing the updated buffers and state of the graph.
    """

    # Define identity function for lax.cond
    def id(*xs):
        return xs[1:]
    
    def loop_fn(carry, _):
        # n tracks where we are in the sparse graph representation
        # i, j track the current indices of in the ingoing and outgoing 
        # representations of the buffer
        n, i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals = carry

        # Get the current edge
        edge = edge_conn[n]
        # print(edge)
        
        # Edge is ingoing edge of the vertex
        out = lax.cond(edge[1] == vertex, del_and_copy_edge, id, 
                        n, i, in_pos, in_vals, edge_conn, edge_vals)
        i, in_pos, in_vals, edge_conn, edge_vals = out
        
        # Edge is outgoing edge of the vertex
        out = lax.cond(edge[0] == vertex, del_and_copy_edge, id, 
                        n, j, out_pos, out_vals, edge_conn, edge_vals)
        j, out_pos, out_vals, edge_conn, edge_vals = out

        # i: ingoing edge index in buffer.
        # j: outgoing edge index in buffer.
                        
        carry = (n+1, i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals)
        return carry, 0
    
    # Loop running over all the edges in the sparse representation of the graph
    carry_init = (0, 0, 0, -jnp.ones((IN_VAL_BUFFER_SIZE, 2)), 
                            jnp.zeros((IN_VAL_BUFFER_SIZE, 5)), 
                            -jnp.ones((OUT_VAL_BUFFER_SIZE, 2)), 
                            jnp.zeros((OUT_VAL_BUFFER_SIZE, 5)), edge_conn, edge_vals)
    output, _ = lax.scan(loop_fn, carry_init, None, length=edge_conn.shape[0])
    
    return output[1:]


def add_edge(edge: Array, 
            in_jac: Array, 
            out_jac: Array, 
            n: int, 
            k: int, 
            edge_conn: Array, 
            edge_vals: Array, 
            free_idxs: Array,
            n_ops: int) -> Tuple:
    """
    Function that adds an edge to the computational graph. If the edge already
    exists, the current value is added to the product of the ingoing and outgoing
    edge. It uses the `free_idxs` buffer to keep track of where in the 
    computational graph representation we can add new edges after the edges 
    connected to the vertex in question have been removed by the `get_edges` function.

    Args:
        edge (Array): The edge that we want to add to the computational graph.
                        It consists of the sender and receiver of the edge.
        in_jac (Array): Sparsity type and Jacobian shape of the ingoing edge.
        out_jac (Array): Sparsity type and Jacobian shape of the outgoing edge.
        n (int): Counter variable that tracks where we are in the `edge_combos`
                buffer of the `make_new_edges` function.
        k (int): Counter variable that tracks where we are in the `free_idxs` buffer.
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
        free_idxs (Array): Buffer that keeps track of where we can add new edges
                            in the graph representation.
        n_ops (int): Counter variable that tracks the number of multiplications
                    incurred by the vertex elimination.

    Returns:
        Tuple: Returns a tuple containing the updated counter variables, buffers
                and state of the computational graph.
    """
    # print("edge: ", edge)
    # Check if the edge we want to create aleady exists
    edge_exists = jnp.all(edge_conn == edge, axis=-1)
    existing_edge_idx = jnp.argwhere(edge_exists, size=1, fill_value=-1)[0][0]
    # Depending on whether the edge exists or not, we either add the value of the
    # edge to the existing edge or create a new edge
    # TODO add more documentation
    ops, jac = sparse_mul(in_jac, out_jac)
    k, idx, jac = lax.cond(existing_edge_idx > -1,
                            lambda k: (k, existing_edge_idx, 
                                        sparse_add(jac, edge_vals[existing_edge_idx])),
                            lambda k: (k+1, free_idxs[k].astype(jnp.int32), jac), k)

    # Add the edge to the graph representation
    edge_vals = edge_vals.at[idx].set(jac)
    edge_conn = edge_conn.at[idx].set(edge)
    return n+1, k, edge_conn, edge_vals, n_ops+ops


def make_new_edges(edge_combos: Array, 
                    in_vals: Array, 
                    out_vals: Array, 
                    edge_conn: Array, 
                    edge_vals: Array, 
                    free_idxs: Array) -> Tuple:
    """
    Function that creates new edges in the computational graph. It uses the 
    combination of ingoing and outgoing edges to create new edges stored in
    the `edge_combos` variable. 

    Args:
        edge_combos (Array): 
        in_vals (Array): Values of the ingoing edges.
        out_vals (Array): Values fo the outgoing edges.
        edge_conn (Array): Connectivity of the graph. Essentially contains senders
                            and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
        free_idxs (Array): Buffer that keeps track of where we can add new edges.

    Returns:
        Tuple: Returns a tuple containing the updated counter variables and
                graph representation as well as the number of multiplications
                that the vertex elimination incurred.
    """
    # Define identity function for lax.cond
    def id(edge, in_val, out_val, n, k, edge_conn, edge_vals, free_idxs, n_ops):
        return (n+1, k, edge_conn, edge_vals, n_ops)
    
    def loop_fn(carry, edge):
        n, k, edge_conn, edge_vals, n_ops = carry
        # Check whether the edge is valid by checking if the sender and receiver
        # are not -1.
        is_valid_edge = jnp.logical_and(edge[0]>=0, edge[1]>=0)
        
        # Add the edge to the graph representation
        # TODO use the correct number of `n` for a mixed size of senders and receivers
        in_val = in_vals[n % IN_VAL_BUFFER_SIZE]
        out_val = out_vals[n // OUT_VAL_BUFFER_SIZE]
        carry = lax.cond(is_valid_edge, add_edge, id,
                        edge, in_val, out_val, n, k, edge_conn, edge_vals, free_idxs, n_ops)

        return carry, None
        
    # Loop running over all the edges in the `edge_combos` buffer
    carry_init = (0, 0, edge_conn, edge_vals, 0.)
    output, _ = lax.scan(loop_fn, carry_init, edge_combos)
    return output[1:]

########################################################################################


def vertex_eliminate(vertex: int, graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements vertex elimination in a sparse, jittable fashion.
    TODO add more documentation

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.
        vertex (int): Vertex that we want to eliminate.

    Returns:
        GraphsTuple: The resulting graph after the vertex elimination.
    """
    # print('vertex: ', vertex)
    # Divide the graph representation into its components
    # edge_conn contains the senders and receivers of the graph, i.e. the connectivity
    # of the vertices with each other
    # edge_vals contains the values of the edges
    # print('Sparse eliminated vertex: ',vertex)
    edge_conn = jnp.stack([graph.senders, graph.receivers]).T
    edge_vals = graph.edges
    # Get the edges connected to the vertex
    # i, j are the used number of places in the buffers, i.e. the number of ingoing
    # and outgoing edges
    i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals = get_edges(vertex, edge_conn, edge_vals)
    # jax.debug.print("vertex: {v}, in edges: {i}, out edges: {j}", v=vertex, i=i, j=j)
    # Calculate the new edges and where in the graph representation we can add them
    is_zero = jnp.all(edge_vals == 0, axis=-1)
    free_idxs = jnp.argwhere(is_zero, size=IN_VAL_BUFFER_SIZE * OUT_VAL_BUFFER_SIZE).flatten() # edges where features are set to zero.
    edge_combos = jnp.stack(jnp.meshgrid(in_pos[:, 0], out_pos[:, 1]))
    edge_combos = edge_combos.reshape(2, IN_VAL_BUFFER_SIZE*OUT_VAL_BUFFER_SIZE).T

    # Add the new edges to the graph representation
    # k is the number of newly created edges
    output = make_new_edges(edge_combos, in_vals, out_vals, edge_conn, edge_vals, free_idxs)
    k, edge_conn, edge_vals, n_ops = output
    # Build everything into a new graph
    senders = edge_conn[:, 0]
    receivers = edge_conn[:, 1]
    
    nodes = graph.nodes.at[vertex].set(1)
    graph = GraphsTuple(nodes=nodes,
                        edges=edge_vals,
                        senders=senders,
                        receivers=receivers,
                        n_node=graph.n_node,
                        n_edge=graph.n_edge-i-j+k, # TODO this does not give the correct value yet
                        globals=graph.globals+n_ops)
    return graph


def cross_country(order: Sequence[int], graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements the cross-country AD using the 
    vertex elimination algorithm.
    TODO add more documentation
    Args:
        order (Sequence[int]): The order in which the vertices are eliminated.
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after vertex elimination.
    """
    # Eliminate the vertices only if they are not masked
    def loop_fn(carry, i):
        graph, d = carry

        graph = lax.cond(graph.nodes[i] == 0,
                        lambda g: vertex_eliminate(i, g),
                        lambda g: g, graph)
        
        out = jnp.array([i, graph.globals[0][0] - d])
        carry = (graph, graph.globals[0][0])
        return carry, out
    
    # Looping over all the vertices in the order specified
    # print("order: ", order)
    graph, out = lax.scan(loop_fn, (graph, 0), order)
    out = jnp.stack(out).T
    return graph[0], out
    
    
def forward(graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements forward-mode AD on the computational graph.

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after executing forward-mode AD.
    """
    order = jnp.arange(0, len(graph.nodes))
    graph = cross_country(order, graph)
    return graph


def reverse(graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements reverse-mode AD on the computational graph.

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after executing reverse-mode AD.
    """
    order = jnp.arange(0, len(graph.nodes))[::-1]
    graph, out = cross_country(order, graph)
    return graph, out


def embed(num_nodes: int, num_edge: int, graph: GraphsTuple) -> GraphsTuple:
    """
    Function that embeds a computational graph into a larger computational graph
    with `num_nodes` and `num_edge` nodes and edges respectively.

    Args:
        num_nodes (int): Number of nodes of the new computational graph.
        num_edge (int): Number of edges of the new computational graph.
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after embedding.
    """
    node_padding = num_nodes - graph.n_node[0]
    edge_padding = num_edge - graph.n_edge[0]
    
    # Add padding to the nodes
    nodes = jnp.concatenate([graph.nodes, jnp.ones(node_padding)])
    edges = jnp.concatenate([graph.edges, jnp.zeros((edge_padding, 5))])
    
    senders = jnp.concatenate([graph.senders, -jnp.ones(edge_padding)])
    receivers = jnp.concatenate([graph.receivers, -jnp.ones(edge_padding)])
    
    graph = GraphsTuple(nodes=nodes,
                        edges=edges,
                        senders=senders,
                        receivers=receivers,
                        n_node=jnp.array([num_nodes]),
                        n_edge=jnp.array([num_edge]),
                        globals=graph.globals)
    return graph