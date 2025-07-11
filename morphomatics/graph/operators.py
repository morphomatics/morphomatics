################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2025 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
from jax import tree_util as tree
import jax.numpy as jnp
import jraph

from morphomatics.manifold import Manifold


def mfdg_gradient(M: Manifold, graph: jraph.GraphsTuple) -> jnp.array:
    """Given a graph with manifold-valued features returns the gradient as defined in https://arxiv.org/abs/1702.05293

    :param M: feature manifold
    :param graph: M-valued graph
    :return: edge gradients
    """

    # gradients on directed edges are stored along first axis (orderd given by graph.senders/receivers)
    logs = jax.vmap(M.connec.log)(graph.nodes[graph.senders], graph.nodes[graph.receivers])

    return vec_times_kd(jnp.sqrt(graph.edges), logs)


def mfdg_laplace(M: Manifold, graph: jraph.GraphsTuple) -> jnp.array:
    """Given a graph with manifold-valued features returns the isotropic graph 2-Laplacian as defined in
    https://arxiv.org/abs/1702.05293

    :param M: feature manifold
    :param graph: M-valued graph
    :return: Laplace vectors at node feature
    """

    # multiply negative gradients with corresponding square root edge weights
    weighted_gradients = vec_times_kd(jnp.sqrt(graph.edges), -mfdg_gradient(M, graph))

    return jax.ops.segment_sum(weighted_gradients, graph.senders, num_segments=len(graph.nodes))  # already ordered


def max_pooling(G: jraph.GraphsTuple, z: jnp.ndarray) -> jnp.array:
    """Graph-wise max pooling of features z over graph G.

    :param G: graph
    :param z: scalar graph features
    :return: max value for each graph in the batch
    """
    sum_n_node = tree.tree_leaves(G.nodes)[0].shape[0]
    n_graph = G.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)

    node_gr_idx = jnp.repeat(
        graph_idx, G.n_node, axis=0, total_repeat_length=sum_n_node)

    # return constant (not -inf) for graphs without edges as these can appear through batching
    return jraph.segment_max_or_constant(z, node_gr_idx, n_graph, indices_are_sorted=True)


def mean_pooling(G: jraph.GraphsTuple, z: jnp.ndarray) -> jnp.array:
    """Graph-wise mean pooling of features z over graph G.

    :param G: graph
    :param z: scalar graph features
    :return: mean value for each graph in the batch
    """
    sum_n_node = tree.tree_leaves(G.nodes)[0].shape[0]
    n_graph = G.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)

    node_gr_idx = jnp.repeat(
        graph_idx, G.n_node, axis=0, total_repeat_length=sum_n_node)

    return jax.ops.segment_sum(z, node_gr_idx, n_graph, indices_are_sorted=True) / jnp.clip(G.n_node, 1).reshape((n_graph,) + (1,)*(z.ndim-1))


def in_degree_centrality(G: jraph.GraphsTuple) -> float:
    """ Normalized in-degree centrality

    """
    def f(n: int):
        return jnp.count_nonzero(G.receivers == n)

    d = jax.vmap(f)(jnp.arange(G.n_node[0]))

    return d / jnp.max(d)


def out_degree_centrality(G: jraph.GraphsTuple) -> float:
    """ Normalized out-degree centrality

    """
    def f(n: int):
        return jnp.count_nonzero(G.senders == n)

    d = jax.vmap(f)(jnp.arange(G.n_node[0]))

    return d / jnp.max(d)


def atleast_kd(array, k) -> jnp.array:
    """Make array at least k-dimensional"""
    array = jnp.asarray(array)
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)


def vec_times_kd(vec: jnp.array, tensor: jnp.array) -> jnp.array:
    """Multiply the i-th slice (along first axis) of a tensor with the i-th element of a given vector

       The vector can be 2-dimensional as long as one of the dimensions is of size one, i.e, it is a proper vector.

       Attention: non-jitable with asserts
    """

    return atleast_kd(vec, len(tensor.shape)) * tensor
