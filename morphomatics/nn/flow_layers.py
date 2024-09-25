################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from typing import Iterable, Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

import jraph
from flax import linen as nn

from morphomatics.manifold import Manifold, PowerManifold
from morphomatics.opt import RiemannianNewtonRaphson
from morphomatics.graph.operators import mfdg_laplace
from morphomatics.nn.tangent_layers import TangentMLP


class FlowLayer(nn.Module):
    """
    Graph flow layer for graphs with manifold-valued features. The flow equation is integrated explicitly by default,
    but an implicit scheme is also available.

    See

    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)

    for a detailed description.

    Inputs:
    :param M: manifold in which the features lie
    :param n_steps: number of explicit steps to approximate the flow with explicit Euler
    :param implicit: boolean indicating whether to use implicit or explicit Euler integration
    :param max_step_length: maximum step size for Euler integration

    Note: Too long Euler steps can lead to numerical instabilities with some manifolds, e.g., the hyperbolic space.
    In this case, a maximal step length should be ued.

    """

    M: Manifold
    n_steps: int = 1
    implicit: bool = False
    max_step_length: float = jnp.inf
    t_init: Callable = lambda *args: nn.initializers.truncated_normal(stddev=1.)(*args) + 1.
    delta_init: Callable = lambda *args: nn.initializers.truncated_normal(stddev=1.)(*args) + 1.

    def _single_euler_step(self, G: jraph.GraphsTuple, time: jnp.ndarray, delta: jnp.ndarray) -> jraph.GraphsTuple:
        """Single step of the explicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :param delta: vector of "minimal step sizes"
        :return: updated graph
        """

        def _multi_laplace(channel):
            return mfdg_laplace(self.M, G._replace(nodes=channel))

        def _activation(feature, vector, d):
            nrm = jnp.sqrt(self.M.metric.inner(feature, vector, vector) + jnp.finfo(jnp.float64).eps)
            alp = jax.nn.sigmoid(nrm - d)

            # make sure that the step size is not larger than max_step_length
            return jax.lax.cond(nrm * alp <= self.max_step_length,
                                lambda w: alp * w,
                                lambda w: w * self.max_step_length / nrm, vector)

        v = jax.vmap(_multi_laplace, in_axes=1, out_axes=1)(G.nodes)

        # ReLU-type activation
        delta = jnp.stack([delta, ] * v.shape[0])
        v = jax.vmap(jax.vmap(_activation))(G.nodes, v, delta)

        v = -v * time.reshape((1, -1) + (1,) * (v.ndim - 2))
        x = jax.vmap(jax.vmap(self.M.connec.exp))(G.nodes, v)
        return G._replace(nodes=x)

    def _implicit_euler_step(self, G: jraph.GraphsTuple, time: jnp.ndarray, delta=None) -> jraph.GraphsTuple:
        """Single step of the implicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :param delta: only needed for out of syntax reasons
        :return: updated graph
        """
        # n_nodes x n_channels x point_shape
        n, c, *shape = G.nodes.shape

        # power manifold
        P = PowerManifold(self.M, n * c)

        # current state
        x_cur = G.nodes.reshape(-1, *shape)

        # root of F characterizes solution to implicit Euler step
        def F(x: jnp.array):
            L = lambda a: mfdg_laplace(self.M, G._replace(nodes=a))
            Lx = jax.vmap(L, in_axes=1, out_axes=1)(x.reshape(n, c, *shape))
            tLx = Lx * time.reshape((1, -1) + (1,) * len(shape))
            diff = P.connec.log(x, x_cur)
            return diff - tLx.reshape(-1, *shape)

        # x_next = RiemannianNewtonRaphson.solve(P, F, x_cur, maxiter=1)
        ###############################
        # unroll single interation
        ###############################
        # solve for update direction: v = -J⁻¹F(x)
        J = lambda v: jax.jvp(F, (x_cur,), (v,))[1]
        v, _ = bicgstab(J, -F(x_cur))
        # step
        x_next = P.connec.exp(x_cur, v)

        return G._replace(nodes=x_next.reshape(n, c, *shape))

    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        :param G: graphs tuple with features of shape: num_nodes * num_channels * point_shape
        :return: graphs tuple with features of shape: num_nodes * num_channels * point_shape

        Apply discretized diffusion flow (with final activation) to each channel.
        """
        step_method = self._implicit_euler_step if self.implicit else self._single_euler_step

        width = G.nodes.shape[1]  # number of channels
        ####################
        t = self.param("t_sqrt", self.t_init, (width,), G.nodes.dtype)
        delta = self.param("delta_sqrt", self.delta_init, (width,), G.nodes.dtype)

        # map to non-negative weights
        t = t ** 2
        delta = delta ** 2

        def step(graph, _):
            graph = step_method(graph, t / self.n_steps, delta)
            return graph, None

        G, _ = jax.lax.scan(step, G, None, self.n_steps, unroll=self.n_steps)

        return G


class MfdGcnBlock(nn.Module):
    """
    Manifold convolution network block as proposed in
    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)

    An explicit skip connection can be added that attaches the input to the output as an additional channels.
    Note: If skip connections between each flow-tMLP unit are wanted, use several blocks with only one layer.

    :param M: manifold constituting the signal domain
    :param channel_sizes: sequence of channel sizes
    :param n_steps: number of Euler steps that are performed in the flow layer
    :param implicit: boolean indicating whether to use implicit or explicit Euler integration
    :param max_step_length: maximum step size for Euler integration (see the flow layer)
    :param explicit_skip: boolean indicating whether additionally to perform an explicit skip connection
    :param inputs_are_copies: when true, only the first input channel is passed through by the skip connection
    """

    M: Manifold
    channel_sizes: Iterable[int]
    n_steps: int = 1
    implicit: bool = False
    max_step_length: float = jnp.inf
    explicit_skip: bool = False
    inputs_are_copies: bool = False

    def setup(self):
        layers = []
        channel_sizes = tuple(self.channel_sizes)
        for i, channel_size in enumerate(channel_sizes):
            layers.append(
                (
                    FlowLayer(self.M, self.n_steps, self.implicit, self.max_step_length),
                    TangentMLP(self.M, (channel_size,))
                )
            )
        self.layers = tuple(layers)

    def __call__(self, G: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        :param G: graphs tuple with features of shape: num_nodes * num_channels * point_shape
        :return: graphs tuple with features of shape: num_nodes * out_channels * point_shape

        We use jraph pooling; hence, the batches are combined in the same graph and thus "hidden" in num_nodes.
        The number of output channels is the number of output channels of the last (tangentMLP) layer plus, if
        activated, the number of channels that are passed through by the skip connection (either num_channels or 1).
        """

        # save input for the skip connection
        if self.explicit_skip:
            if self.inputs_are_copies:
                z = G.nodes[:, 1, None]
            else:
                z = G.nodes

        for layer_unit in self.layers:
            # flow layer
            G = layer_unit[0](G)
            # tangent MLP
            G = G._replace(nodes=layer_unit[1](G.nodes[None])[0])

        # skip connection
        if self.explicit_skip:
            G = G._replace(nodes=jax.lax.concatenate([z, G.nodes], 1))

        return G
