################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from typing import Sequence

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

import jraph
import haiku as hk

from morphomatics.manifold import Manifold, PowerManifold
from morphomatics.opt import RiemannianNewtonRaphson
from morphomatics.graph.operators import mfdg_laplace
from morphomatics.nn.tangent_layers import TangentMLP


class flow_layer(hk.Module):
    """
    Graph flow layer for graphs with manifold-valued features. The flow equation is integrated explicitly.
    """

    def __init__(self, M: Manifold, n_steps=1, implicit=False, name=None):
        """
        :param M: manifold in which the features lie
        :param n_steps: number of explicit steps to approximate the flow with explicit Euler
        :param implicit: whether to use implicit or explicit Euler integration
        :param name: layer name (see haiku documentation of haiku)
        """
        super().__init__(name=type(self).__name__ if name is None else name)
        self.M = M
        self.n_steps = n_steps
        self.step = self._implicit_euler_step if implicit else self._single_euler_step

    def _single_euler_step(self, G: jraph.GraphsTuple, time: jnp.ndarray, delta: jnp.ndarray) -> jraph.GraphsTuple:
        """Single step of the explicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :return: updated graph
        """
        def _multi_laplace(channel):
            return mfdg_laplace(self.M, G._replace(nodes=channel))

        def _activation(feature, vector, d):
            return jax.lax.cond(self.M.metric.norm(feature, vector) >= d,
                                lambda _, w: w,
                                lambda _, w: jnp.zeros_like(w), feature, vector)

        v = jax.vmap(_multi_laplace, in_axes=1, out_axes=1)(G.nodes)

        # ReLU-type activation
        delta = jnp.stack([delta, ] * v.shape[0])
        v = jax.vmap(jax.vmap(_activation))(G.nodes, v, delta)

        v = -v * time.reshape((1, -1) + (1,) * (v.ndim - 2))
        x = jax.vmap(jax.vmap(self.M.connec.exp))(G.nodes, v)
        return G._replace(nodes=x)

    def _implicit_euler_step(self, G: jraph.GraphsTuple, time: jnp.ndarray, delta: jnp.ndarray) -> jraph.GraphsTuple:
        """Single step of the implicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :return: updated graph
        """
        # n_nodes x n_channels x point_shape
        n, c, *shape = G.nodes.shape

        # power manifold
        P = PowerManifold(self.M, n*c)

        # current state
        x_cur = G.nodes.reshape(-1, *shape)

        # root of F characterizes solution to implicit Euler step
        def F(x: jnp.array):
            L = lambda a: mfdg_laplace(self.M, G._replace(nodes=a))
            Lx = jax.vmap(L, in_axes=1, out_axes=1)(x.reshape(n, c, *shape))
            tLx = Lx * time.reshape((1, -1) + (1,)*len(shape))
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

    def __call__(self, G: jraph.GraphsTuple) -> jraph.GraphsTuple:
        width = G.nodes.shape[1]  # number of channels
        # init parameter
        t_init = hk.initializers.TruncatedNormal(stddev=1, mean=1)
        # t_init = hk.initializers.Constant(0.75)
        delta_init = hk.initializers.TruncatedNormal(stddev=1, mean=1)
        ####################
        t = hk.get_parameter("t_sqrt", shape=[width], init=t_init)
        delta = hk.get_parameter("delta_sqrt", shape=[width], init=delta_init)
        ####################

        # print(t)

        # map to non-negative weights
        t = t**2
        delta = delta**2

        # make n_steps explicit Euler steps for each graph in the batch
        # for _ in range(self.n_steps):
        #     G = self.step(G, t / self.n_steps)

        def step(graph, _):
            graph = self.step(graph, t / self.n_steps, delta)
            return graph, None

        G, _ = jax.lax.scan(step, G, None, self.n_steps, unroll=self.n_steps)

        return G


class MfdGcnBlock(hk.Module):
    """
    Manifold convolution network block as proposed in
    M. Hanik, G. Steidl, C. v. Tycowicz (2024)
    "Manifold GCN: Diffusion-based Convolutional Neural Network for Manifold-valued Graphs."
    """

    def __init__(self, M: Manifold, out_channels: Sequence[int] = [3,], name=None):
        """
        :param M: manifold constituting the signal domain
        :param out_channels: number of out channels (sequence thereof) of the tangent MLPs
        :param name: name of block
        """
        super().__init__(name=type(self).__name__ if name is None else name)

        self._M: Manifold = M
        self.out_channels = out_channels

    def __call__(self, G: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        :param G: graphs tuple with features of sizes (num_nodes, num_channels, point_shape)
        :return: graphs tuple with features of sizes (num_nodes, out_channels[-1], point_shape)

        Use jraph pooling -> batches are hidden in num_nodes
        """

        for i in range(len(self.out_channels)):
            # diffusion layer
            G = flow_layer(self._M)(G)
            G = G._replace(nodes=TangentMLP(self._M, (self.out_channels[i],))(G.nodes[None])[0])

            return G
