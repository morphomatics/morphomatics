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


class FlowLayer(hk.Module):
    """
    Graph flow layer for graphs with manifold-valued features. The flow equation is integrated explicitly by default,
    but an implicit scheme is also available.

    See

    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)

    for a detailed description.

    """

    def __init__(self, M: Manifold, n_steps=1, implicit=False, name=None, max_step_length: float = jnp.inf):
        """
        :param M: manifold in which the features lie
        :param n_steps: number of explicit steps to approximate the flow with explicit Euler
        :param implicit: whether to use implicit or explicit Euler integration
        :param name: layer name (see haiku documentation of haiku)
        :param max_step_length: maximum step size for Euler integration

        Note: Too long Euler steps can lead to numerical instabilities with some manifolds, e.g., the hyperbolic space.
        In this case, a maximal step length should be ued.
        """
        super().__init__(name=type(self).__name__ if name is None else name)
        self._M = M
        self._n_steps = n_steps
        self.step = self._implicit_euler_step if implicit else self._single_euler_step
        self._max_step_length = max_step_length if max_step_length > 0 else -max_step_length

    def _single_euler_step(self, G: jraph.GraphsTuple, time: jnp.ndarray, delta: jnp.ndarray) -> jraph.GraphsTuple:
        """Single step of the explicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :param delta: vector of "minimal step sizes"
        :return: updated graph
        """
        def _multi_laplace(channel):
            return mfdg_laplace(self._M, G._replace(nodes=channel))

        def _activation(feature, vector, d):
            nrm = jnp.sqrt(self._M.metric.inner(feature, vector, vector) + jnp.finfo(jnp.float64).eps)
            alp = jax.nn.sigmoid(nrm - d)

            # make sure that the step size is not larger than max_step_length
            return jax.lax.cond(nrm * alp <= self._max_step_length,
                                lambda w: alp * w,
                                lambda w: w * self._max_step_length / nrm, vector)

        v = jax.vmap(_multi_laplace, in_axes=1, out_axes=1)(G.nodes)

        # ReLU-type activation
        delta = jnp.stack([delta, ] * v.shape[0])
        v = jax.vmap(jax.vmap(_activation))(G.nodes, v, delta)

        v = -v * time.reshape((1, -1) + (1,) * (v.ndim - 2))
        x = jax.vmap(jax.vmap(self._M.connec.exp))(G.nodes, v)
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
        P = PowerManifold(self._M, n*c)

        # current state
        x_cur = G.nodes.reshape(-1, *shape)

        # root of F characterizes solution to implicit Euler step
        def F(x: jnp.array):
            L = lambda a: mfdg_laplace(self._M, G._replace(nodes=a))
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
        """
        :param G: graphs tuple with features of shape: num_nodes * num_channels * point_shape
        :return: graphs tuple with features of shape: num_nodes * num_channels * point_shape

        Apply discretized diffusion (with final activation) flow to each channel
        """

        width = G.nodes.shape[1]  # number of channels
        # init parameter
        t_init = hk.initializers.TruncatedNormal(stddev=1, mean=1)
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
            graph = self.step(graph, t / self._n_steps, delta)
            return graph, None

        G, _ = jax.lax.scan(step, G, None, self._n_steps, unroll=self._n_steps)

        return G


class MfdGcnBlock(hk.Module):
    """
    Manifold convolution network block as proposed in
    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)
    """

    def __init__(self, M: Manifold, out_channels: Sequence[int] = [3,], n_steps: int = 1, name=None):
        """
        :param M: manifold constituting the signal domain
        :param out_channels: number of out channels (sequence thereof) of the tangent MLPs
        :param n_steps: number of Euler steps that are performed in the flow layer
        :param name: name of block
        """
        super().__init__(name=type(self).__name__ if name is None else name)

        self._M: Manifold = M
        self.out_channels = out_channels
        self.n_steps = n_steps

    def __call__(self, G: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        :param G: graphs tuple with features of shape: num_nodes * num_channels * point_shape
        :return: graphs tuple with features of shape: num_nodes * out_channels[-1] * point_shape

        We use jraph pooling; hence, the batches are combined in the same graph and thus "hidden" in num_nodes.
        """

        for nC in self.out_channels:
            # diffusion layer
            G = FlowLayer(self._M, self.n_steps)(G)
            G = G._replace(nodes=TangentMLP(self._M, (nC,))(G.nodes[None])[0])

        return G
