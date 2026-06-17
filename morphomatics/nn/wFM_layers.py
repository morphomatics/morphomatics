################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2026 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx

from morphomatics.manifold import Manifold


class MfdFC(nnx.Module):
    """
    Fully connected layer for manifold-valued features as proposed in
    R. Chakraborty; J. Bouza; J. H. Manton; B. C. Vemuri. "Manifoldnet: A deep neural network for manifold-valued data
    with applications." IEEE Transactions on Pattern Analysis and Machine Intelligence (2020)

    Weighted Fréchet means are used to generalize linear combinations."""

    def __init__(self, M: Manifold, n_in: int, n_out: int, rngs: nnx.Rngs):
        """
        :param M: Manifold input signal takes values in
        :param n_in: number of input feature channels
        :param n_out: number of output feature channels
        :param rngs: random number generator
        """
        self.M = M
        self.n_in = n_in
        self.n_out = n_out
        self.w = nnx.Param(nnx.initializers.truncated_normal(stddev=1.)(rngs.param(), shape=(n_in, n_out)))

    def __call__(self, x):
        """
        Apply the fully connected layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
        :return: output with shape: batch * sequence_length * out_channel * M.point_shape
        """
        out_shape = x.shape[:2] + (self.n_out,) + x.shape[3:]

        # map to positive weights that sum to 1
        w = jnp.exp(self.w - np.log(self.n_in))

        # weights sum to 1
        w = w / w.sum(axis=0)[None]

        # flatten the first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape((-1,) + x.shape[2:])

        # compute means
        channel_map = jax.vmap(lambda x_i, w_j: wFM(x_i, w_j, self.M), in_axes=(None, -1))
        y = jax.vmap(channel_map, in_axes=(0, None if w.ndim == 2 else 0))(x, w)

        return y.reshape(out_shape)


class MfdInvariant(nnx.Module):
    """
    Last invariant layer as proposed in
    R. Chakraborty; J. Bouza; J. H. Manton; B. C. Vemuri. "Manifoldnet: A deep neural network for manifold-valued data
    with applications." IEEE Transactions on Pattern Analysis and Machine Intelligence (2020)."""

    def __init__(self, M: Manifold, n_in: int, n_out: int, rng: nnx.Rngs, n_means: int = 1):
        """
        :param M: Manifold input signal takes values in
        :param n_in: number of input feature channels
        :param n_out: number of output feature channels
        :param rng: random number generator
        :param n_means: number of weighted means to which distances are computed
        """

        self.M = M
        self.n_in = n_in
        self.n_out = n_out
        self.n_means = n_means
        self.mfd_fc = MfdFC(M, n_in, n_means, rng)
        self.linear = nnx.Linear(n_in * n_means, n_out, rngs=rng)

    def __call__(self, x):
        """
        Apply the fully connected layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
        :return: output with shape: batch * sequence_length * out_channel
        """

        nBatch, n_seq, *_ = x.shape

        # compute means
        y = self.mfd_fc(x)

        # flatten the first two axes -> shape: (batch * sequence_length) * #channels * M.point_shape
        x = x.reshape((-1,) + x.shape[2:])
        y = y.reshape((-1,) + y.shape[2:])

        # compute distances, shape: (batch * sequence_length) * #channels_x * #channels_y
        y_channel = jax.vmap(self.M.metric.squared_dist, in_axes=(None, 0))
        x_channel = jax.vmap(y_channel, in_axes=(0, None))
        d = jax.vmap(x_channel)(x, y)
        d = jnp.sqrt(d + 1e-6)
        # reshape to batch * sequence_length * (#channels_x * #channels_y)
        d = d.reshape(nBatch, n_seq, -1)

        return self.linear(d)


def wFM(x: jnp.ndarray, w: jnp.ndarray, M: Manifold):
    """
    Compute weighted Fréchet mean.
    :param x: input features with shape: leading_shape * M.point_shape
    :param w: weights with shape: leading_shape
    :param M: underlying manifold
    :return: mean
    """

    # flatten leading_shape
    w = w.reshape(-1)
    x = x.reshape(-1, *M.point_shape)

    #####################################
    # (unrolled) Newton-type iteration
    #####################################

    def body(a, _):
        grad = jnp.einsum('i, i...', w, jax.vmap(M.connec.log, (None, 0))(a, x))
        return M.connec.exp(a, grad), None
    y, _ = jax.lax.scan(body, x[0], None, 3, unroll=3)

    #####################################
    # recursive estimator
    #####################################

    # # number of points
    # num = np.size(w)
    #
    # w = w / w.sum()
    # idx = jnp.argsort(-w)
    # weights = w[idx]
    # t = weights / jnp.cumsum(weights)
    #
    # # loop = lambda i, mean: M.metric.geopoint(mean, x[idx[i]], t[i])
    # # y = jax.lax.fori_loop(1, num, loop, x[0])

    return y
