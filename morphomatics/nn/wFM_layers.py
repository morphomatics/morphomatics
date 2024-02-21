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

import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from morphomatics.manifold import Manifold


class MfdFC(hk.Module):
    '''
    Fully connected layer for manifold-valued features.
    Weighted Fréchet means are used to generalize linear combinations.
    '''

    def __init__(self, M: Manifold, out_channel: int, name=None):
        '''
        :param M: Manifold input signal takes values in
        :param out_channel: number of output feature channels
        '''

        super().__init__(name=type(self).__name__ if name is None else name)
        self.M = M
        self.out_channel = out_channel

    def __call__(self, x, w=None):
        '''
        Apply fully connected layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
            (M being the underlying manifold)
        :param w: weights for the Fréchet means of shape n_in, n_out
        :return: output with shape: batch * sequence_length * out_channel * M.point_shape
        '''
        n_in, n_out = x.shape[2], self.out_channel
        out_shape = x.shape[:2] + (n_out,) + x.shape[3:]

        if w is None:
            # init parameter
            w_init = hk.initializers.TruncatedNormal(mean=-np.log(n_in))

            w = hk.get_parameter("w", shape=(n_in, n_out), dtype=x.dtype, init=w_init)

        # map to positive weights that sum to 1
        w = jnp.exp(w)

        # weights sum to 1
        w = w / w.sum(axis=0)[None]

        # flatten first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape((-1,) + x.shape[2:])

        # compute means
        channel_map = jax.vmap(lambda x_i, w_j: wFM(x_i, w_j, self.M), in_axes=(None, -1))
        y = jax.vmap(channel_map, in_axes=(0, None if w.ndim == 2 else 0))(x, w)

        return y.reshape(out_shape)


class MfdInvariant(hk.Module):
    '''
    Last invariant layer.
    '''

    def __init__(self, M: Manifold, out_channel: int, nC: int = 1, with_bias=True, name=None):
        '''
        :param M: Manifold input signal takes values in
        :param out_channel: number of output feature channels
        :param nC: number of weighted means to which distances are employed
        '''

        super().__init__(name=type(self).__name__ if name is None else name)

        self.M: Manifold = M
        self.MfdFC = MfdFC(M, nC)
        self.FC = hk.Linear(out_channel, with_bias=with_bias)

    def __call__(self, x):
        '''
        Apply fully connected layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
            (M being the underlying manifold)
        :return: output with shape: batch * sequence_length * out_channel
        '''

        nBatch, L, *_ = x.shape

        # compute means
        y = self.MfdFC(x)

        # flatten first two axes -> shape: (batch * sequence_length) * #channels * M.point_shape
        x = x.reshape((-1,) + x.shape[2:])
        y = y.reshape((-1,) + y.shape[2:])

        # compute distances, shape: (batch * sequence_length) * #channels_x * #channels_y
        y_channel = jax.vmap(self.M.metric.squared_dist, in_axes=(None, 0))
        x_channel = jax.vmap(y_channel, in_axes=(0, None))
        d = jax.vmap(x_channel)(x, y)
        d = jnp.sqrt(d + 1e-6)
        # reshape to batch * sequence_length * (#channels_x * #channels_y)
        d = d.reshape(nBatch, L, -1)

        return self.FC(d)


def wFM(x: jnp.array, w: jnp.array, M: Manifold):
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
