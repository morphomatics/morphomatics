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

import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from morphomatics.manifold import Manifold


class TangentMLP(hk.Module):
    '''
    Generalized multi-layer perceptron for manifold-valued features.
    '''

    def __init__(self, M: Manifold, out_sizes: Sequence[int], name=None):
        '''
        :param M: Manifold input signal takes values in
        :param out_sizes: number of output feature channels (sequence thereof)
        '''

        super().__init__(name=type(self).__name__ if name is None else name)
        self.M = M
        self.mlp = VectorNeuronMLP(out_sizes)

    def __call__(self, x):
        '''
        Apply tangent MLP layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
            (M being the underlying manifold)
        :return: output with shape: batch * sequence_length * out_channel * M.point_shape
        '''

        n_batch, n_seq, n_in, *pt_shape = x.shape

        # flatten first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape(-1, n_in, *pt_shape)

        # compute logs -> tangent vectors
        pernode = jax.vmap(self.M.connec.log, in_axes=(None, 0))
        v = jax.vmap(pernode)(x[:, 0], x[:, 1:])

        # shape into vectors
        v = v.reshape(n_batch, n_seq, n_in-1, -1)

        # apply vector neuron MLP
        v = self.mlp(v)

        # shape back into tangent vectors
        v = v.reshape(n_batch * n_seq, -1, *pt_shape)

        # map back to manifold
        pernode = jax.vmap(self.M.connec.exp, in_axes=(None, 0))
        y = jax.vmap(pernode)(x[:, 0], v)

        return y.reshape(n_batch, n_seq, -1, *pt_shape)

class TangentInvariant(hk.Module):
    '''
    Invariant layer for manifold-valued features extending the TangentMLP layer.
    Specifically, computes inner products of the input (linearized via Log) with
     a set of tangent vectors obtained from them (using TangentMLP).
     Finally, the products are passed through a fully connected layer to match desired output size.
    '''

    def __init__(self, M: Manifold, out_channel: int, vec_sizes: Sequence[int] = [3,], with_bias=True, name=None):
        '''
        :param M: Manifold input signal takes values in
        :param out_channel: number of output feature channels
        :param vec_sizes: sequence of widths for TangentMLP
        :param with_bias: whether to use bias in the final fully connected layer
        '''

        super().__init__(name=type(self).__name__ if name is None else name)
        self.M = M
        self.mlp = VectorNeuronMLP(vec_sizes)
        self.FC = hk.Linear(out_channel, with_bias=with_bias)

    def __call__(self, x):
        '''
        Apply tangent MLP layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
            (M being the underlying manifold)
        :return: output with shape: batch * sequence_length * out_channel * M.point_shape
        '''

        n_batch, n_seq, n_in, *pt_shape = x.shape

        # flatten first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape(-1, n_in, *pt_shape)

        # compute logs -> tangent vectors
        log_batched = jax.vmap(self.M.connec.log, in_axes=(None, 0))
        v = jax.vmap(log_batched)(x[:, 0], x[:, 1:])

        # shape into vectors
        w = v.reshape(n_batch, n_seq, n_in-1, -1)

        # apply vector neuron MLP
        w = self.mlp(w)

        # shape back into tangent vectors
        w = v.reshape(n_batch * n_seq, -1, *pt_shape)

        # lower indices of tangent vectors (either v or w)
        flat_batched = jax.vmap(self.M.metric.flat, in_axes=(None, 0))
        if w.shape[1] > n_in:
            v = jax.vmap(flat_batched)(x[:, 0], v)
        else:
            w = jax.vmap(flat_batched)(x[:, 0], w)
        # compute inner products
        y = jnp.einsum('...ij,...kj', v.reshape(*v.shape[:2], -1), w.reshape(*w.shape[:2], -1))

        return self.FC(y.reshape(n_batch, n_seq, -1))


class VectorNeuronMLP(hk.Module):
    """
    Vector Neuron MLP layer as described in
    Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., & Guibas, L. J. (2021).
    Vector neurons: A general framework for SO(3)-equivariant networks.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 12200-12209).
    """

    def __init__(self, output_sizes: Sequence[int], negative_slope: float = 0.2, name=None):
        """
        :param output_sizes: sequence (length: m+1) of layer widths
        :param name: layer name (see haiku documentation)
        """
        super().__init__(name=type(self).__name__ if name is None else name)
        self.output_sizes = output_sizes
        self.negative_slope = negative_slope

    def __call__(self, x: jnp.array):
        '''
        Apply layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * vector_dim
        :return: output with shape: batch * sequence_length * out_channel * vector_shape
        '''

        n_in = x.shape[2]

        # apply layers
        for i, n_out in enumerate(self.output_sizes):
            # initialize weights
            u_init = hk.initializers.TruncatedNormal(stddev=1./np.sqrt(n_in))
            w_init = hk.initializers.TruncatedNormal(mean=1. / n_in, stddev=1. / np.sqrt(n_in))
            U = hk.get_parameter(f"U_{i}", [n_out, n_in], dtype=x.dtype, init=u_init)
            W = hk.get_parameter(f"W_{i}", [n_out, n_in], dtype=x.dtype, init=w_init)
            n_in = n_out

            # compute direction k (and its squared norm)
            k = jnp.einsum('ij,...jk', U, x)
            sqnrm_k = jnp.sum(k**2, axis=-1, keepdims=True) + np.finfo(np.float64).eps

            # compute feature q
            q = jnp.einsum('ij,...jk', W, x)

            # Rotation-equivariant ReLU
            dot_qk = jnp.sum(q * k, axis=-1, keepdims=True)
            x = q + k * jax.nn.relu(-dot_qk) / sqnrm_k
            # Leaky ReLU (weighted average of q and ReLU(q))
            x = self.negative_slope * q + (1 - self.negative_slope) * x

        return x
