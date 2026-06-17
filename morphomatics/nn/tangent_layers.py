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

from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx

from morphomatics.manifold import Manifold


class TangentMLP(nnx.Module):
    """
    Generalized multi-layer perceptron for manifold-valued features as proposed in
    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)."""

    def __init__(self, M: Manifold, feature_channels: Sequence[int], rngs: nnx.Rngs):
        """
        :param M: Manifold input signal takes values in
        :param feature_channels: the number of feature channels (sequence thereof)
        :param rngs: random number generator
        """
        self.M = M
        self.n_in = feature_channels[0]
        self.n_out = feature_channels[-1]
        feature_channels = list(feature_channels)
        feature_channels[0] -= 1
        self.vector_neuron_MLP = VectorNeuronMLP(feature_channels, rngs)


    def __call__(self, x):
        """
        Apply tangent MLP layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
            (M being the underlying manifold)
        :return: output with shape: batch * sequence_length * out_channel * M.point_shape
        """
        n_batch, n_seq, _, *pt_shape = x.shape

        # flatten the first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape(-1, self.n_in, *pt_shape)

        # compute logs -> tangent vectors
        pernode = jax.vmap(self.M.connec.log, in_axes=(None, 0))
        v = jax.vmap(pernode)(x[:, 0], x[:, 1:])

        # shape into vectors
        v = v.reshape(n_batch, n_seq, self.n_in-1, -1)

        # apply vector neuron MLP
        v = self.vector_neuron_MLP(v)

        # shape back into tangent vectors
        v = v.reshape(n_batch * n_seq, -1, *pt_shape)

        # map back to manifold
        pernode = jax.vmap(self.M.connec.exp, in_axes=(None, 0))
        y = jax.vmap(pernode)(x[:, 0], v)

        return y.reshape(n_batch, n_seq, -1, *pt_shape)


class TangentInvariant(nnx.Module):
    """
    Invariant layer for manifold-valued features extending the TangentMLP layer.
    Specifically, computes inner products of the input (linearized via Log) with
    a set of tangent vectors obtained from them (using TangentMLP).
    Finally, the products are passed through a fully connected layer to match the desired output size."""

    def __init__(self,
                 M: Manifold,
                 n_in: int,
                 n_out: int,
                 rngs: nnx.Rngs,
                 use_bias: bool = True):
        """
        :param M: Manifold input signal takes values in
        :param n_in: number of input channels
        :param n_out: number of output channels
        :param rngs: random number generator
        :param use_bias: whether to use bias in the final fully connected layer
        """

        self.M = M
        self.n_in = n_in
        self.n_out = n_out
        self.vector_neuron_mlp = VectorNeuronMLP([n_in-1, n_out], rngs)
        self.linear = nnx.Linear(n_out * (n_in-1), n_out, rngs=rngs, use_bias=use_bias)

    def __call__(self, x):
        """
        Apply tangent MLP layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * M.point_shape
        :return: output with shape: batch * sequence_length * out_channel
        """

        n_batch, n_seq, _, *pt_shape = x.shape

        # flatten the first two axes -> shape: (batch * sequence_length) * in_channel * M.point_shape
        x = x.reshape(-1, self.n_in, *pt_shape)

        # compute logs -> tangent vectors
        log_batched = jax.vmap(self.M.connec.log, in_axes=(None, 0))
        v = jax.vmap(log_batched)(x[:, 0], x[:, 1:])

        # shape into vectors
        w = v.reshape(n_batch, n_seq, self.n_in-1, -1)

        # apply vector neuron MLP
        w = self.vector_neuron_mlp(w)

        # shape back into tangent vectors
        w = w.reshape(n_batch * n_seq, -1, *pt_shape)

        # lower indices of tangent vectors (either v or w)
        flat_batched = jax.vmap(self.M.metric.flat, in_axes=(None, 0))
        if w.shape[1] > self.n_in:
            v = jax.vmap(flat_batched)(x[:, 0], v)
        else:
            w = jax.vmap(flat_batched)(x[:, 0], w)
        # compute inner products
        y = jnp.einsum('...ij,...kj', v.reshape(*v.shape[:2], -1), w.reshape(*w.shape[:2], -1))

        return self.linear(y.reshape(n_batch, n_seq, -1))


class VectorNeuronMLP(nnx.Module):
    """
    Vector Neuron MLP layer as described in
    Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., & Guibas, L. J. (2021).
    Vector neurons: A general framework for SO(3)-equivariant networks.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 12200-12209)."""

    def __init__(self, feature_channels: Sequence[int], rngs: nnx.Rngs, negative_slope: float = 0.2):
        """
        :param feature_channels: sequence of layer widths, the first entry being the input dimension
        :param rngs: random number generator
        :parameter negative_slope: slope of the Leaky ReLU
        """
        self.feature_sizes = feature_channels
        self.negative_slope = negative_slope
        self.U = nnx.Param(
            [nnx.initializers.truncated_normal(stddev=np.sqrt(2/feat_in))(rngs.param(), (feat_out, feat_in))
             for feat_in, feat_out in zip(feature_channels, feature_channels[1:])])
        self.W = nnx.Param(
            [nnx.initializers.truncated_normal(stddev=1/np.sqrt(feat_in))(rngs.param(), (feat_out, feat_in)) + 1/feat_in
             for feat_in, feat_out in zip(feature_channels, feature_channels[1:])])

    def __call__(self, x: jnp.ndarray):
        """
        Apply layer.
        :param x: input sequence with shape: batch * sequence_length * in_channel * vector_dim
        :return: output with shape: batch * sequence_length * out_channel * vector_shape
        """

        # apply layers
        for U, W in zip(self.U.value, self.W.value):
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
