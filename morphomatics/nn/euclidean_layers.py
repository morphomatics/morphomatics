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

from typing import Sequence, Callable

from flax import nnx

import jax.numpy as jnp


class MLP(nnx.Module):
    def __init__(self,
                 features_sizes: Sequence[int],  # first is input dim
                 rngs: nnx.Rngs,
                 hidden_activation: Callable = nnx.leaky_relu,
                 final_activation: Callable = None):

        self.layers = nnx.List([nnx.Linear(feat_in, feat_out, rngs=rngs)
                       for feat_in, feat_out in zip(features_sizes, features_sizes[1:])])
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation

    def __call__(self, x: jnp.ndarray):
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.hidden_activation(x)
            elif self.final_activation is not None:
                x = self.final_activation(x)
        return x
