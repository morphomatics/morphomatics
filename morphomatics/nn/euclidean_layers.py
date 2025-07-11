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

from typing import Sequence, Callable
import flax.linen as nn

import jax.numpy as jnp


class MLP(nn.Module):
    features: Sequence[int]
    hidden_activation: Callable = nn.leaky_relu
    final_activation: Callable = None

    def setup(self):
        # biases are used
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs: jnp.array):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.hidden_activation(x)
            elif self.final_activation is not None:
                x = self.final_activation(x)
        return x
