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

# postponed evaluation of annotations to circumvent cyclic dependencies (will be default behavior in Python 4.0)
from __future__ import annotations
from typing import Tuple

import jax

import jax.numpy as jnp
import jax.numpy.linalg as jla

from morphomatics.stats import ExponentialBarycenter as Mean


class RiemannianDissimilarityMeasures(object):
    """
    Methods for statistics based on dissimilarity measures on Riemannian manifolds
    """

    def __init__(self, M: Manifold):
        """
        :param M: Riemannian manifold
        """

        self.M = M

    def frechet_mean(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        :param data: array of (sufficiently close) elements in M
        :return: Fréchet mean of data points
        """
        return Mean.compute(self.M, data, max_iter=100)

    def mahalanobis_distance(self, A: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        """ Mahalanobis distance in G, see
                X. Pennec. “Intrinsic statistics on Riemannian manifolds: basic tools for
                geometric measurements”. In: J. Math. Imaging Vision 25 (2006), pp. 127–154.

        :param A: array of data points in G
        :param g: element in G
        :return: Mahalanobis distance of g to the distribution of the data points in A
        """

        S, mean = self.sample_covariance(A)
        c = self.M.tangent_coords(mean, self.M.connec.log(mean, g))

        x = jla.solve(S, c)
        return jnp.sqrt(jnp.inner(c.squeeze(), x.squeeze()))

    def hotellingT2(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """ Hotelling T^2 statistic in M, see
                P. Muralidharan and P. T. Fletcher. “Sasaki metrics for analysis of longitudinal
                data on manifolds”. In: Proceedings of the 2012 IEEE Conference on Computer
                Vision and Pattern Recognition. IEEE, 2012, pp. 1027–1034.

        :param A: array of data points in M
        :param B: array of data points in M
        :return: Hotelling T^2 statistic between the distribution of the samples in A and B
        """

        mean_A = self.frechet_mean(A)
        mean_B = self.frechet_mean(B)

        return 1/2 * (self.mahalanobis_distance(A, mean_B)**2 + self.mahalanobis_distance(B, mean_A)**2)

    def bhattacharyya(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Bhattacharyya distance in M, see
                Y. Hong et al. “Group testing for longitudinal data”. In: International
                Conference on Information Processing in Medical Imaging. Springer, 2015,
                pp. 139–151.

        :param A: array of data points in M
        :param B: array of data points in M
        :return: Bhattacharyya distance between the distribution of the samples in A and B
        """
        S_A, mean_A = self.sample_covariance(A)
        S_B, mean_B = self.sample_covariance(B)

        t = self.hotellingT2(A, B)
        h = (jnp.linalg.det(S_A) + jnp.linalg.det(S_B)) / (2 * jnp.sqrt(jnp.linalg.det(S_A) * jnp.linalg.det(S_B)))

        return 1/8 * t + 1/2 * jnp.log(h)

    def sample_covariance(self, A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Sample covariance of M–valued data
        :param A: array of data points in M
        :return: covariance matrix at the data's Fréchet mean
        """
        m = len(A)
        # mean of data
        mean = self.frechet_mean(A)

        def outer_prod(q):
            x = self.M.tangent_coords(mean, self.M.connec.log(mean, q))
            return jnp.outer(x, x)

        # covariance matrix
        S = jax.vmap(outer_prod)(A)
        S = S.sum(axis=0) / m

        return S, mean
