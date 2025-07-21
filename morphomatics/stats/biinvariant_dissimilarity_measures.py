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

# postponed evaluation of annotations to circumvent cyclic dependencies (will be default behavior in Python 4.0)
from __future__ import annotations
from typing import Tuple

import jax

import jax.numpy as jnp
import jax.numpy.linalg as jla

from morphomatics.stats import ExponentialBarycenter as Mean


class BiinvariantDissimilarityMeasures(object):
    """
    Methods for bi-invariant statistics on Lie groups; for details see
    Hanik, Hege, and von Tycowicz (2020): Bi-invariant Two-Sample Tests in Lie Groups for Shape Analysis
    """

    def __init__(self, G: Manifold, variant: str ='left'):
        """
        :param G: Lie group
        :param variant: indicate whether all tangent vectors are left (variants='left') or right (variants='right')
        translated to the identity
        """

        self.G = G

        if variant == 'left':
            self.translation = G.group.lefttrans
        else:
            self.translation = G.group.righttrans

    def two_sample_test(self,
                        data_A: jnp.array,
                        data_B: jnp.array,
                        measure: str,
                        n_permutations: int,
                        key: jax.random.key) -> Tuple[jnp.array, jnp.array, jnp.array]:
        """Bi-invariant two-sample permutation test for data in G.
        Null hypothesis: 'Means of distributions underlying the 2 data sets are equal' if Hotelling T2 statistic is used
                         'Means and covariance underlying the 2 data sets are equal' if Bhattacharyya distance is used
        :param data_A: data array of first set; data is sorted along first axis
        :param data_B: data array of second set; data is sorted along first axis
        :param measure: indicate which measure to use; 'hotelling' and 'bhattacharyya' are possible
        :param n_permutations: number of permutations performed for the test
        :param key: random key
        :return: p-value, original distance d_orig between data, vector d-perm of distances between permuted data sets
        """
        if measure == 'hotelling':
            distMeasure = self.hotellingT2
        elif measure == 'bhattacharyya':
            distMeasure = self.bhattacharyya

        n = jnp.shape(data_A)[0]

        # distance between distributions of data
        d_orig = distMeasure(data_A, data_B)

        D = jnp.concatenate((data_A, data_B), axis=0)

        def permute_and_recompute(key_):
            # mix data
            D_perm = jax.random.permutation(key_, D)
            # distance between shuffled groups
            return distMeasure(D_perm[:n], D_perm[n:])

        # vectorize
        random_keys = jax.random.split(key, n_permutations)
        d_perm = jax.vmap(permute_and_recompute)(random_keys)

        # p-value, i.e., approximate probability of observing d_orig under the null hypothesis
        p_value = jnp.count_nonzero(d_perm > d_orig) / (n_permutations + 1)

        return p_value, d_orig, d_perm

    def groupmean(self, data: jnp.array) -> jnp.array:
        """
        :param data: array of (sufficiently close) elements in G
        :return: group mean of data points
        """
        return Mean.compute(self.G, data, max_iter=100)

    def mahalanobisdist(self, A: jnp.array, g: jnp.array) -> jnp.array:
        """ Bi-invariant Mahalanobis distance in G
        :param A: array of data points in G
        :param g: element in G
        :return: Mahalanobis distance of g to the distribution of the data points in A
        """

        S, mean = self.centralized_sample_covariance(A)
        c = self.G.group.coords(self.diff_at_e(mean, g))

        x = jla.solve(S, c)
        return jnp.sqrt(jnp.inner(c.squeeze(), x.squeeze()))

    def hotellingT2(self, A: jnp.array, B: jnp.array) -> jnp.array:
        """ Bi-invariant Hotelling T^2 statistic in G
        :param A: array of data points in G
        :param B: array of data points in G
        :return: Hotelling T^2 statistic between the distribution of the samples in A and B
        """
        m, n = len(A), len(B)
        S_pool, _, _, mean_A, mean_B = self.pooled_sample_covariance(A, B)

        c = self.G.group.coords(self.diff_at_e(mean_A, mean_B))
        x = jla.solve(S_pool, c)

        return m*n/(m+n) * jnp.inner(c.squeeze(), x.squeeze())

    def bhattacharyya(self, A: jnp.array, B: jnp.array) -> jnp.array:
        """ Bi-invariant Bhattacharyya distance in G
        :param A: array of data points in G
        :param B: array of data points in G
        :return: Bhattacharyya distance between the distribution of the samples in A and B
        """
        S_avg, S_A, S_B, mean_A, mean_B = self.averaged_sample_covariance(A, B)

        c = self.G.group.coords(self.diff_at_e(mean_A, mean_B))
        x = jla.solve(S_avg, c)

        D_B = 1 / 8 * jnp.inner(c.squeeze(), x.squeeze()) \
              + 1 / 2 * jnp.log(jla.det(S_avg) / jnp.sqrt(jla.det(S_A) * jla.det(S_B)))

        return D_B

    def centralized_sample_covariance(self, A: jnp.array) -> jnp.array:
        """ Centralized sample covariance of G–valued data
        :param A: array of data points in G
        :return: covariance matrix defined on (coordinate representations of) tangent vectors at the identity
        """
        m = len(A)
        # mean of data
        mean = self.groupmean(A)
        # compute the inverse only once
        mean_inv = self.G.group.inverse(mean)

        def outer_prod(a):
            x = self.translation(a, mean_inv)
            x = self.G.group.coords(self.G.group.log(x))
            return jnp.outer(x, x)

        # covariance matrix
        S = jax.vmap(outer_prod)(A)
        S = S.sum(axis=0) / m

        return S, mean

    def pooled_sample_covariance(self, A: jnp.array, B: jnp.array) \
            -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """Pooled sample covariance of two data sets in G.
        :param A: array of data points
        :param B: array of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        m, n = len(A), len(B)

        S_A, mean_A = self.centralized_sample_covariance(A)
        S_B, mean_B = self.centralized_sample_covariance(B)

        S_pool = 1 / (m + n - 2) * (m * S_A + n * S_B)
        return S_pool, S_A, S_B, mean_A, mean_B

    def averaged_sample_covariance(self, A: jnp.array, B: jnp.array) \
            -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """Averaged sample covariance of two data sets in G.
        :param A: array of data points
        :param B: array of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        S_A, mean_A = self.centralized_sample_covariance(A)
        S_B, mean_B = self.centralized_sample_covariance(B)

        S_avg = 1 / 2 * (S_A + S_B)
        return S_avg, S_A, S_B, mean_A, mean_B

    def diff_at_e(self, f: jnp.array, g: jnp.array) -> jnp.array:
        """ "Difference vector" between two elements in G after translating to a neighborhood of the
        identity e.
        :param f: element of G
        :param g: element of G
        :return: group logarithm after translating with f^(-1).
        """
        f_inv = self.G.group.inverse(f)
        return self.G.group.log(self.translation(g, f_inv))
