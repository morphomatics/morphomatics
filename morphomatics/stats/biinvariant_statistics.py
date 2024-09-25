################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random

from morphomatics.manifold import LieGroup
from morphomatics.stats import ExponentialBarycenter as Mean


class BiinvariantStatistics(object):
    """
    Methods for bi-invariant statistics on Lie groups; for details see
    Hanik, Hege, and von Tycowicz (2020): Bi-invariant Two-Sample Tests in Lie Groups for Shape Analysis
    """

    def __init__(self, G: LieGroup, variant='left'):
        """
        :param G: Lie group
        :param variant: indicate whether all tangent vectors are left (variants='left') or right (variants='right')
        translated to the identity
        """

        self.G = G

        self.variant = variant

    def two_sample_test(self, data_A, data_B, measure, n_permutations=10000):
        """Bi-invariant two-sample permutation test for data in G.
        Null hypothesis: 'Means of distributions underlying the 2 data sets are equal' if Hotelling T2 statistic is used
                         'Means and covariance underlying the 2 data sets are equal' if Bhattacharyya distance is used
        :param data_A: data array of first set; data is sorted along first axis
        :param data_B: data array of second set; data is sorted along first axis
        :param measure: indicate which measure to use; 'hotelling' and 'bhattacharyya' are possible
        :param n_permutations: number of permutations performed for the test
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
        d_perm = []
        # count how often d_orig is smaller than distance between randomly shuffled groups
        counter = 0

        # def shuffle(A):
        #     A_perm = np.random.permutation(A)
        #     return distMeasure(A_perm[:n], A_perm[n:])
        #
        # with Parallel(n_jobs=-1, prefer='threads', verbose=0) as parallel:
        #     d_perm = parallel(delayed(shuffle)(D) for _ in range(n_permutations))

        key = random.PRNGKey(0)
        # permute and recompute
        for i in range(n_permutations):
            key, subkey = random.split(key)
            # permute along first axis
            D_perm = random.permutation(key, D)
            # distance between shuffled groups
            d_perm_i = distMeasure(D_perm[:n], D_perm[n:])
            d_perm.append(d_perm_i)

            # increase count if d_orig < d_perm_i
            if d_orig < d_perm_i:
                counter += 1

        for d_perm_i in d_perm:
            if d_orig < d_perm_i:
                counter = counter + 1

        return counter / (n_permutations + 1), d_orig, d_perm

    def groupmean(self, data):
        """
        :param data: array of elements in G (sufficiently close)
        :return: group mean of data points
        """
        return Mean.compute(self.G, data)

    def mahalanobisdist(self, A, g):
        """ Bi-invariant Mahalanobis distance in G
        :param A: array of data points in G
        :param g: element of G
        :return: Mahalanobis distance of g to the distribution of the data points in A
        """

        C, mean = self.centralized_sample_covariance(A)
        x = self.G.group.coords(self.diff_at_e(mean, g))

        return float(jnp.sqrt(x.transpose() @ jla.inv(C) @ x).squeeze())

    def hotellingT2(self, A, B):
        """ Bi-invariant Hotelling T^2 statistic in G
        :param A: array of data points in G
        :param B: array of data points in G
        :return: Hotelling T^2 statistic between the distribution of the samples in A and B
        """
        m, n = len(A), len(B)
        C_pool, _, _, mean_A, mean_B = self.pooled_sample_covariance(A, B)
        x = self.G.group.coords(self.diff_at_e(mean_A, mean_B))

        return m*n/(m+n) * float((x.transpose() @ jla.inv(C_pool) @ x).squeeze())

    def bhattacharyya(self, A, B):
        """ Bi-invariant Bhattacharyya distance in G
        :param A: array of data points in G
        :param B: array of data points in G
        :return: Bhattacharyya distance between the distribution of the samples in A and B
        """
        C_avg, C_A, C_B, mean_A, mean_B = self.averaged_sample_covariance(A, B)
        x = self.G.group.coords(self.diff_at_e(mean_A, mean_B))
        D_B = 1/8 * x.transpose() @ jla.inv(C_avg) @ x \
              + 1/2 * jnp.log(jla.det(C_avg) / jnp.sqrt(jla.det(C_A) * jla.det(C_B)))

        return float(D_B.squeeze())

    def centralized_sample_covariance(self, A):
        """ Centralized sample covariance of G–valued data
        :param A: array of data points in G
        :return: covariance matrix defined on (coordinate representations of) tangent vectors at the identity
        """
        m = len(A)
        # mean of data
        mean = self.groupmean(A)
        # inverse only once
        mean_inv = self.G.group.inverse(mean)
        # set up covariance matrix
        C = jnp.zeros((self.G.dim, self.G.dim))

        for a in A:
            x = self.G.group.coords(self.diff_at_e(mean, a))
            C = C + x @ x.transpose()

        return 1/m * C, mean

    def pooled_sample_covariance(self, A, B):
        """Pooled sample covariance of two data sets in G.
        :param A: array of data points
        :param B: array of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        m, n = len(A), len(B)

        C_A, mean_A = self.centralized_sample_covariance(A)
        C_B, mean_B = self.centralized_sample_covariance(B)

        C_pool = 1/(m+n-2) * (m * C_A + n * C_B)
        return C_pool, C_A, C_B, mean_A, mean_B

    def averaged_sample_covariance(self, A, B):
        """Averaged sample covariance of two data sets in G.
        :param A: array of data points
        :param B: array of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        C_A, mean_A = self.centralized_sample_covariance(A)
        C_B, mean_B = self.centralized_sample_covariance(B)

        C_avg = 1/2 * (C_A + C_B)
        return C_avg, C_A, C_B, mean_A, mean_B

    def diff_at_e(self, a, b):
        """ "Difference vector" between two elements in G after translating to a neighborhood of the
        identity e.
        :param a: element of G
        :param b: element of G
        :return: group logarithm after translating such that a is mapped e.
        """
        if self.variant == 'left':
            x = self.G.group.log(self.G.group.lefttrans(b, self.G.group.inverse(a)))
        else:
            x = self.G.group.log(self.G.group.righttrans(b, self.G.group.inverse(a)))
        return x
