################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2021 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np
import numpy.linalg as la


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

    def groupmean(self, data):
        """
        :param data: list of elements in G (sufficiently close)
        :return: group mean of data points
        """
        return Mean.compute(self.G, data)

    def mahalanobisdist(self, A, g):
        """ Bi-invariant Mahalanobis distance
        :param A: list of data points in G
        :param g: element of G
        :return: Mahalanobis distance of g to the distribution of the data points in A
        """

        C, mean = self.centralized_sample_covariance(A)
        if self.variant == 'left':
            x = self.G.group.coords(self.G.group.log(self.G.group.lefttrans(g, self.G.group.inverse(mean))))
        else:
            x = self.G.group.coords(self.G.group.log(self.G.group.righttrans(g, self.G.group.inverse(mean))))

        return np.asscalar(np.sqrt(x.transpose() @ la.inv(C) @ x))

    def hotellingT2(self, A, B):
        """ Bi-invariant Hotelling T^2 statistic
        :param A: list of data points in G
        :param B: list of data points in G
        :return: Hotelling T^2 statistic between the distribution of the samples in A and B
        """
        m, n = len(A), len(B)
        C_pool, _, _, mean_A, mean_B = self.pooled_sample_covariance(A, B)
        if self.variant == 'left':
            x = self.G.group.coords(self.G.group.log(self.G.group.lefttrans(mean_B, self.G.group.inverse(mean_A))))
        else:
            x = self.G.group.coords(self.G.group.log(self.G.group.righttrans(mean_B, self.G.group.inverse(mean_A))))

        return m*n/(m+n) * np.asscalar(x.transpose() @ la.inv(C_pool) @ x)

    def bhattacharyya(self, A, B):
        """ Bi-invariant Bhattacharyya distance
        :param A: list of data points in G
        :param B: list of data points in G
        :return: Bhattacharyya distance between the distribution of the samples in A and B
        """
        C_avg, C_A, C_B, mean_A, mean_B = self.averaged_sample_covariance(A, B)
        if self.variant == 'left':
            x = self.G.group.coords(self.G.group.log(self.G.group.lefttrans(mean_B, self.G.group.inverse(mean_A))))
        else:
            x = self.G.group.coords(self.G.group.log(self.G.group.righttrans(mean_B, self.G.group.inverse(mean_A))))
        D_B = 1/8 * x.transpose() @ la.inv(C_avg) @ x + 1/2 * np.log(la.det(C_avg) / np.sqrt(la.det(C_A) * la.det(C_B)))

        return np.asscalar(D_B)

    def centralized_sample_covariance(self, A):
        """ Centralized sample covariance of G-valued data
        :param A: list of data points in G
        :return: covariance matrix defined on (coordinate representations of) tangent vectors vectors at the identity
        """
        m = len(A)
        # mean of data
        mean = self.groupmean(A)
        # inverse only once
        mean_inv = self.G.group.inverse(mean)
        # set up covariance matrix
        C = np.zeros((self.G.dim, self.G.dim))

        for a in A:
            if self.variant == 'left':
                x = self.G.group.log(self.G.group.lefttrans(a, mean_inv))
            else:
                x = self.G.group.log(self.G.group.righttrans(a, mean_inv))
            x = self.G.group.coords(x)
            C += x @ x.transpose()

        return 1/m * C, mean

    def pooled_sample_covariance(self, A, B):
        """Pooled sample covariance of two data sets in a Lie group.
        :param A: list of data points
        :param B: list of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        m, n = len(A), len(B)

        C_A, mean_A = self.centralized_sample_covariance(A)
        C_B, mean_B = self.centralized_sample_covariance(B)

        C_pool = 1/(m+n-2) * (m * C_A + n * C_B)
        return C_pool, C_A, C_B, mean_A, mean_B

    def averaged_sample_covariance(self, A, B):
        """Averaged sample covariance of two data sets in a Lie group.
        :param A: list of data points
        :param B: list of data points
        :return: covariance operator acting on vectors in the tangent space at the identity
        """
        C_A, mean_A = self.centralized_sample_covariance(A)
        C_B, mean_B = self.centralized_sample_covariance(B)

        C_avg = 1/2 * (C_A + C_B)
        return C_avg, C_A, C_B, mean_A, mean_B
