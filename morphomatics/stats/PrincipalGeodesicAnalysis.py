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

from pymanopt.manifolds.manifold import Manifold

from . import ExponentialBarycenter as Mean

class PrincipalGeodesicAnalysis(object):
    """
    Principal Geodesic Analysis (PGA) as introduced by
    Fletcher et al. (2003): Statistics of manifold via principal geodesic analysis on Lie groups.
    """

    def __init__(self, mfd: Manifold, data, mu=None):
        """
        Setup PGA.

        :arg mfd: underlying data space (Assumes that mfd#inner(...) supports list of vectors)
        :arg data: list of data points
        :arg mu: intrinsic mean of data
        """
        assert mfd.connec and mfd.metric
        self.mfd = mfd
        N = len(data)

        # assure mean
        if mu is None:
            mu = Mean.compute(mfd, data)
        self._mean = mu

        ################################
        # inexact PGA, aka tangent PCA
        ################################

        # map data to tangent space at mean
        v = [mfd.connec.log(mu, x) for x in data]

        # setup dual-covariance operator / (scaled) gram matrix
        C = mfd.metric.inner(mu, v, v) / N #TODO: inner() does not support lists in general -> change to (parallel) for loops

        # decompose
        vals, vecs = np.linalg.eigh(C)

        # set variance and modes
        n = np.sum(vals > 1e-6)
        e = N - n - 1 if n<N else None # n<N (at least constant vector should be in kernel)
        self._variances = vals[:e:-1]
        self._modes = np.diag(1/np.sqrt(N*self._variances)) @ vecs[:,:e:-1].T @ v

        # determine coefficients of input surfaces
        self._coeffs = vecs[:,:e:-1] @ np.diag(np.sqrt(N*self._variances))

    @property
    def mean(self):
        """
        :return: Intrinsic mean.
        """
        return self._mean

    @property
    def modes(self):
        """
        :return: Principal geodesic modes of variation.
        """
        return self._modes

    @property
    def variances(self):
        """
        :return: Variances.
        """
        return self._variances

    @property
    def coeffs(self):
        """
        :return: Coefficients of data w.r.t. principal modes.
        """
        return self._coeffs
