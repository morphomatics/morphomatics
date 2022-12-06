################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2022 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################


import numpy as np
import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold

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

        dual = False
        if mfd.dim > N:
            dual = True


        # assure mean
        if mu is None:
            mu = Mean.compute(mfd, data)
        self._mean = mu

        ################################
        # inexact PGA, aka tangent PCA
        ################################

        # map data to tangent space at mean
        v = jax.vmap(jax.jit(mfd.connec.log), (None, 0))(mu, data)

        if dual:
            # setup dual-covariance operator / (scaled) gram matrix
            # C = mfd.metric.inner(mu, v, v) / N #TODO: inner() does not support lists in general -> change to (parallel) for loops
            idx = jnp.triu_indices(N)
            C = jnp.zeros((N,N))
            C = C.at[idx].set(
                jax.vmap(jax.jit(lambda i: mfd.metric.inner(mu, v[i[0]], v[i[1]]) / N))(idx))
            C = (C.T + C) / (jnp.ones((N,N))+jnp.eye(N))

            variances, modes, coeffs = self.compute_dual(C, v)
        else:
            # setup covariance operator
            v_vec = v.reshape(N, -1)
            C = 1/N * v_vec.T @ v_vec

            variances, modes, coeffs = self.compute_cov(C, v_vec)

        self._variances = variances
        self._modes = modes
        self._coeffs = coeffs

    def compute_cov(self, C, v):
        d = self.mfd.dim
        # decompose
        vals, vecs = jnp.linalg.eigh(C)

        # set variance and modes
        n = jnp.sum(vals > 1e-6)
        e = d - n - 1 if n<d else -d-1
        variances = vals[:e:-1]
        modes = vecs[:,:e:-1].T.reshape((n,) + self.mfd.point_shape)

        coeffs = v @ vecs[:,:e:-1]

        return variances, modes, coeffs

    def compute_dual(self, C, v):
        N = C.shape[0]
        # decompose
        vals, vecs = jnp.linalg.eigh(C)

        # set variance and modes
        n = jnp.sum(vals > 1e-6)
        e = N - n - 1 if n<N else None # n<N (at least constant vector should be in kernel)
        variances = vals[:e:-1]
        modes = jnp.diag(1/jnp.sqrt(N*variances)) @ vecs[:,:e:-1].T @ v.reshape(N,-1)
        modes = modes.reshape((n,)+self.mfd.point_shape)

        # determine coefficients of input surfaces
        coeffs = vecs[:,:e:-1] @ jnp.diag(jnp.sqrt(N*variances))

        return variances, modes, coeffs

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
