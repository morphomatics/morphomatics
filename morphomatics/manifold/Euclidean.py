################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2023 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Metric


class Euclidean(Manifold):
    """The euclidean space of [... x k x m]-tensors .
    """

    def __init__(self, point_shape=(3,), structure='Canonical'):
        name = 'Euclidean space of dimension ' + 'x'.join(map(str, point_shape))
        dimension = np.prod(point_shape)
        super().__init__(name, dimension, point_shape)
        if structure:
            getattr(self, f'init{structure}Structure')()

    def initCanonicalStructure(self):
        """
        Instantiate Euclidean space with canonical structure.
        """
        structure = Euclidean.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    def rand(self, key: jax.random.KeyArray):
        return jax.random.normal(key, self.point_shape)

    def randvec(self, X, key: jax.random.KeyArray):
        return jax.random.normal(key, self.point_shape)

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, p, X):
        return X

    class CanonicalStructure(Metric):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "Canonical euclidean structure"

        @property
        def typicaldist(self):
            return jnp.sqrt(self.dim)

        def inner(self, p, X, Y):
            return euclidean_inner(X, Y)

        def flat(self, p, X):
            return X

        def sharp(self, p, dX):
            return dX

        def norm(self, p, X):
            return jnp.sqrt(self.inner(p, X, X))

        def egrad2rgrad(self, p, X):
            return self._M.proj(p, X)

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            return H

        def retr(self, p, X):
            return self.exp(p, X)

        def exp(self, p, X):
            return p + X

        def log(self, p, q):
            return q - p

        def curvature_tensor(self, p, X, Y, Z):
            return jnp.zeros(self._M.point_shape)

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def transp(self, p, q, X):
            return X

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            return jnp.linalg.norm(q - p)

        def eval_jacobiField(self, p, q, t, X):
            return (1-t) * X

        def eval_adjJacobi(self, p, q, t, X):
            return 1/(1-t) * X


def euclidean_inner(X, Y):
    return (X * Y).sum()
