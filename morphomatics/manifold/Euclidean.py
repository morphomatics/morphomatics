################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
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
    """The Euclidean space
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

    def proj(self, x, X):
        return X

    class CanonicalStructure(Metric):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^nxn)^k endowed with the usual trace inner product.
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
            return jnp.sqrt(self._M.dim)

        def inner(self, x, X, Y):
            return euclidean_inner(X, Y)

        def flat(self, p, X):
            return X

        def sharp(self, p, dX):
            return dX

        def norm(self, x, X):
            return jnp.linalg.norm(X)

        def egrad2rgrad(self, x, X):
            return X

        def ehess2rhess(self, x, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            return H

        def retr(self, x, X):
            return self.exp(x, X)

        def exp(self, x, X):
            return x + X

        def log(self, x, y):
            return y - x

        def curvature_tensor(self, x, X, Y, Z):
            return jnp.zeros(self._M.point_shape)

        def geopoint(self, x, y, t):
            return x + t * (y - x)

        def transp(self, x, y, X):
            return X

        def pairmean(self, x, y):
            return self.geopoint(x, y, .5)

        def dist(self, x, y):
            return jnp.linalg.norm(y - x)

        def squared_dist(self, x, y):
            return jnp.sum((y-x)**2)

        def jacobiField(self, x, y, t, X):
            return [self.geopoint(x, y, t), (1-t) * X]

        def adjJacobi(self, x, y, t, X):
            return 1/(1-t) * X


def euclidean_inner(X, Y):
    return (X * Y).sum()
