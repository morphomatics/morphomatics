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

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Metric, LieGroup


class Euclidean(Manifold):
    """The Euclidean space
    """

    def __init__(self, point_shape=(3,), structure='Canonical'):
        name = 'Euclidean space of dimension ' + 'x'.join(map(str, point_shape))
        dimension = np.prod(point_shape)
        super().__init__(name, dimension, point_shape)
        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children, aux+(self.point_shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *aux_data, shape = aux_data
        obj = cls(shape, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initCanonicalStructure(self):
        """
        Instantiate Euclidean space with canonical structure.
        """
        structure = Euclidean.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    def rand(self, key: jax.Array):
        return jax.random.normal(key, self.point_shape)

    def randvec(self, X, key: jax.Array):
        return jax.random.normal(key, self.point_shape)

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, x, X):
        return X

    class CanonicalStructure(Metric, LieGroup):
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

        def retr(self, x, X):
            return self.exp(x, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            return jax.lax.cond(len(argv) == 1,
                                lambda A: A[-1],
                                lambda A:  A[-1] + A[0],
                                (argv[0], argv[-1]))

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            return jax.lax.cond(len(argv) == 1,
                                     lambda A: A[-1],
                                     lambda A: A[-1]- A[0],
                                     argv)

        def curvature_tensor(self, x, X, Y, Z):
            return jnp.zeros(self._M.point_shape)

        def geopoint(self, x, y, t):
            return x + t * (y - x)

        @property
        def identity(self):
            return jnp.zeros(self._M.point_shape)

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

        def lefttrans(self, g, f):
            """Left translation of g by f.
            """
            return f+g

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return g+f

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return -g

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return X

        def coords_inv(self, c):
            """Inverse of coords"""
            return self.coords(c)

        def bracket(self, X, Y):
            return self.identity

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')

def euclidean_inner(X, Y):
    return (X * Y).sum()
