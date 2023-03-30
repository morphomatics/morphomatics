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
from typing import Sequence

import jax
import jax.numpy as jnp

from morphomatics.manifold import ShapeSpace, Metric, Sphere


class Kendall(ShapeSpace):
    """
    Kendall's shape space: (SO_m)-equivalence classes of preshape points (projection of centered landmarks onto the sphere)
    """

    def __init__(self, shape: Sequence[int], structure='Canonical'):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")

        # Pre-Shape space (sphere)
        self._S = Sphere(shape)
        dimension = int(self._S.dim - shape[-1] * (shape[-1] - 1) / 2)

        self.ref = None

        name = 'Kendall shape space of ' + 'x'.join(map(str, shape[:-1])) + ' Landmarks in  R^' + str(shape[-1])
        super().__init__(name, dimension, shape)
        if structure:
            getattr(self, f'init{structure}Structure')()

    def update_ref_geom(self, v):
        self.ref = self.to_coords(v)

    def to_coords(self, v):
        '''
        :arg v: array of landmark coordinates
        :return: manifold coordinates
        '''
        return Kendall.project(v)

    def from_coords(self, c):
        '''
        :arg c: manifold coords.
        :returns: array of landmark coordinates
        '''
        return c

    @property
    def ref_coords(self):
        """ :returns: Coordinates of reference shape """
        return self.ref

    def rand(self, key: jax.random.KeyArray):
        p = jax.random.normal(key, self.point_shape)
        return self.project(p)

    def randvec(self, p, key: jax.random.KeyArray):
        v = jax.random.normal(key, self.point_shape)
        v = self.center(v)
        return self.horizontal(p, self._S.proj(p, v))

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    @staticmethod
    def wellpos(x, y):
        """
        Rotate y such that it aligns to x.
        :param x: (centered) reference landmark configuration.
        :param y: (centered) landmarks to be aligned.
        :returns: y well-positioned to x.
        """
        m = x.shape[-1]
        sigma = jnp.ones(m)
        # full_matrices=False equals full_matrices=True for quadratic input but allows for auto diff
        u, _, v = jnp.linalg.svd(x.reshape(-1, m).T @ y.reshape(-1, m), full_matrices=False)
        sigma = sigma.at[-1].set(jnp.sign(jnp.linalg.det(u @ v)))
        return jnp.einsum('...i,ji,j,kj', y, v, sigma, u)

    @staticmethod
    def center(x):
        """
        Remove mean from x.
        """
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        return x - mean

    @staticmethod
    def project(x):
        """
        Project to pre-shape space.
        : param x: Point to project.
        :returns: Projected x.
        """
        x = Kendall.center(x)
        return x / jnp.linalg.norm(x)

    def proj(self, p, X):
        """ Project a vector X from the ambient Euclidean space onto the tangent space at p. """
        X = Kendall.center(X)
        # TODO: think about naming convention.
        return Kendall.horizontal(p, self._S.proj(p, X))

    @staticmethod
    def vertical(p, X):
        """
        Compute vertical component of X at base point p by solving the sylvester equation
        App^T+pp^TA = Xp^T-pX^T for A. If p has full rank (det(pp^T) > 0), then there exists a unique solution
        A, it is skew-symmetric and Ap is the vertical component of X
        """
        d = p.shape[-1]
        S = p.reshape(-1, d).T @ p.reshape(-1, d)
        rhs = X.reshape(-1, d).T @ p.reshape(-1, d)
        rhs = rhs.T - rhs
        S = jnp.kron(jnp.eye(d), S) + jnp.kron(S, jnp.eye(d))
        A, *_ = jnp.linalg.lstsq(S, rhs.reshape(-1))
        return jnp.einsum('...i,ij', p, A.reshape(d, d))

    @staticmethod
    def horizontal(p, X):
        """
        compute horizontal component of X.
        """
        return X - Kendall.vertical(p, X)

    def initCanonicalStructure(self):
        """
        Instantiate the preshape sphere with canonical structure.
        """
        structure = Kendall.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

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
            self._S = M._S

        def __str__(self):
            return "canonical structure"

        @property
        def typicaldist(self):
            return np.pi/2

        def inner(self, p, X, Y):
            return self._S.metric.inner(p, X, Y)

        def norm(self, p, X):
            return self._S.metric.norm(p, X)

        def flat(self, p, X):
            """Lower vector X at p with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, p, dX):
            """Raise covector dX at p with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def egrad2rgrad(self, p, X):
            return self._M.proj

        def ehess2rhess(self, p, G, H, X):
            """ Convert the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            Y = self._S.metric.ehess2rhess(p, G, H, X)
            return Kendall.horizontal(p, Y)

        def exp(self, p, X):
            return self._S.connec.exp(p, X)

        retr = exp

        def log(self, p, q):
            q = Kendall.wellpos(p, q)
            return self._S.connec.log(p, q)

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def transp(self, p, q, X):
            # TODO
            return None

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            q = Kendall.wellpos(p, q)
            return self._S.metric.dist(p, q)

        def squared_dist(self, p, q):
            q = Kendall.wellpos(p, q)
            return self._S.metric.squared_dist(p, q)

        def eval_jacobiField(self, p, q, t, X):
            # return self.proj(*super().eval_jacobiField(p, q, t, X))

            # q = Kendall.wellpos(p, q)
            # phi = self._S.metric.dist(p, q)
            # v = self._S.connec.log(p, q)
            # gamTS = self._S.connec.exp(p, t * v)
            #
            # v = v / self._S.metric.norm(p, v)
            # Xtan = self._S.metric.inner(p, X, v) * v
            # Xorth = X - Xtan
            #
            # # TODO: Xtan not jet at gamTS
            # return gamTS, (np.sin((1 - t) * phi) / np.sin(phi)) * Kendall.horizontal(gamTS, Xorth) + (1 - t) * Xtan

            raise NotImplementedError('This function has not been implemented yet.')

        def eval_adjJacobi(self, p, q, t, X):
            # return self.proj(p, super().eval_adjJacobi(p, q, t, X))

            raise NotImplementedError('This function has not been implemented yet.')
