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
from typing import Sequence

from morphomatics.manifold import ShapeSpace, Metric, Connection, Sphere

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

    def rand(self):
        p = np.random.rand(self.point_shape)
        return self.project(p)

    def randvec(self, p):
        v = np.random.rand(self.point_shape)
        v = self.center(v)
        return self.horizontal(p, self._S.metric.proj(p, v))

    def zerovec(self):
        return np.zeros(self.point_shape)

    @staticmethod
    def wellpos(x, y):
        """
        Rotate y such that it aligns to x.
        :param x: (centered) reference landmark configuration.
        :param y: (centered) landmarks to be aligned.
        :returns: y well-positioned to x.
        """
        m = x.shape[-1]
        sigma = np.ones(m)
        # full_matrices=False equals full_matrices=True for quadratic input but allows for auto diff
        u, _, v = np.linalg.svd(x.reshape(-1, m).T @ y.reshape(-1, m), full_matrices=False)
        sigma[-1] = np.sign(np.linalg.det(u @ v))
        return np.einsum('...i,ji,j,kj', y, v, sigma, u)

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
        return x / np.linalg.norm(x)

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
        S = np.kron(np.eye(d), S) + np.kron(S, np.eye(d))
        A, *_ = np.linalg.lstsq(S, rhs.reshape(-1))
        return np.einsum('...i,ij', p, A.reshape(d, d))

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

    class CanonicalStructure(Metric, Connection):
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

        def proj(self, p, X):
            """ Project a vector X from the ambient Euclidean space onto the tangent space at p. """
            X = Kendall.center(X)
            return Kendall.horizontal(p, self._S.metric.proj(p, X))

        egrad2rgrad = proj

        def ehess2rhess(self, p, G, H, X):
            """ Convert the Euclidean gradient G and Hessian H of a function at
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

        def eval_jacobiField(self, p, q, t, X):
            q = Kendall.wellpos(p, q)
            phi = self._S.metric.dist(p, q)
            v = self._S.connec.log(p, q)
            gamTS = self._S.connec.exp(p, t*v)

            v = v / self._S.metric.norm(p, v)
            Xtan = self._S.metric.inner(p, X, v) * v
            Xorth = X - Xtan

            return (np.sin((1-t) * phi) / np.sin(phi)) * Kendall.horizontal(gamTS, Xorth) + (1-t)*Xtan

        def jacobiField(self, p, q, t, X):
            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                return self.eval_jacobiField(p, q, t, X)

        def adjJacobi(self, p, q, t, X):
            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                return self.eval_jacobiField(self.geopoint(p, q, t), q, -t / (1 - t), X)
