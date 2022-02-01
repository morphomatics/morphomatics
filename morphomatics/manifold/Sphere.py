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
import numpy.random as rnd
import numpy.linalg as la

from morphomatics.manifold import Manifold, Metric, Connection
from morphomatics.manifold.util import gram_schmidt


class Sphere(Manifold):
    """The n-dimensional sphere embedded in R(n+1)

    Elements are represented as normalized (row) vectors of length n + 1.
    """

    def __init__(self, n=2, structure='Canonical'):

        name = 'Sphere S({n})'.format(n=n)
        dimension = n
        point_shape = [n + 1]
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def initCanonicalStructure(self):
        """
        Instantiate S2 with canonical structure.
        """
        structure = Sphere.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    def rand(self):
        p = np.random.rand(3)
        return p / np.linalg.norm(p)

    def randvec(self, X):
        H = np.random.rand(3)
        return H - np.dot(X, H) * X

    def zerovec(self):
        return np.zeros(3)

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

        @property
        def __str__(self):
            return "S2-canonical structure"

        @property
        def typicaldist(self):
            return np.pi

        @staticmethod
        def antipode(p):
            return -p

        def inner(self, p, X, Y):
            return np.inner(X, Y)

        def norm(self, p, X):
            return np.sqrt(self.inner(p, X, X))

        def proj(self, p, X):
            return X - self.inner(p, p, X) * p

        egrad2rgrad = proj

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            # TODO
            return

        def retr(self, p, X):
            return self.exp(p, X)

        def exp(self, p, X):
            if la.norm(X) < 1e-15:
                return p
            else:
                q = np.cos(la.norm(X)) * p + np.sin(la.norm(X)) * X/la.norm(X)
                return q / np.linalg.norm(q)

        def log(self, p, q):
            # assert not np.allclose(p, -q)

            if np.allclose(p, q):
                return self.zerovec()
            else:
                X = self.dist(p, q) * (q - np.inner(p, q) * p)/la.norm(q - np.inner(p, q) * p)
                return self.proj(p, X)

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def rand(self, sig=1):
            v = np.random.normal(size=self.dim + 1)
            return sig * v/la.norm(v)

        def randvec(self, p, sig=1):
            n = sig * np.random.normal(size=self._M.dim  + 1)  # Gaussian in embedding space
            return n - np.inner(n, p) * p  # project to TpM (keeps Gaussianness)

        def zerovec(self):
            return np.zeros(self._M.dim + 1)

        def transp(self, p, q, X):
            if np.allclose(p, q) or self.dist(p, q) < 1e-13:
                return X
            else:
                return X - self.inner(p, self.log(p, q), X)/self.dist(p, q)**2 * (self.log(p, q) + self.log(q, p))

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            # make sure inner product is between -1 and 1
            inner = max(min(self.inner(None, p, q), 1), -1)
            return np.arccos(inner)

        def jacONB(self, p, q):
            """Let Jac be the Jacobi operator along the geodesic from p to q. This code diagonalizes J.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param p: element of S^n
            :param q: element of S^n
            :returns lam, G: eigenvalues and orthonormal eigenbasis of Jac at R
            """

            # eigenvector w.r.t. eigenvalue lambda = 0
            v = self.log(p, q)
            v /= self.norm(p, v)

            # tangent space is n-dimensional
            G = np.zeros((self._M.dim + 1, self._M.dim ))
            G[:, 0] = v
            # probably not very good
            for i in range(1, self._M.dim ):
                G[:, i] = self.randvec(p)

            G = gram_schmidt(G)

            lam = np.ones(self._M.dim)
            lam[0] = 0

            return lam, G.transpose()

        def jacobiField(self, p, q, t, X):
            # TODO
            return

        def adjJacobi(self, p, q, t, X):
            """Evaluates an adjoint Jacobi field along the geodesic gam from p to q.

            See "A variational model for data fitting on manifolds by minimizing the acceleration of a Bézier curve"
            (Bergmann, Gousenbourger) for details.

            :param p: element of S^n
            :param q: element of S^n
            :param t: scalar in [0,1]
            :param X: tangent vector at gam(t)
            :return: tangent vector at p
            """
            assert p.shape == q.shape == X.shape and np.isscalar(t)

            if t == 0:
                return X

            # gam(t)
            o = self.geopoint(p, q, t)

            # orthonormal eigenvectors of the Jacobi operator that generate the parallel frame field along gam and the
            # eigenvalues
            lam, F = self.jacONB(p, q)

            # expand X w.r.t. F at gam(t)
            Fp = np.zeros_like(F)
            alpha = np.zeros_like(lam)
            weights = np.ones_like(lam)

            for i in range(self._M.dim):
                Fp[i] = self.transp(p, o, F[i])
                # calculate coefficients of X w.r.t. to the parallel translated basis in T_gam(t)S^n
                alpha[i] = self.inner(o, Fp[i], X)

            # weights for the linear combination of the three basis fields
            weights = np.sin(self.dist(p, q) * (1 - t) * np.sqrt(lam[1])) / np.sin(self.dist(p, q) * np.sqrt(lam[1])) * weights
            weights[0] = 1 - t

            # evaluate the adjoint Jacobi field at p
            for i in range(self._M.dim ):
                F[i] = alpha[i] * weights[i] * F[i]

            return self.proj(p, np.sum(F, axis=0))

        def adjDxgeo(self, p, q, t, X):
            """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t the starting point p at X,
            i.e, the adjoint  of d_p gamma(t; ., q) applied to X, which is en element of the tangent space at gamma(t).
            """
            assert p.shape == q.shape == X.shape and np.isscalar(t)

            return self.adjJacobi(p, q, t, X)

        def adjDygeo(self, p, q, t, X):
            """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t the endpoint q at X,
            i.e, the adjoint  of d_q gamma(t; p, .) applied to X, which is en element of the tangent space at gamma(t).
            """
            assert p.shape == q.shape == X.shape and np.isscalar(t)

            return self.adjJacobi(q, p, 1 - t, X)

        def projToGeodesic(self, p, q, r, max_iter=10):
            # TODO
            return
