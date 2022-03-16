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

from morphomatics.manifold import Manifold, Metric, Connection


class Sphere(Manifold):
    """The sphere of [... x k x m]-tensors embedded in R(n+1)
    Elements are represented as normalized (row) vectors of length n + 1.
    """

    def __init__(self, point_shape=(3,), structure='Canonical'):
        name = 'Points with unit Frobenius norm in ' +\
               'x'.join(map(str, point_shape)) + '-dim. space.'
        dimension = np.prod(point_shape)-1
        super().__init__(name, dimension, point_shape)
        if structure:
            getattr(self, f'init{structure}Structure')()

    def initCanonicalStructure(self):
        """
        Instantiate Sphere with canonical structure.
        """
        structure = Sphere.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    def rand(self):
        # key = jax.random.PRNGKey(np.random.randint(1 << 32))
        # p = jax.random.normal(key, self.point_shape)
        p = np.random.rand(self.point_shape)
        return p / np.linalg.norm(p)

    def randvec(self, X):
        # key = jax.random.PRNGKey(np.random.randint(1 << 32))
        # H = jax.random.normal(key, self.point_shape)
        H = np.random.rand(self.point_shape)  # Normalization?
        return H - np.dot(X.reshape(-1), H.reshape(-1)) * X

    def zerovec(self):
        return np.zeros(self.point_shape)

    def normalize(self, X):
        """Return Frobenius-normalized version of X in ambient space."""
        return X / np.linalg.norm(X)

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

        def __str__(self):
            return "Canonical structure"

        @property
        def typicaldist(self):
            return np.pi

        @staticmethod
        def antipode(p):
            return -p

        def inner(self, p, X, Y):
            return (X*Y).sum()

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
            # TODO?
            # return self.proj(p, H) - (np.transpose(p.flatten('F') )@G.flatten('F'))@X
            return

        def retr(self, p, X):
            return self.exp(p, X)


        def exp(self, p, X):

            X = self.proj(p, X)

            def full_exp(sqn):
                n = np.sqrt(sqn + np.finfo(np.float64).eps)
                return np.cos(n) * p + np.sinc(n/np.pi) * X

            def trunc_exp(sqn):
                #return (1-sqn/2+sqn**2/24-sqn**3/720) * p + (1-sqn/6+sqn**2/120-sqn**3/5040) * X
                # 4th-order approximation
                return (1-sqn/2+sqn**2/24) * p + (1-sqn/6+sqn**2/120) * X

            sq_norm = (X ** 2).sum()
            sq_norm = np.where(sq_norm < 0, 0, sq_norm)
            q = np.where(sq_norm < 1e-6, trunc_exp(sq_norm), full_exp(sq_norm))
            return q

        def log(self, p, q):

            def full_log(a2):
                a = np.sqrt(a2 + np.finfo(np.float64).eps)
                return 1/np.sinc(a/np.pi) * q - a/np.tan(a) * p

            def trunc_log(a2):
                return (1 + a2/6 + 7*a2**2/360 + 31*a2**3/15120) * q - (1 - a2/3 - a2**2/45 - a2**3/945) * p
                #return (1 + a**2/6 + 7*a**4/360) * q - (1 - a**2/3 - a**4/45) * p

            sqd = self.squared_dist(p, q)
            return np.where(sqd < 1e-6, trunc_log(sqd), full_log(sqd))

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def transp(self, p, q, X):
            d = self.dist(p, q)
            do_transp = lambda X: X - self.inner(p, self.log(p, q), X)/d**2 * (self.log(p, q) + self.log(q, p))
            if d < 1e-6:
                return X
            else:
                return do_transp(X)

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            inner = (p * q).sum()
            inner_ = np.where(np.abs(inner) >= 1, np.sign(inner), inner)
            return np.where(np.abs(inner) >= 1, (inner < 0)*np.pi, np.arccos(inner_))

        def squared_dist(self, p, q):
            inner = (p * q).sum()
            inner_ = np.where(inner > 1-1e-6, 1-1e-6, inner)
            return np.where(inner > 1-1e-6, np.sum((q-p)**2), np.arccos(inner_)**2)

        def eval_jacobiField(self, p, q, t, X):
            phi = self.dist(p, q)
            v = self.log(p, q)

            v = v / self.norm(p, v)
            Xtan = self.inner(p, X, v) * v
            Xorth = X - Xtan

            return (np.sin((1 - t) * phi) / np.sin(phi)) * Xorth + (1 - t) * Xtan

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
