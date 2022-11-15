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

import jax
import jax.numpy as jnp

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

    def rand(self, key: jax.random.KeyArray):
        p = jax.random.normal(key, self.point_shape)
        return p / jnp.linalg.norm(p)

    def randvec(self, X, key: jax.random.KeyArray):
        H = jax.random.normal(key, self.point_shape)
        return H - jnp.dot(X.reshape(-1), H.reshape(-1)) * X

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def normalize(self, X):
        """Return Frobenius-normalized version of X in ambient space."""
        return X / jnp.linalg.norm(X)

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
            return jnp.sqrt(self.inner(p, X, X))

        def proj(self, p, X):
            return X - self.inner(p, p, X) * p

        egrad2rgrad = proj

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            # TODO?
            # return self.proj(p, H) - (jnp.transpose(p.flatten('F') )@P_G.flatten('F'))@X
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, p, X):
            return self.exp(p, X)

        def exp(self, p, X):
            # numerical safeguard
            X = self.proj(p, X)

            def full_exp(sqn):
                n = jnp.sqrt(sqn)
                return jnp.cos(n) * p + jnp.sinc(n/jnp.pi) * X

            def trunc_exp(sqn):
                #return (1-sqn/2+sqn**2/24-sqn**3/720) * p + (1-sqn/6+sqn**2/120-sqn**3/5040) * X
                # 4th-order approximation
                return (1-sqn/2+sqn**2/24) * p + (1-sqn/6+sqn**2/120) * X

            sq_norm = (X ** 2).sum()
            q = jax.lax.cond(sq_norm < 1e-6, trunc_exp, full_exp, sq_norm)
            return q

        def log(self, p, q):

            def full_log(a2):
                a = jnp.sqrt(a2)
                return 1/jnp.sinc(a/jnp.pi) * q - a/jnp.tan(a) * p

            def trunc_log(a2):
                return (1 + a2/6 + 7*a2**2/360 + 31*a2**3/15120) * q - (1 - a2/3 - a2**2/45 - a2**3/945) * p
                #return (1 + a**2/6 + 7*a**4/360) * q - (1 - a**2/3 - a**4/45) * p

            sqd = self.squared_dist(p, q)
            return jax.lax.cond(sqd < 1e-6, trunc_log, full_log, sqd)

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return (Y*Z).sum() * X - (X*Z).sum() * Y

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def transp(self, p, q, X):
            d = self.dist(p, q)
            do_transp = lambda X: X - self.inner(p, self.log(p, q), X)/d**2 * (self.log(p, q) + self.log(q, p))
            return jax.lax.cond(d < 1e-6, lambda X: X, do_transp, X)

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            inner = (p * q).sum()
            return jax.lax.cond(jnp.abs(inner) >= 1, lambda i: (i < 0)*jnp.pi, jnp.arccos, inner)

        def squared_dist(self, p, q):
            inner = (p * q).sum()
            return jax.lax.cond(inner > 1-1e-6, lambda _: jnp.sum((q-p)**2), lambda i: jnp.arccos(i)**2, inner)

        # def eval_jacobiField(self, p, q, t, X):
        #     phi = self.dist(p, q)
        #     v = self.log(p, q)
        #     gamTS = self.exp(p, t*v)
        #
        #     v = v / phi
        #     Xtan_norm = self.inner(p, X, v)
        #     Xtan = Xtan_norm * v
        #     Xorth = X - Xtan
        #
        #     # tangential component of J: (1-t) * transp(p, gamTS, Xtan)
        #     Jtan = Xtan_norm / phi * self.log(gamTS, q)
        #     return gamTS, (jnp.sin((1 - t) * phi) / jnp.sin(phi)) * Xorth + Jtan
