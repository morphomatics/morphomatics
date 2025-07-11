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

from morphomatics.manifold import Manifold, Metric
from morphomatics.manifold.euclidean import euclidean_inner
from morphomatics.manifold.metric import _eval_adjJacobi_embed


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
        Instantiate Sphere with canonical structure.
        """
        structure = Sphere.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    def rand(self, key: jax.Array):
        p = jax.random.normal(key, self.point_shape)
        return p / jnp.linalg.norm(p)

    def randvec(self, X, key: jax.Array):
        H = jax.random.normal(key, self.point_shape)
        return H - jnp.dot(X.reshape(-1), H.reshape(-1)) * X

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    @staticmethod
    def antipode(p):
        return -p

    @staticmethod
    def normalize(X):
        """Return Frobenius-normalized version of X in ambient space."""
        return X / jnp.sqrt((X**2).sum() + np.finfo(np.float64).eps)

    def proj(self, p, X):
        return X - euclidean_inner(p, X) * p

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
            return "Canonical structure"

        @property
        def typicaldist(self):
            return np.pi

        def inner(self, p, X, Y):
            return euclidean_inner(X, Y)

        def norm(self, p, X):
            return jnp.sqrt(self.inner(p, X, X))

        def flat(self, p, X):
            """Lower vector X at p with the metric"""
            return X

        def sharp(self, p, dX):
            """Raise covector dX at p with the metric"""
            return dX

        def egrad2rgrad(self, p, X):
            return self._M.proj(p, X)

        def retr(self, p, X):
            return self.exp(p, X)

        def exp(self, p, X):
            # numerical safeguard
            p = Sphere.normalize(p)
            X = self._M.proj(p, X)

            def full_exp(sqn):
                n = jnp.sqrt(sqn + jnp.finfo(jnp.float64).eps)
                return jnp.cos(n) * p + jnp.sinc(n/jnp.pi) * X

            def trunc_exp(sqn):
                #return (1-sqn/2+sqn**2/24-sqn**3/720) * p + (1-sqn/6+sqn**2/120-sqn**3/5040) * X
                # 4th-order approximation
                return (1-sqn/2+sqn**2/24) * p + (1-sqn/6+sqn**2/120) * X

            sq_norm = (X ** 2).sum()
            q = jax.lax.cond(sq_norm < 1e-6, trunc_exp, full_exp, sq_norm)
            return Sphere.normalize(q)

        def log(self, p, q):

            def full_log(a2):
                a = jnp.sqrt(a2 + jnp.finfo(jnp.float64).eps)
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
            d2 = self.squared_dist(p, q)
            def do_transp(V):
                log_p_q = self.log(p, q)
                return V - self.inner(p, log_p_q, V)/d2 * (log_p_q + self.log(q, p))
            return jax.lax.cond(d2 < 1e-6, lambda X: X, do_transp, X)

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            inner = (p * q).sum()
            return jax.lax.cond(jnp.abs(inner) >= 1, lambda i: (i < 0)*jnp.pi, jnp.arccos, jnp.clip(inner, -1, 1))

        def squared_dist(self, p, q):
            inner = (p * q).sum()
            # return jax.lax.cond(inner > 1-1e-6, lambda _: jnp.sum((q-p)**2), lambda i: jnp.arccos(i)**2, jnp.clip(inner, None, 1-1e-6))
            return jax.lax.cond(inner > 1 - 1e-6,
                                lambda i: -2*(i-1) + (i-1)**2/3 - 4*(i-1)**3/45,
                                lambda i: jnp.arccos(i) ** 2,
                                inner)

        def jacobiField(self, p, q, t, X):
            phi = self.dist(p, q)
            v = self.log(p, q)
            gamTS = self.exp(p, t*v)

            v = v / phi
            Xtan_norm = self.inner(p, X, v)
            Xtan = Xtan_norm * v
            Xorth = X - Xtan

            # tangential component of J: (1-t) * transp(p, gamTS, Xtan)
            Jtan = Xtan_norm / phi * self.log(gamTS, q)
            return gamTS, (jnp.sin((1 - t) * phi) / jnp.sin(phi)) * Xorth + Jtan

        def _adjJacobi(self, p, q, t, w):
            # alternative version to adjJacobi relying on automatic differentiation
            return self._M.proj(p, _eval_adjJacobi_embed(self, p, q, t, w))

        def adjJacobi(self, p, q, t, w):
            """Evaluate an adjoint Jacobi field.

            The decomposition of the curvature operator and the fact that only two of its eigenvectors are necessary is
            used in the algorithm.

            :param p: element of the hyperboloid
            :param q: element of the hyperboloid
            :param t: scalar in [0,1]
            :param w: tangent vector at gamma(t;p,q)
            :returns: tangent vector at p
            """
            dist = self.dist(p, q)

            def _eval(dW):
                d, W = dW
                # all computations can be done at p, so only w has to be parallel translated
                W = self.transp(self.geopoint(p, q, t), p, W)

                # first eigenvector is normalized tangent of the geodesic -> corresponding eigenvalue is 0
                T = self.log(p, q) / jnp.clip(d, 1e-6)  # clipping only for NAN debugging

                # second eigenvector is Gram Schmidt orthonormalization of W against T -> corresponding eigenvalue is 1
                b1 = self.inner(p, W, T)
                U = W - b1 * T
                # Check whether W is (numerically) parallel to T;
                # then, the adoint Jacobi field is only a scaling of the parallel transported tangent.
                U = jax.lax.cond(jnp.linalg.norm(U) > 1e-3,
                                 lambda v: v / jnp.clip(self.norm(p, v), 1e-3),  # clipping only for NAN debugging
                                 lambda v: jnp.zeros_like(v), U)

                b2 = self.inner(p, W, U)

                a1 = 1 - t  # corresponds to the eigenvalue 0
                a2 = jnp.sin((1 - t) * d) / jnp.sin(d)  # corresponds to the eigenvalue 1

                return a1 * b1 * T + a2 * b2 * U

            return jax.lax.cond(dist > 1e-6, _eval, lambda args: 1/jnp.clip(1-t, 1e-3) * args[-1], (dist, w))
