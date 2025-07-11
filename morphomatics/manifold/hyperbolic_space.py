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
from morphomatics.manifold.metric import _eval_adjJacobi_embed


class HyperbolicSpace(Manifold):
    """n-dimensional hyperbolic space represented by the hyperboloid model. Elements are represented as (row) vectors of
    length n + 1.
    """

    def __init__(self, point_shape=(3,), structure='Canonical'):
        name = 'Points in ' +\
               'x'.join(map(str, point_shape)) + '-dim. space ' + 'for which the Minkowski quadratic form is -1.'
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
        structure = HyperbolicSpace.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    @staticmethod
    def minkowski_inner(x, y):
        """Minkowski semi-Riemannian metric on R^(n+1). The hyperboloid model of n-dimensional
        hyperbolic space consists of all points x with minkowski_inner(x, x) = -1. The tangent space at x consists of
        all vectors v in R^(n+1) with minkowski_inner(x, v) = 0."""
        return -x[-1] * y[-1] + (x[:-1] * y[:-1]).sum()

    @staticmethod
    def project_to_manifold(x):
        """Projection onto the hyperboloid."""
        return x.at[-1].set(jnp.sqrt(1 + jnp.sum(x[:-1] ** 2)))

    @staticmethod
    def regularize(p):
        """Regularize/project points that are slightly off the hyperboloid to avoid numerical problems. Only apply when
        minkowski_inner(p, p) < 0."""
        return p / jnp.sqrt(jnp.abs(HyperbolicSpace.minkowski_inner(p, p)) + np.finfo(np.float64).eps)

    def rand(self, key: jax.Array):
        p = jax.random.normal(key, self.point_shape)
        return self.project_to_manifold(p)

    def randvec(self, p, key: jax.Array):
        H = jax.random.normal(key, self.point_shape)
        return H + HyperbolicSpace.minkowski_inner(p, H) * p

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def pole(self):
        o = jnp.zeros(self.point_shape)
        o = o.at[-1].set(1)
        return o

    def proj(self, p, H):
        return H + HyperbolicSpace.minkowski_inner(p, H) * p

    class CanonicalStructure(Metric):
        """
        The Riemannian metric used is the induced metric from the Minkowski sub-Riemannian metric on the embedding space
        R^n+1.
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
            return jnp.sqrt(self._M.dim)

        def inner(self, p, X, Y):
            return HyperbolicSpace.minkowski_inner(X, Y)

        @property
        def metric_matrix(self):
            """Matrix representation of the Minkowski metric"""
            G = jnp.eye(self._M.dim + 1)
            G = G.at[-1, -1].set(-1)
            return G

        def norm(self, p, X):
            # inner can be smaller than 0 due to cancelling of digits when 2 similar numbers are subtracted
            return jnp.sqrt(jax.nn.relu(self.inner(p, X, X)))

        def dist(self, p, q):
            return jnp.sqrt(self.squared_dist(p, q))

        def squared_dist(self, p, q):
            # in theory minkowski_inner(p, q) < -1 always holds, but for numerical reasons we clip
            mink_inner_neg = jnp.clip(-HyperbolicSpace.minkowski_inner(p, q), a_min=1)

            def trunc_dist_sq(x):
                # 4th order approximation of arccosh**2 around 1
                return 2*(x-1) - 1/3*(x-1)**2 + 4/45*(x-1)**3

            d2 = jax.lax.cond(mink_inner_neg > 1+1e-6, lambda i: jnp.acosh(i)**2, trunc_dist_sq, mink_inner_neg)

            return jax.nn.relu(d2)

        def egrad2rgrad(self, p, H):
            H = H.at[-1].set(-H[-1])
            return self._M.proj(p, H)

        def exp(self, p, X):
            # numerical safeguard
            X = self._M.proj(p, X)

            def full_exp(n2):
                n = jnp.sqrt(n2)
                return jnp.cosh(n) * p + jnp.sinh(n) * X/n
            
            def trunc_exp(n2):
                # 6th order approximation of cosh(n)*p + sinh(n)*X/n around 0
                return p+X + 1/6 * n2 * (3*p+X) + 1/120 * n2**2 * (5*p+X)

            sqnorm_X = self.inner(p, X, X)
            sqnorm_X = jnp.clip(sqnorm_X, a_min=0.)
            
            p = jax.lax.cond(sqnorm_X < 1e-6, trunc_exp, full_exp, sqnorm_X)
            
            return HyperbolicSpace.project_to_manifold(p)

        def retr(self, p, X):
            return self.exp(p, X)

        def log(self, p, q):
            sqd = self.squared_dist(p, q)
            
            def full_log(d2):
                # For the formula see Eq. (13) in Pennec, "Hessian of the Riemannian squared distance", HAL INRIA, 2017.
                d = jnp.sqrt(d2)
                return d/jnp.sinh(d) * (q - jnp.cosh(d) * p)
            
            def trunc_log(d2):
                # see also Pennec, "Hessian of the Riemannian squared distance", HAL INRIA, 2017.
                return (1 - d2/6) * (q - (1 + d2/2 + d2**2/24) * p)
            
            v = jax.lax.cond(sqd < 1e-6, trunc_log, full_log, sqd)
            
            return self._M.proj(p, v)

        def geopoint(self, p, q, t):
            """Evaluate gam(t;p,q) = exp_p(t*log_p(q))."""
            dist = self.dist(p, q)

            def full_geopoint(d):
                r = q / jnp.sinh(d) - 1/jnp.tanh(d) * p  # log_p(q) / d
                return jnp.cosh(t*d) * p + jnp.sinh(t*d) * r

            return jax.lax.cond(dist > 1e-6, full_geopoint, lambda _: (1-t)*p + t*q, dist)

        def flat(self, p, X):
            """Lower vector with the metric"""
            dX = X.at[-1].set(-X[-1])
            return dX

        def sharp(self, p, dX):
            """Raise covector with the metric"""
            X = dX.at[-1].set(-dX[-1])
            return X

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return -1 * (HyperbolicSpace.minkowski_inner(Z, Y) * X - HyperbolicSpace.minkowski_inner(Z, X) * Y)

        def transp(self, p, q, X):
            sqd = self.squared_dist(p, q)

            def do_transp(X):
                sum_l = self.log(p, q) + self.log(q, p)
                return X - self.inner(p, self.log(p, q), X)/sqd * sum_l

            return jax.lax.cond(sqd < 1e-6, lambda X: X, do_transp, X)

        def jacobiField(self, p, q, t, X):
            return NotImplementedError('This function has not been implemented yet.')

        def _adjJacobi(self, p, q, t, w):
            # alternative version to adjJacobi relying on automatic differentiation
            return self._M.proj(p, _eval_adjJacobi_embed(self, p, q, t, w))

        def adjJacobi(self, p, q, t, w):
            """Evaluate an adjoint jacobi field.

            The decomposition of the curvature operator and the fact that only two of its eigenvectors are necessary is
            used in the algorithm.

            :param p: element of the hyperboloid
            :param q: element of the hyperboloid
            :param t: scalar in [0,1]
            :param w: tangent vector at gamma(t;p,q)
            :returns V: tangent vector at p
            """
            dist = self.dist(p, q)

            def _eval(dW):
                d, W = dW
                # all computations can be done at p, so only w has to be parallel translated
                W = self.transp(self.geopoint(p, q, t), p, W)

                # first eigenvector is normalized tangent of the geodesic -> corresponding eigenvalue is 0
                T = self.log(p, q) / d

                # second eigenvector is Gram Schmidt orthonormalization of W against T -> corresponding eigenvalue is -1
                b1 = self.inner(p, W, T)
                U = W - b1 * T
                # Check whether W is (numerically) parallel to T;
                # then, the adoint Jacobi field is only a scaling of the parallel transported tangent.
                U = jax.lax.cond(jnp.linalg.norm(U) > 1e-6,
                                 lambda v: v / self.norm(p, v),
                                 lambda v: jnp.zeros_like(v), U)

                b2 = self.inner(p, W, U)

                a1 = 1-t  # corresponds to the eigenvalue 0
                a2 = jnp.sinh((1-t)*d) / jnp.sinh(d)  # corresponds to the eigenvalue -1

                return a1 * b1 * T + a2 * b2 * U

            return jax.lax.cond(dist > 1e-6, _eval, lambda args: 1/(1-t) * args[-1], (dist, w))
