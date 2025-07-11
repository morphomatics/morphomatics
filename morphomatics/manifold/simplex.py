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


class Simplex(Manifold):
    """
    The d-dimensional open simplex, that is the set of all vectors x = (x_0, ..., x_d) such that
    x_i > 0 for all i and sum(x) = 1.

    Meassurements in this space are called compositional data and include probabilities, proportions, and percentages.
    """

    def __init__(self, d, structure='Aitchison'):
        if d <= 0:
            raise RuntimeError("d must be an integer no less than 1.")

        name = '{d}-simplex'.format(d=d)

        self._d = d

        dimension = int(d)
        point_shape = (dimension+1,)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children, aux+(self._d,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *aux_data, d = aux_data
        obj = cls(d, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def rand(self, key: jax.Array):
        d = self.dim
        x = jnp.zeros(d+2).at[-1].set(1)
        x = x.at[1:-1].set(jax.random.uniform(key, (d,)).sort())
        return x[1:] - x[:-1]

    def randvec(self, X, key: jax.Array):
        x = self.rand(key)
        return x - 1/(self.dim+1)

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, S, H):
        """Orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector onto the tangent space at S"""
        return H - jnp.mean(H)

    def initAitchisonStructure(self):
        """
        Instantiate Aitchison's vector space structure.
        """
        structure = Simplex.AitchisonStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    class AitchisonStructure(Metric, LieGroup):
        """
        Aitchison's vector space structure on the d-dimensional open simplex.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return f"Aitchison geometry on {self._M.dim}-simplex"

        @property
        def typicaldist(self):
            return jnp.sqrt(self._M.dim)

        def inner(self, p, X, Y):
            return jnp.sum(X*Y)

        def egrad2rgrad(self, p, X):
            _, vjp = jax.vjp(jax.nn.softmax, clr(p))
            return vjp(self._M.proj(p, X))[0]

        def retr(self, p, X):
            return self.exp(p, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # cond: group or Riemannian exp
            X = argv[0] if len(argv) == 1 else (argv[-1] + clr(argv[0]))
            return jax.nn.softmax(X)

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            v = clr(argv[-1])
            return v if len(argv) == 1 else v - clr(argv[0])

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return jnp.zeros(self._M.point_shape)

        def transp(self, p, q, X):
            """Computes a vector transport which transports a vector X in the
            tangent space at p to the tangent space at q.
            """
            return X

        def dist(self, p, q):
            """Aitchison's distance"""
            return jnp.sqrt(self.squared_dist(p, q))

        def squared_dist(self, p, q):
            """Squared log-Euclidean distance function in Sym+(d)"""
            # d = self.log(p, q)
            # return self.inner(p, d, d)
            v = clr(self.lefttrans(p, self.inverse(q)))
            return jnp.sum(v*v)

        def flat(self, p, X):
            """Lower vector X at p"""
            return X

        def sharp(self, p, dX):
            """Raise covector dX at p"""
            return dX

        def jacobiField(self, p, q, t, X):
            o = self.geopoint(p, q, t)
            return o, (1 - t) * self.transp(p, o, X)

        def adjJacobi(self, p, q, t, X):
            o = self.geopoint(p, q, t)
            return (1 - t) * self.transp(o, p, X)

        @property
        def identity(self):
            return jnp.full(self._M.point_shape, 1/(self._M.dim+1))

        def lefttrans(self, g, f):
            """(Commutative) translation of g by f, i.e. Aitchison's pertubation operation."""
            p = g*f
            return p/jnp.sum(p)

        righttrans = lefttrans

        def inverse(self, p):
            """Inverse map of the Lie group, i.e. Aitchison's power of minus one.
            """
            q = 1/p
            return q / jnp.sum(q)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return X[:-1]

        def coords_inv(self, c):
            """Inverse of coords"""
            return jnp.append(c, -jnp.sum(c))

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.zeros(self._M.point_shape)

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')

def clr(X):
    """Center log-ratio transform: Map the open d-simplex to the hyperplane of zero-mean vectors in R^(d+1).
    This is an isometry w.r.t. the Aitchison metric on the simplex.

    The inverse of this transform is given by the softmax function.
    """
    logX = jnp.log(X)
    return logX - jnp.mean(logX)
