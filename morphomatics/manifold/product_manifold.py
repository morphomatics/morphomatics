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

from typing import Sequence
import functools

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Connection, Metric, LieGroup


class ProductManifold(Manifold):
    """ Product manifold """

    def __init__(self, mfds: Sequence[Manifold], weights: jnp.array = None, structure: str = 'Product'):
        assert weights is None or len(weights) == len(mfds)

        point_shape = (np.sum([np.prod(m.point_shape) for m in mfds]),)
        name = f'Product ' + functools.reduce(lambda a, b: f'{a}x{b}', [str(m.__class__.__name__) for m in mfds]) + '.'
        dimension = np.sum([m.dim for m in mfds])
        super().__init__(name, dimension, point_shape)
        self._manifolds = mfds
        self._weights = weights
        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children + (self.manifolds,) + (self.weights,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, mfds, w = children
        obj = cls(mfds, w, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    @property
    def manifolds(self) -> Sequence[Manifold]:
        """Return the manifolds this product is composed of. """
        return self._manifolds

    @property
    def weights(self) -> jnp.array:
        """Return weights of product metric. """
        return self._weights

    def initProductStructure(self):
        """
        Set up the product manifold with a product structure.
        """

        if np.all([m.metric is not None for m in self.manifolds]):
            self._metric = self._connec = ProductManifold.ProductMetric(self)
        elif np.all([m.connec is not None for m in self.manifolds]):
            self._connec = ProductManifold.ProductConnection(self)

        if np.all([m.group is not None for m in self.manifolds]):
            self._group = ProductManifold.ProductGroup(self)

    def disentangle(self, c):
        """
        :arg c: vectorized fundamental coords. (tangent vectors)
        :returns: de-vectorized tuple of rotations and stretches (skew-sym. and sym. matrices)
        """
        p = []
        o = 0
        for mfd in self._manifolds:
            l = np.prod(mfd.point_shape)
            p.append(c[o:o+l].reshape(mfd.point_shape))
            o += l
        return p

    def entangle(self, p: Sequence[jnp.array]) -> jnp.array:
        """
        Inverse of #disentangle().
        :arg p: list of elements in #manifolds
        :returns: concatenated and vectorized version
        """
        return jnp.concatenate([c.ravel() for c in p])

    def rand(self, key: jax.Array) -> jnp.array:
        """ Random element of the product manifold
        :param key: a PRNG key
        """
        subkeys = jax.random.split(key, len(self.manifolds))
        p = []
        for mfd, k in zip(self.manifolds, subkeys):
            p.append(mfd.rand(k))
        return self.entangle(p)

    def randvec(self, p: jnp.array, key: jax.Array) -> jnp.array:
        """Random vector in the tangent space of the point pu

        :param p: element of M^k
        :param key: a PRNG key
        :return: random tangent vector at p
        """
        subkeys = jax.random.split(key, len(self.manifolds))
        v = []
        for mfd, x, k in zip(self.manifolds, self.disentangle(p), subkeys):
            v.append(mfd.randvec(x, k))
        return self.entangle(v)

    def zerovec(self) -> jnp.array:
        """Zero vector in any tangent space
        """
        return jnp.zeros(self.point_shape)

    def proj(self, p, z):
        """Project ambient vector onto the product manifold

        :param p: element of M^k
        :param z: ambient vector
        :return: projection of z to the tangent space at p
        """
        x = []
        for mfd, q, y in zip(self.manifolds, self.disentangle(p), self.disentangle(z)):
            x.append(mfd.proj(q, y))
        return self.entangle(x)

    class ProductConnection(Connection):
        """ Product connection """

        def __init__(self, M):
            self._M = M

        def __str__(self) -> str:
            return "Product connection"

        def exp(self, p, X):
            """Exponential map of the connection at p applied to the tangent vector X.
            """
            x = []
            for mfd, q, v in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(X)):
                x.append(mfd.connec.exp(q, v))
            return self._M.entangle(x)

        def retr(self, p, X):
            """Computes a retraction mapping a vector X in the tangent space at
            p to the manifold.
            """
            x = []
            for mfd, q, v in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(X)):
                x.append(mfd.connec.retr(q, v))
            return self._M.entangle(x)

        def log(self, p, q):
            """Logarithmic map of the connection at p applied to q.
            """
            v = []
            for mfd, x, y in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q)):
                v.append(mfd.connec.log(x, y))
            return self._M.entangle(v)

        def transp(self, p, q, X):
            """Computes a vector transport which transports a vector X in the
            tangent space at p to the tangent space at q.
            """
            x = []
            for mfd, p_, q_, X_ in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q), self._M.disentangle(X)):
                x.append(mfd.connec.transp(p_, q_, X_))
            return self._M.entangle(x)

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            x = []
            for mfd, p_, X_, Y_, Z_ in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(X),
                                       self._M.disentangle(Y), self._M.disentangle(Z)):
                x.append(mfd.connec.curvature_tensor(p_, X_, Y_, Z_))
            return self._M.entangle(x)

        def jacobiField(self, p, q, t, X):
            """
            Evaluates a Jacobi field (with boundary conditions gam'(0) = X, gam'(1) = 0) along the geodesic gam from p to q.
            :param p: element of the Riemannian manifold
            :param q: element of the Riemannian manifold
            :param t: scalar in [0,1]
            :param X: tangent vector at p
            :return: [b, J] with J and b being the Jacobi field at t and the corresponding basepoint
            """
            x = []
            for mfd, p_, q_, X_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q), self._M.disentangle(X)):
                x.append(mfd.connec.jacobiField(p_, q_, t, X_))
            return self._M.entangle(x)

    class ProductMetric(ProductConnection, Metric):
        """ Product manifold """

        def __str__(self) -> str:
            return "Product metric"

        @property
        def typicaldist(self):
            d2 = np.array([m.metric.typicaldist**2 for m in self._M.manifolds])
            if self._M._weights is not None:
                d2 *= self._M.weights
            return np.sum(d2)**.5

        def dist(self, p, q):
            """Returns the geodesic distance between two points p and q on the
            product manifold."""
            return self.squared_dist(p, q) ** .5

        def squared_dist(self, p, q):
            x = self._M.disentangle(p)
            y = self._M.disentangle(q)
            d2 = jnp.asarray([m.metric.squared_dist(a, b) for (m, a, b) in zip(self._M.manifolds, x, y)])
            if self._M.weights is not None:
                d2 = d2 * self._M.weights
            return jnp.sum(d2)

        def inner(self, p, X, Y):
            """Returns the inner product (i.e., the Riemannian metric) between two
            tangent vectors X and Y from the tangent space at p.
            """
            q = self._M.disentangle(p)
            v = self._M.disentangle(X)
            w = self._M.disentangle(Y)
            i = jnp.asarray([m.metric.inner(q_, v_, w_) for (m, q_, v_, w_) in zip(self._M.manifolds, q, v, w)])
            if self._M.weights is not None:
                i = i * self._M.weights
            return np.sum(i)

        def egrad2rgrad(self, p, X):
            """Maps the Euclidean gradient X in the ambient space on the tangent
            space of the manifold at p.
            """
            x = []
            for mfd, p_, X_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(X)):
                x.append(mfd.metric.egrad2rgrad(p_, X_))
            if self._M.weights is not None:
                x = [x_/w for x_, w in zip(x, self._M.weights)]
            return self._M.entangle(x)

        def flat(self, p, X):
            """Lower vector X at p with the metric"""
            v = []
            for mfd, p_, X_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(X)):
                v.append(mfd.metric.flat(p_, X_))
            if self._M.weights is not None:
                x = [v_*w for v_, w in zip(v, self._M.weights)]
            return self._M.entangle(v)

        def sharp(self, p, dX):
            """Raise covector dX at p with the metric"""
            v = []
            for mfd, p_, dX_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(dX)):
                v.append(mfd.metric.sharp(p_, dX_))
            if self._M.weights is not None:
                x = [v_/w for v_, w in zip(v, self._M.weights)]
            return self._M.entangle(v)

        def adjJacobi(self, p, q, t, X):
            """Evaluates an adjoint Jacobi field for the geodesic gam from p to q at p.
            :param p: element of the Riemannian manifold
            :param q: element of the Riemannian manifold
            :param t: scalar in [0,1]
            :param X: tangent vector at gam(t)
            :return: tangent vector at p
            """
            if self._M.weights is not None:
                raise NotImplementedError('This function has not been implemented yet for non-trivial metric weights.')

            v = []
            for mfd, p_, q_, X_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q), self._M.disentangle(X)):
                v.append(mfd.metric.adjJacobi(p_, q_, t, X_))
            return self._M.entangle(v)

        def projToGeodesic(self, p, q, s, max_iter=10):
            v = []
            for mfd, p_, q_, s_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q), self._M.disentangle(s)):
                v.append(mfd.metric.projToGeodesic(p_, q_, s_, max_iter))
            return self._M.entangle(v)


    class ProductGroup(LieGroup):
        """ Product group """

        def __init__(self, M):
            self._M = M

        def __str__(self) -> str:
            return "Product group"

        @property
        def identity(self):
            """Returns the identity element e of the Lie group."""
            return self._M.entangle([m.group.identity() for m in self._M.manifolds])

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return self._M.entangle([m.group.coords(X_) for m, X_ in zip(self._M.manifolds, self._M.disentangle(X))])

        def coords_inv(self, X):
            """Coordinate map for the tangent space at the identity."""
            return self._M.entangle([m.group.coords_inverse(X_) for m, X_ in zip(self._M.manifolds, self._M.disentangle(X))])

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return self._M.entangle([m.group.bracket(X_, Y_) for m, X_, Y_ in
                                     zip(self._M.manifolds, self._M.disentangle(X), self._M.disentangle(Y))])

        def lefttrans(self, g, f):
            """Left translation of g by f.
            """
            return self._M.entangle([m.group.lefttrans(g_, f_) for m, g_, f_ in
                                     zip(self._M.manifolds, self._M.disentangle(g), self._M.disentangle(f))])

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return self._M.entangle([m.group.righttrans(g_, f_) for m, g_, f_ in
                                     zip(self._M.manifolds, self._M.disentangle(g), self._M.disentangle(f))])

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return self._M.entangle([m.group.righttrans(g_) for m, g_ in
                                     zip(self._M.manifolds, self._M.disentangle(g))])
        def exp(self, X):
            """Computes the Lie-theoretic exponential map of a tangent vector X at e.
            """
            return self._M.entangle([m.group.exp(X_) for m, X_ in
                                     zip(self._M.manifolds, self._M.disentangle(X))])

        retr = exp

        def log(self, g):
            """Computes the Lie-theoretic logarithm of g. This is the inverse of `exp`.
            """
            return self._M.entangle([m.group.log(g_) for m, g_ in
                                     zip(self._M.manifolds, self._M.disentangle(g))])

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            return self._M.entangle([m.group.adjrep(g_, X_) for m, g_, X_ in
                                     zip(self._M.manifolds, self._M.disentangle(g), self._M.disentangle(X))])

        def jacobiField(self, p, q, t, X):
            """
            Evaluates a Jacobi field (with boundary conditions gam'(0) = X, gam'(1) = 0) along the geodesic gam of the
            CCS connection from p to q.
            :param p: element of the Lie group
            :param q: element of the Lie group
            :param t: scalar in [0,1]
            :param X: tangent vector at p
            :return: [b, J] with J and b being the Jacobi field at t and the corresponding basepoint
            """
            x = []
            for mfd, p_, q_, X_, in zip(self._M.manifolds, self._M.disentangle(p), self._M.disentangle(q), self._M.disentangle(X)):
                x.append(mfd.group.jacobiField(p_, q_, t, X_))
            return self._M.entangle(x)
