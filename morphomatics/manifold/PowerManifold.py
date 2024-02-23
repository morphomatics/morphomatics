################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Metric, LieGroup


class PowerManifold(Manifold):
    """ Product manifold M^k of several copies of a single (atom) manifold M """

    def __init__(self, M: Manifold, k: int, structure: str = 'Product'):
        point_shape = tuple([k, *M.point_shape])
        name = f'Product of {k} copies of ' + M.__str__() + '.'
        dimension = M.dim * k
        super().__init__(name, dimension, point_shape)
        self._atom_manifold = M
        self._k = k
        if structure:
            getattr(self, f'init{structure}Structure')()

    @property
    def atom_manifold(self) -> Manifold:
        """Return the atom manifold, i.e., the manifold of which we take the power """
        return self._atom_manifold

    @property
    def k(self) -> int:
        """Return the atom manifold, i.e., the manifold of which we take the power """
        return self._k

    def ith_component(self, x: jnp.array, i: int) -> jnp.array:
        """Projection to the i-th element for both points and tangent vectors of M^k"""
        return x[i]

    def initProductStructure(self):
        """
        Instantiate the power manifold with product structure.
        """
        structure = PowerManifold.ProductStructure(self)
        self._metric = structure if self.atom_manifold.metric is not None else None
        self._connec = structure if self.atom_manifold.connec is not None else None
        self._group = structure if self.atom_manifold.group is not None else None

    def rand(self, key: jax.Array) -> jnp.array:
        """ Random element of the power manifold
        :param key: a PRNG key
        """
        subkeys = jax.random.split(key, self.k)
        return jax.vmap(self.atom_manifold.rand)(subkeys)

    def randvec(self, p: jnp.array, key: jax.Array) -> jnp.array:
        """Random vector in the tangent space of the point pu

        :param p: element of M^k
        :param key: a PRNG key
        :return: random tangent vector at p
        """
        subkeys = jax.random.split(key, self.k)
        return jax.vmap(self.atom_manifold.randvec)(p, subkeys)

    def zerovec(self) -> jnp.array:
        """Zero vector in any tangen space
        """
        return jnp.zeros(self.point_shape)

    def proj(self, p, z):
        """Project ambient vector onto the power manifold

        :param p: element of M^k
        :param z: ambient vector
        :return: projection of z to the tangent space at p
        """

        return jax.vmap(self.atom_manifold.proj)(p, z)

    class ProductStructure(Metric, LieGroup):
        """ Product structure, i.e., product metric, product connection, and, if applicable, product Lie group structure
        on M^k
        """

        def __init__(self, PM: Manifold):
            self._PM = PM
            self._M = PM.atom_manifold

        def __str__(self) -> str:
            return "Product structure"

        #### metric interface ####

        @property
        def typicaldist(self) -> float:
            """Typical distance in the product manifold"""
            d = jax.vmap(lambda _: self._M.metric.typicaldist**2)(jnp.arange(self._PM.k))

            return jnp.sqrt(jnp.sum(d))

        def inner(self, p: jnp.array, v: jnp.array, w: jnp.array) -> float:
            """Product metric

            :param p: element of M^k
            :param v: tangent vector at p
            :param w: tangent vector at p
            :return: inner product of v and w
            """

            return jnp.sum(jax.vmap(self._M.metric.inner)(p, v, w))

        def dist(self, p: jnp.array, q: jnp.array) -> float:
            """Distance function of the product metric

            :param p: element of M^k
            :param q: element of M^k
            :return: distance between p and q
            """

            return jnp.sqrt(self.squared_dist(p, q))

        def squared_dist(self, p: jnp.array, q: jnp.array) -> float:
            """Squared distance function of the product metric

            :param p: element of M^k
            :param q: element of M^k
            :return: squared distance between p and q
            """

            return jnp.sum(jax.vmap(lambda _p, _q: self._M.metric.squared_dist(_p, _q))(p, q))

        def flat(self, p: jnp.array, v: jnp.array) -> jnp.array:
            """Lower vector v at p with the metric

            :param p: element of M^k
            :param v: tangent vector at p
            :return: covector at p
            """

            return jax.vmap(self._M.metric.flat)(p, v)

        def sharp(self, p: jnp.array, dv: jnp.array) -> jnp.array:
            """Raise covector dv at p with the metric

            :param p: element of M^k
            :param dv: covector at p
            :return: tangent vector at p
            """

            return jax.vmap(self._M.metric.sharp)(p, dv)

        def adjJacobi(self, p: jnp.array, q: jnp.array, t: float, X: jnp.array) -> jnp.array:
            """
            Evaluates an adjoint Jacobi field along the geodesic gam from p to q. X is a vector at gam(t)

            :param p: element of M^k
            :param q: element of M^k
            :param t: scalar in [0,1]
            :param X: tangent vector at gam(t)
            :return: tangent vector at p
            """
            return jax.vmap(self._M.metric.adjJacobi)(p, q, t, X)

        def egrad2rgrad(self, p, z):
            """Transform the Euclidean gradient of a function into the corresponding Riemannian gradient, i.e.,
            directions pointing away from the manifold are removed

            :param p: element of M^k
            :param z: Euclidean gradient at p
            :return: Riemannian gradient at p
            """
            return jax.vmap(self._M.metric.egrad2rgrad)(p, z)

        def ehess2rhess(self, pu, G, H, vw):
            """Converts the Euclidean gradient G and Hessian H of a function at
            a point pv along a tangent vector uw to the Riemannian Hessian
            along X on the manifold.
            """
            return jax.vmap(self._M.metric.ehess2rhess)(pu, G, H, vw)

        #### connection interface ####

        def exp(self, p: jnp.array, v: jnp.array) -> jnp.array:
            """Riemannian exponential

            :param p: element of M^k
            :param v: tangent vector at p
            :return: point at time 1 of the geodesic that starts at p with initial velocity v
            """
            return jax.vmap(self._M.connec.exp)(p, v)

        retr = exp

        def log(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Riemannian logarithm

            :param p: element of M^k
            :param q: element of M^k
            :return: vector at p with exp(p, v) = q
            """
            return jax.vmap(self._M.connec.log)(p, q)

        def geopoint(self, p: jnp.array, q: jnp.array, t: float) -> jnp.array:
            """Geodesic map

            :param p: element of M^k
            :param q: element of M^k
            :param t: scalar between 0 and 1
            :return: element of M^k on that is reached in the geodesic between p an q at time t
            """
            return jax.vmap(self._M.connec.geopoint, (0, 0, None))(p, q, t)

        def transp(self, p: jnp.array, q: jnp.array, v: jnp.array) -> jnp.array:
            """Parallel transport map

            :param p: element of M^k
            :param q: element of M^k
            :param v: tangent vector at p
            :return: tangent vector at q that is the parallel transport of v along the geodesic from p to q
            """
            return jax.vmap(self._M.connec.transp)(p, q, v)

        def pairmean(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Pair-wise mean

            :param p: element of M^k
            :param q: element of M^k
            :return: mean of p and q
            """
            return jax.vmap(self._M.connec.pairmean)(p, q)

        def curvature_tensor(self, p: jnp.array, v: jnp.array, w: jnp.array, x: jnp.array) -> jnp.array:
            """Curvature tensor

            :param p: element of M^k
            :param v: tangent vector at p
            :param w: tangent vector at p
            :param x: tangent vector at p
            :return: tangent vector at p that is the value R(v,w)x of the curvature tensor
            """
            return jax.vmap(self._M.connec.curvature_tensor)(p, v, w, x)

        def jacobiField(self, p: jnp.array, q: jnp.array, t: float, X: jnp.array) -> jnp.array:
            """
            Evaluates a Jacobi field (with boundary conditions gam'(0) = X, gam'(1) = 0) along the geodesic gam from p to
            q.

            :param p: element of M^k
            :param q: element of M^k
            :param t: scalar in [0,1]
            :param X: tangent vector at p
            :return: [gam(t), J], where J is the value of the Jacobi field (which is an element-wise Jacobi field) at gam(t)
            """
            return jax.vmap(self._M.connec.jacobiField)(p, q, t, X)

        #### group interface ####

        def identity(self) -> jnp.array:
            """Identity element"""

            return jax.vmap(self._M.group.identity)(jnp.arange(self._PM.k))

        def coords(self, v: jnp.array) -> jnp.array:
            """Coordinate map for the tangent space at the identity

            :param X: tangent vector at the identity
            :return: vector of coordinates
            """
            x = jax.vmap(self._M.group.identity)(v)
            return x.reshape(-1)

        def bracket(self, v: jnp.array, w: jnp.array) -> jnp.array:
            """Lie bracket in Lie algebra

            :param v: tangent vector at the identity
            :param w: tangent vector at the identity
            :return: tangent vector at the identity that is the Lie bracket of v and w
            """
            return jax.vmap(self._M.group.backet)(v, w)

        def lefttrans(self, g: jnp.array, f: jnp.array) -> jnp.array:
            """Left translation of g by f

            :param g: element of the Lie group M^k
            :param f: element of the Lie group M^k
            :return: left-translated element
            """
            return jax.vmap(self._M.lefttrans)(g, f)

        def righttrans(self, g: jnp.array, f: jnp.array) -> jnp.array:
            """Right translation of g by f

            :param g: element of the Lie group M^k
            :param f: element of the Lie group M^k
            :return: right-translated element
            """
            return jax.vmap(self._M.righttrans)(g, f)

        def inverse(self, g):
            """Inverse map of the Lie group

            :param g: element of the Lie group M^k
            :return: element of M^k that is inverse to g
            """
            return jax.vmap(self._M.inverse)(g)

        # the group exponential and logarithm are given by the connection group and logarithm (by using them with a
        # different number of arguments).

        def dleft(self, f, v):
            """Derivative of the left translation by f at the identity applied to the tangent vector v

            :param f: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: left-translated tangent vector represented at the identity
            """
            return jax.vmap(self._M.dleft)(f, v)

        def dright(self, f, v):
            """Derivative of the right translation by f at e applied to the tangent vector v

            :param f: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: right-translated tangent vector represented at the identity
            """
            return jax.vmap(self._M.dright)(f, v)

        def dleft_inv(self, f, v):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector v

            :param f: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: translated vector represented at the identity
            """
            return jax.vmap(self._M.dleft_inv)(f, v)

        def dright_inv(self, f, v):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector v

            :param f: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: translated vector represented at the identity
            """
            return jax.vmap(self._M.dright_inv)(f, v)

        def adjrep(self, g, v):
            """Adjoint representation of g applied to the tangent vector v at the identity

            :param g: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: tangent vector at the identity
            """
            return jax.vmap(self._M.adjrep)(g, v)
