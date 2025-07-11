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

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Metric, LieGroup


class PowerManifold(Manifold):
    """ Product manifold M^k consisting of k copies of a single (atom) manifold M """

    def __init__(self, M: Manifold, k: int, metric_weights: jnp.array = None, structure: str = 'Product'):
        assert metric_weights is None or len(metric_weights) == k

        point_shape = tuple([k, *M.point_shape])
        name = f'Product of {k} copies of ' + M.__str__() + '.'
        dimension = M.dim * k
        super().__init__(name, dimension, point_shape)
        self._atom_manifold = M
        self._k = k
        self.metric_weights = metric_weights
        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children + (self.atom_manifold, self.metric_weights), aux + (self.k,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, M, w = children
        *aux_data, k = aux_data
        obj = cls(M, k, w, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    @property
    def atom_manifold(self) -> Manifold:
        """Return the atom manifold M"""
        return self._atom_manifold

    @property
    def k(self) -> int:
        """Return the power k"""
        return self._k

    def ith_component(self, x: jnp.array, i: int) -> jnp.array:
        """Projection to the i-th element for both points and tangent vectors of M^k"""
        return x[i]

    def initProductStructure(self):
        """
        Instantiate the power manifold with product structure.
        """
        MetricStructure = PowerManifold.ProductMetric(self)
        GroupStructure = PowerManifold.ProductGroup(self)

        self._metric = MetricStructure if self.atom_manifold.metric is not None else None
        self._connec = MetricStructure if self.atom_manifold.connec is not None else None
        self._group = GroupStructure if self.atom_manifold.group is not None else None

    def rand(self, key: jax.Array) -> jnp.array:
        """ Random element of the power manifold
        :param key: a PRNG key
        """
        subkeys = jax.random.split(key, self.k)
        return jax.vmap(self.atom_manifold.rand)(subkeys)

    def randvec(self, p: jnp.array, key: jax.Array) -> jnp.array:
        """Random vector in the tangent space of the point p

        :param p: element of M^k
        :param key: a PRNG key
        :return: random tangent vector at p
        """
        subkeys = jax.random.split(key, self.k)
        return jax.vmap(self.atom_manifold.randvec)(p, subkeys)

    def zerovec(self) -> jnp.array:
        """Zero vector in any tangent space
        """
        return jnp.zeros(self.point_shape)

    def proj(self, p, z):
        """Project ambient vector onto the power manifold

        :param p: element of M^k
        :param z: ambient vector
        :return: projection of z to the tangent space at p
        """
        return jax.vmap(self.atom_manifold.proj)(p, z)

    class ProductMetric(Metric):
        """ Product metric with product Levi-Civita connection on M^k
        """
        def __init__(self, M):
            self._M: PowerManifold = M

        def __str__(self) -> str:
            return "Product metric"

        @property
        def atom_mfd(self):
            return self._M.atom_manifold

        @property
        def weights(self) -> jnp.array:
            return self._M.metric_weights

        #### metric interface ####

        @property
        def typicaldist(self) -> jnp.array:
            """Typical distance in the product manifold"""
            if self.weights is None:
                d = jax.vmap(lambda _: self.atom_mfd.metric.typicaldist ** 2)(jnp.arange(self._M.k))
            else:
                d = jax.vmap(lambda lam: lam * self.atom_mfd.metric.typicaldist ** 2)(self.weights)

            return jnp.sqrt(jnp.sum(d))

        def inner(self, p: jnp.array, v: jnp.array, w: jnp.array) -> jnp.array:
            """Product metric

            :param p: element of M^k
            :param v: tangent vector at p
            :param w: tangent vector at p
            :return: inner product of v and w
            """
            i = jax.vmap(self.atom_mfd.metric.inner)(p, v, w)
            if self.weights is not None:
                i = i * self.weights

            return jnp.sum(i)

        def dist(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Distance function of the product metric

            :param p: element of M^k
            :param q: element of M^k
            :return: distance between p and q
            """

            return jnp.sqrt(self.squared_dist(p, q))

        def squared_dist(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Squared distance function of the product metric

            :param p: element of M^k
            :param q: element of M^k
            :return: squared distance between p and q
            """
            d2 = jax.vmap(self.atom_mfd.metric.squared_dist)(p, q)
            if self.weights is not None:
                d2 = d2 * self.weights

            return jnp.sum(d2)

        def flat(self, p: jnp.array, v: jnp.array) -> jnp.array:
            """Lower vector v at p with the metric

            :param p: element of M^k
            :param v: tangent vector at p
            :return: covector at p
            """
            dv = jax.vmap(self.atom_mfd.metric.flat)(p, v)
            if self.weights is not None:
                dv = dv * self.weights.reshape((-1,) + (1,) * len(self.atom_mfd.point_shape))

            return dv

        def sharp(self, p: jnp.array, dv: jnp.array) -> jnp.array:
            """Raise covector dv at p with the metric

            :param p: element of M^k
            :param dv: covector at p
            :return: tangent vector at p
            """
            v = jax.vmap(self.atom_mfd.metric.flat)(p, dv)
            if self.weights is not None:
                dv = dv / self.weights.reshape((-1,) + (1,) * len(self.atom_mfd.point_shape))

            return dv

        def adjJacobi(self, p: jnp.array, q: jnp.array, t: float, v: jnp.array) -> jnp.array:
            """
            Evaluates an adjoint Jacobi field along the geodesic gam from p to q. X is a vector at gam(t)

            :param p: element of M^k
            :param q: element of M^k
            :param t: scalar in [0,1]
            :param v: tangent vector at gam(t)
            :return: tangent vector at p
            """
            if self.weights is None:
                return jax.vmap(self.atom_mfd.metric.adjJacobi, (0, 0, None, 0))(p, q, t, v)
            else:
                raise NotImplementedError('This function has not been implemented yet for non-trivial metric weights.')

        def egrad2rgrad(self, p: jnp.array, z: jnp.array) -> jnp.array:
            """Transform the Euclidean gradient of a function into the corresponding Riemannian gradient, i.e.,
            directions pointing away from the manifold are removed

            :param p: element of M^k
            :param z: Euclidean gradient at p
            :return: Riemannian gradient at p
            """
            g = jax.vmap(self.atom_mfd.metric.egrad2rgrad)(p, z)
            if self.weights is not None:
                g = g / self.weights.reshape((-1,) + (1,) * len(self.atom_mfd.point_shape))

            return g

        #### connection interface ####

        # Note that the Levi-Civita connection does not change under a constant re-scaling of the metric.
        # (This can, e.g., be deduced from the Koszul formula.) Therefore, all notions that only depend on the metric
        # implicitly through the connection are not influenced by metric weights.

        def exp(self, p: jnp.array, v: jnp.array) -> jnp.array:
            """Riemannian exponential

            :param p: element of M^k
            :param v: tangent vector at p
            :return: point at time 1 of the geodesic that starts at p with initial velocity v
            """
            return jax.vmap(self.atom_mfd.connec.exp)(p, v)

        retr = exp

        def log(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Riemannian logarithm

            :param p: element of M^k
            :param q: element of M^k
            :return: vector at p with exp(p, v) = q
            """
            return jax.vmap(self.atom_mfd.connec.log)(p, q)

        def geopoint(self, p: jnp.array, q: jnp.array, t: float) -> jnp.array:
            """Geodesic map

            :param p: element of M^k
            :param q: element of M^k
            :param t: scalar between 0 and 1
            :return: element of M^k on that is reached in the geodesic between p and q at time t
            """
            return jax.vmap(self.atom_mfd.connec.geopoint, (0, 0, None))(p, q, t)

        def transp(self, p: jnp.array, q: jnp.array, v: jnp.array) -> jnp.array:
            """Parallel transport map

            :param p: element of M^k
            :param q: element of M^k
            :param v: tangent vector at p
            :return: tangent vector at q that is the parallel transport of v along the geodesic from p to q
            """
            return jax.vmap(self.atom_mfd.connec.transp)(p, q, v)

        def pairmean(self, p: jnp.array, q: jnp.array) -> jnp.array:
            """Pair-wise mean

            :param p: element of M^k
            :param q: element of M^k
            :return: mean of p and q
            """
            return jax.vmap(self.atom_mfd.connec.pairmean)(p, q)

        def curvature_tensor(self, p: jnp.array, v: jnp.array, w: jnp.array, x: jnp.array) -> jnp.array:
            """Curvature tensor

            :param p: element of M^k
            :param v: tangent vector at p
            :param w: tangent vector at p
            :param x: tangent vector at p
            :return: tangent vector at p that is the value R(v,w)x of the curvature tensor
            """
            return jax.vmap(self.atom_mfd.connec.curvature_tensor)(p, v, w, x)

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
            return jax.vmap(self.atom_mfd.connec.jacobiField, (0, 0, None, 0))(p, q, t, X)

        def projToGeodesic(self, p, q, s, max_iter=10):
            return jax.vmap(self.atom_mfd.metric.projToGeodesic, (0, 0, 0, None))(p, q, s, max_iter)

    class ProductGroup(LieGroup):
        """ Product group structure with product CCS connection on M^k
        """

        def __init__(self, M):
            self._M: PowerManifold = M

        def __str__(self) -> str:
            return "Product group structure"

        @property
        def atom_mfd(self):
            return self._M.atom_manifold

        @property
        def identity(self) -> jnp.array:
            """Identity element"""

            return jax.vmap(lambda _: self.atom_mfd.group.identity)(jnp.arange(self._M.k))

        def coords(self, v: jnp.array) -> jnp.array:
            """Coordinate map for the tangent space at the identity

            :param v: tangent vector at the identity
            :return: vector of coordinates
            """
            c = jax.vmap(self.atom_mfd.group.coords)(v)
            return c.reshape(-1)

        def coords_inv(self, X):
            return jax.vmap(self.atom_mfd.group.coords_inv)(X.reshape(self._M.k, -1))

        def bracket(self, v: jnp.array, w: jnp.array) -> jnp.array:
            """Lie bracket in Lie algebra

            :param v: tangent vector at the identity
            :param w: tangent vector at the identity
            :return: tangent vector at the identity that is the Lie bracket of v and w
            """
            return jax.vmap(self.atom_mfd.group.bracket)(v, w)

        def lefttrans(self, g: jnp.array, f: jnp.array) -> jnp.array:
            """Left translation of g by f

            :param g: element of the Lie group M^k
            :param f: element of the Lie group M^k
            :return: left-translated element
            """
            return jax.vmap(self.atom_mfd.group.lefttrans)(g, f)

        def righttrans(self, g: jnp.array, f: jnp.array) -> jnp.array:
            """Right translation of g by f

            :param g: element of the Lie group M^k
            :param f: element of the Lie group M^k
            :return: right-translated element
            """
            return jax.vmap(self.atom_mfd.group.righttrans)(g, f)

        def inverse(self, g: jnp.array) -> jnp.array:
            """Inverse map of the Lie group

            :param g: element of the Lie group M^k
            :return: element of M^k that is inverse to g
            """
            return jax.vmap(self.atom_mfd.group.inverse)(g)

        def exp(self, v: jnp.array) -> jnp.array:
            """Group exponential

            :param v: tangent vector at e
            :return: point at time 1 of the 1-parameter subgroup with initial velocity v
            """
            return jax.vmap(self.atom_mfd.group.exp)(v)

        retr = exp

        def log(self, g: jnp.array) -> jnp.array:
            """Group logarithm

            :param g: element of M^k
            :return: vector v at e with exp(v) = g
            """
            return jax.vmap(self.atom_mfd.group.log)(g)

        def adjrep(self, g: jnp.array, v: jnp.array) -> jnp.array:
            """Adjoint representation of g applied to the tangent vector v at the identity

            :param g: element of the Lie group M^k
            :param v: tangent vector at the identity
            :return: tangent vector at the identity
            """
            return jax.vmap(self.atom_mfd.group.adjrep)(g, v)

        def jacobiField(self, g: jnp.array, f: jnp.array, t: float, X: jnp.array) -> jnp.array:
            """
            Evaluates a Jacobi field (with boundary conditions gam'(0) = X, gam'(1) = 0) along the geodesic gam of the
            CCS connection from g to f.

            :param g: element of M^k
            :param f: element of M^k
            :param t: scalar in [0,1]
            :param X: tangent vector at p
            :return: [gam(t), J], where J is the value of the Jacobi field (which is an element-wise Jacobi field) at gam(t)
            """
            return jax.vmap(self.atom_mfd.group.jacobiField, (0, 0, None, 0))(g, f, t, X)
