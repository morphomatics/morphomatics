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

from morphomatics.geom.bezier_spline import BezierSpline, full_set, indep_set
from morphomatics.manifold import Metric
from morphomatics.manifold import Manifold, TangentBundle, PowerManifold


class CubicBezierfold(Manifold):
    """Manifold of _cubic_ Bézier splines.

    """

    def __init__(self, M: Manifold, n_segments: int, isscycle: bool = False, structure='GeneralizedSasaki'):
        """Manifold of cubic Bézier splines.

        :arg M: base manifold in which the curves lie
        :arg n_segments: number of segments
        :arg iscycle: boolean indicating whether the splines are closed
        :arg structure: type of geometric structure
        """

        self._M = M

        self._degrees = np.full(n_segments, 3)

        if isscycle:
            name = 'Manifold of closed, cubic Bézier splines through ' + str(M)
            K = 2*n_segments - 1
        else:
            name = 'Manifold of cubic Bézier splines through ' + str(M)
            K = 2*n_segments + 1

        dimension = (K + 1) * M.dim
        point_shape = ((K + 1)//2, 2) + M.point_shape
        self._K = K
        super().__init__(name, dimension, point_shape)

        self._iscycle = isscycle

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children+(self.M,), aux+(self.nsegments, self.iscycle)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, M = children
        *aux_data, nsegments, iscycle = aux_data
        obj = cls(M, nsegments, iscycle, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initGeneralizedSasakiStructure(self):
        """
        Instantiate generalized Sasaki structure with discrete methods.
        """
        structure = CubicBezierfold.GeneralizedSasakiStructure(self)
        self._metric = structure
        self._connec = structure

    @property
    def M(self) -> Manifold:
        """Return the underlying manifold
        """
        return self._M

    @property
    def nsegments(self) -> int:
        """Returns the number of segments."""
        return len(self._degrees)

    # TODO: likely not needed here anymore
    @property
    def K(self) -> int:
        """Return the generalized degree of a Bezier spline, i.e., the number of independent control points - 1
        """
        return self._K

    @property
    def iscycle(self) -> bool:
        """Return whether the Bezierfold consists of non-closed or closed splines
        """
        return self._iscycle

    def correct_type(self, B: BezierSpline) -> bool:
        """Check whether B has the right segment degrees"""
        if jnp.all(jnp.atleast_1d(B.degrees) == jnp.repeat(3, self.nsegments)):
            return True
        else:
            return False

    def rand(self, key: jax.Array) -> BezierSpline: #TODO: velocity repr.
        """Return random Bézier spline"""
        subkeys = jax.random.split(key, self.K + 1)
        return BezierSpline(self.M, full_set(self.M, jax.vmap(self.M.rand)(subkeys),
                                             self.degrees, self.iscycle))

    def randvec(self, B: BezierSpline, key: jax.Array) -> jnp.array: #TODO: velocity repr.
        """Return random vector for every independent control point"""
        pts = indep_set(B, self.iscycle)
        subkeys = jax.random.split(key, len(pts))
        return jax.vmap(self.M.randvec)(pts, subkeys)

    def zerovec(self) -> jnp.array: #TODO: velocity repr.
        """Return zero vector for every independent control point"""
        return jnp.array([self.M.zerovec() for k in self.K + 1])

    def to_coords(self, B: BezierSpline) -> jnp.array:
        """Get initial and final velocities (elements of the tangent bundle) of the segments of a C^1 Bézier spline with cubic
        segments. velocities at connections are identified and returned only once.

        :param B: Bézier spline with cubic segments through a Riemannian manifold M
        :return: array of elements (ordered along the first dimension) of the tangent bundle TM

        ATTENTION: made for splines with cubic segments only!
        """
        assert jnp.all(B.degrees == 3)

        def f(p):
            return jnp.array([p[-1], -B._M.connec.log(p[-1], p[-2])])

        b = jax.vmap(f)(B.control_points)

        if not self.iscycle:
            a = jnp.array([
                B.control_points[0, 0], B._M.connec.log(B.control_points[0, 0], B.control_points[0, 1])
            ])

            return jnp.concatenate((a[None, ...], b))
        else:
            return b

    def from_coords(self, Q: jnp.array) -> BezierSpline:
        """Compute the cubic-only Bézier spline corresponding to the given velocities

        :param Q: array of velocities
        :return: Bézier spline with cubic segments that corresponds to P

        ATTENTION: made for velocities of splines with cubic segments only!
        """

        def f(pu, qw):
            p, u = pu[0], pu[1]
            q, w = qw[0], qw[1]
            return jnp.array([p, self._M.connec.exp(p, u), self._M.connec.exp(q, -w), q])

        if self.iscycle:
            # last velocity vector is also first velocity vector
            Q = jnp.concatenate([Q[None, -1], Q])

        P = jax.vmap(f)(Q[:-1], Q[1:])

        return BezierSpline(self._M, P, iscycle=self.iscycle)

    def proj(self, pu, vw):
        raise NotImplementedError('This function has not been implemented yet.')

    ############################## Sasaki structure ##############################
    class GeneralizedSasakiStructure(Metric):
        """
        This class implements the generalization of the Sasaki metric to Bézier splines with cubic segments
        """

        def __init__(self, Bf: Manifold, Ns: int = 3):
            """
            Constructor.

            :param Bf: Bézierfold object
            :param Ns: scalar that determines the number of discretization steps used in the approximation of the
            exponential and logarithm maps in the tangent bundle
            """
            self._tangent_bundle_power = PowerManifold(TangentBundle(Bf.M), Bf.nsegments + 1)
            self._Bf = Bf
            self.Ns = Ns

        def __str__(self):
            return "Generalized Sasaki structure"

        @property
        def typicaldist(self) -> float:
            return self._tangent_bundle_power.metric.typicaldist

        def inner(self, p_B: jnp.array, v: jnp.array, w: jnp.array) -> float:
            """Generalized Sasaki metric

            :param p_B: velocities of a Bézier spline
            :param v: tangent vector in the tangent space of the velocities of B
            :param w: tangent vector in the tangent space of the velocities of B
            :return: inner product between X and Y
            """

            return self._tangent_bundle_power.metric.inner(p_B, v, w)

        def flat(self, p_B, v):
            """Lower vector X at p with the metric"""
            return self._tangent_bundle_power.metric.flat(p_B, v)

        def sharp(self, p_B, dv):
            """Raise covector dX at p with the metric"""
            return self._tangent_bundle_power.metric.sharp(p_B, dv)

        def egrad2rgrad(self, p, X):
            return self._Bf.proj

        def exp(self, p_B: jnp.array, v: jnp.array) -> jnp.array:
            """Exponential map

            :param p_B: velocities of a Bézier spline
            :param v: tangent vector in the tangent space of the velocities of Bf
            :return: velocities of the Bézier spline at time 1 on the geodesic with initial velocity v
            """
            return self._tangent_bundle_power.connec.exp(p_B, v)

        def log(self, p_A: jnp.array, p_B: jnp.array) -> jnp.array:
            """Riemannian logarithm map

            :param p_A: velocities of a Bézier spline A
            :param p_B: velocities of a Bézier spline B
            :return: tangent vector in the tangent space of the velocities of A pointing to the velocities of B
            """
            return self._tangent_bundle_power.connec.log(p_A, p_B)

        def curvature_tensor(self, p_B: jnp.array, v: jnp.array, w: jnp.array, x: jnp.array) -> jnp.array:
            """Riemmannian curvature tensor at a point of the Bézierfold

            :param p_B: velocities of a Bézier spline
            :param v: tangent vector in the tangent space of the velocities of B
            :param w: tangent vector in the tangent space of the velocities of B
            :param x: tangent vector in the tangent space of the velocities of B
            :return: tangent vector in the tangent space of the velocities of B
            """
            return self._tangent_bundle_power.connec.curvature_tensor(p_B, v, w, x)

        def geopoint(self, p_A: jnp.array, p_B: jnp.array, t: float) -> jnp.array:
            """Evaluate the geodesic through the Bézierfold between A and Bf at time t

            :param p_A: velocities of a Bézier spline A
            :param p_B: velocities of a Bézier spline B
            :param t: scalar between 0 and 1
            :return: Bézier spline at time t on the geodesic from A to B
            """
            return self.exp(p_A, t * self.log(p_A, p_B))

        def retr(self, p_B: jnp.array, v: jnp.array) -> jnp.array:
            return self.exp(p_B, v)

        def transp(self, p_A: jnp.array, p_B: jnp.array, v: jnp.array) -> jnp.array:
            """Parallel transport along a geoodesic

            :param p_A: velocities of a Bézier spline A
            :param p_B: velocities of a Bézier spline B
            :param v: tangent vector in the tangent space of the velocities of A
            :return: tangent vector in the tangent space of the velocities of B: parallel transport of v along the
            geodesic from A to B
            """
            return self._tangent_bundle_power.connec.transp(p_A, p_B, v)

        def pairmean(self, p_A: jnp.array, p_B: jnp.array) -> jnp.array:
            """Fréchet mean of 2 splines

            :param A: velocities of a Bézier spline A
            :param B: velocities of a Bézier spline B
            :return: velocities of the mean of A and B
            """
            return self.geopoint(p_A, p_B, .5)

        def dist(self, p_A: jnp.array, p_B: jnp.array) -> float:
            """Distance function that is induced on the Bézierfold by the generalized Sasaki metric

            :param p_A: velocities of a Bézier spline A
            :param p_B: velocities of a Bézier spline B
            :return: distance between A and B
            """
            return self._tangent_bundle_power.metric.dist(p_A, p_B)

        def jacobiField(self, p_A: jnp.array, p_B: jnp.array, t: float, X: jnp.array) -> jnp.array:
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, p_A: jnp.array, p_B: jnp.array, t: float, X: jnp.array) -> jnp.array:
            raise NotImplementedError('This function has not been implemented yet.')
