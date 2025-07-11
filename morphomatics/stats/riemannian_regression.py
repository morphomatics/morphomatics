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

from functools import partial, cached_property
from typing import List, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.geom.bezier_spline import BezierSpline, full_set, indep_set
from morphomatics.manifold import Manifold, PowerManifold
from morphomatics.opt import RiemannianSteepestDescent
from morphomatics.stats import ExponentialBarycenter


class RiemannianRegression(object):
    """
    Higher-order regression for estimation of relationship between
    single explanatory and manifold-valued dependent variable.

    The relationship is modeled via intrinsic Bezier splines (morphomatics.manifold.BezierSpline).

    See:
    Martin Hanik, Hans-Christian Hege, Anja Hennemuth, Christoph von Tycowicz:
    Nonlinear Regression on Manifolds for Shape Analysis using Intrinsic Bézier Splines.
    Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2020.
    """

    def __init__(self, M: Manifold, Y: jnp.array, param: jnp.array, degree: int = 1, n_segments: int = 1, iscycle=False,
                 P_init=None, maxiter=100, mingradnorm=1e-6):
        """Compute regression with Bézier splines for data in a manifold M.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        inY. The integer part determines the segment to which the data point belongs.
        :param degree: degree of each segment of the spline. Must be > 0/2 for non-/cyclic splines.
        :param n_segments: number of segments. Must be positive.
        :param iscycle: boolean that determines whether a closed curve C1 spline shall be modeled.
        :param P_init: initial guess
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of control points of the optimal Bézier spline
        """
        degrees = np.full(n_segments, degree)

        self._M = M
        self._Y = Y
        self._param = param

        # initial guess
        if P_init is None:
            P_init = self.initControlPoints(M, Y, param, degrees, iscycle)
        P_init = indep_set(P_init, iscycle)

        # fit spline to data
        P = RiemannianRegression.fit(M, Y, param, P_init, degree, n_segments, iscycle, maxiter, mingradnorm)

        # construct spline from ctrl. pts.
        P = full_set(M, P, degrees, iscycle)
        self._spline = BezierSpline(M, P, iscycle=iscycle)

    @staticmethod
    @partial(jax.jit, static_argnames=['degree', 'n_segments', 'iscycle'])
    def fit(M: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array, degree: int, n_segments: int, iscycle=False,
            maxiter=100, mingradnorm=1e-6) -> jnp.array:
        """Fit Bézier spline to data Y,param in a manifold M using gradient descent.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param P_init: initial guess (independent ctrl. pts. only, see #indep_set)
        :param degree: degree of each segment of the spline. Must be > 0/2 for non-/cyclic splines.
        :param n_segments: number of segments. Must be positive.
        :param iscycle: boolean that determines whether a closed curve C1 spline shall be modeled.
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of independent control points of the optimal Bézier spline.
        """
        degrees = np.full(n_segments, degree)

        # number of independent control points
        k = int(np.sum(degrees - 1)) + (0 if iscycle else 2)
        # search space: k-fold product of M
        N = PowerManifold(M, k)

        # Cost
        def cost(P):
            pts = full_set(M, P, degrees, iscycle)
            return sumOfSquared(BezierSpline(M, pts, iscycle=iscycle), Y, param) / len(Y)

        args = {'stepsize': 1., 'maxiter': maxiter, 'mingradnorm': mingradnorm}
        return RiemannianSteepestDescent.fixedpoint(N, cost, P_init, **args)

    @property
    def trend(self) -> BezierSpline:
        """
        :return: Estimated trajectory encoding relationship between
            explanatory and manifold-valued dependent variable.
        """
        return self._spline

    @cached_property
    def unexplained_variance(self) -> float:
        """Variance in the data set that is not explained by the regressed Bézier spline.
        """
        cost = sumOfSquared(self.trend, self._Y, self._param)
        return cost / len(self._Y)

    @property
    def R2statistic(self) -> float:
        """ Computes Fletcher's generalized R2 statistic for Bézier spline regression. For the definition see
                        Fletcher, Geodesic Regression on Riemannian Manifolds (2011), Eq. 7.

        :return: generalized R^2 statistic (in [0, 1])
        """

        # total variance
        total_var = ExponentialBarycenter.total_variance(self._M, self._Y)

        return 1 - self.unexplained_variance / total_var

    @staticmethod
    def initControlPoints(M: Manifold, Y: jnp.array, param: jnp.array,
                          degrees: jnp.array, iscycle: bool = False) -> jnp.array:
        """Computes an initial choice of control points for the gradient descent steps in non-cyclic Bézier spline
        regression.
        The control points are initialized "along geodesics" near the data
        points such that the differentiability conditions hold.

        :param M:  manifold
        :param Y:  array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param degrees: vector of length L; the l-th entry is the degrees of the l-th segment of the spline. All entries
        must be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.
        :param iscycle: boolean that determines whether a closed curve C1 spline shall be modeled.

        :return P: list of length L containing arrays of control points. The l-th entry is an
               array with degrees(l)+1 elements of M, that are ordered along the first dimension.
        """
        degrees = jnp.atleast_1d(degrees)

        # data, t = RiemannianRegression.segments_from_data(Y, param)
        data, _ = segments_from_data(Y, param)

        P = []
        for l, d in enumerate(degrees):
            Pl =[]
            # first segment
            if l == 0:
                for i in range(0, d + 1):
                    Pl.append(M.connec.geopoint(data[0][0], data[0][-1], i / d))
                Pl = jnp.array(Pl)

            # initial values for the control points of the remaining segments
            else:
                # C^1 condition
                Pl.append(P[-1][-1])
                Pl.append(M.connec.geopoint(P[-1][-2], P[-1][-1], 2))
                # If there are remaining control points, they are free; we initialize them along a geodesic.
                if d > 1:
                    if l != degrees.size - 1 or not iscycle:
                        for i in range(2, d + 1):
                            Pl.append(M.connec.geopoint(Pl[1], data[l][-1], i / d))
                        Pl = jnp.array(Pl)

                    # last segment of closed spline
                    else:
                        # d-3 free control points
                        for i in range(2, d - 1):
                            # on geodesic between neighbours
                            Pl.append(M.connec.geopoint(Pl[1], Pl[-2], (i - 1) / (d - 2)))
                        # C^1 condition
                        Pl.append(M.connec.geopoint(P[0][1], P[0][0], 2))
                        Pl.append(P[0][0])
                        Pl = jnp.array(Pl)

            P.append(Pl)

        return jnp.array(P)


def sumOfSquared(B: BezierSpline, Y: jnp.array, param: jnp.array) -> float:
    """Computes sum of squared distances between the spline
    defined by P and data Y.
    :param B: Bézier spline
    :param Y: array with data points along first axis
    :param param: vector with corresponding parameter values
    :return: non-negative scalar
    """

    return jnp.sum(jax.vmap(lambda y, t: B._M.metric.squared_dist(B.eval(t), y))(Y, param))


def gradSumOfSquared(B: BezierSpline, Y: jnp.array, param: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param Y: array that contains data in the manifold where B is defined (along first axis).
    :param param: vector with the sorted parameter values that correspond to the data in Y. All values must be
    in [0, B.nsegments].
    :return: gradients at the control points of B
    """

    M = B._M

    grad_i = lambda y, t: -2 * B.adjDpB(t, M.connec.log(B.eval(t), y))
    grad_E = jnp.sum(jax.vmap(grad_i)(Y, param), axis=0)

    # Taking care of C1/cycle conditions
    # return RiemannianRegression.grad_constraints(B, grad_E)
    return grad_constraints(B, grad_E)


def grad_constraints(B: BezierSpline, grad_E: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param grad_E: gradients at the control points for each segment
    :return: corrected gradients s.t. C1/cycle conditions are accounted for
    """

    M = B._M

    P = B.control_points

    L = B.nsegments

    # Taking care of C1 conditions
    for l in range(1, L):
        k = (B.degrees[l - 1] + B.degrees[l]) / B.degrees[l - 1]

        X_plus = grad_E[l][1]  # gradient w.r.t. p_l^+
        X_l = M.connec.adjDygeo(P[l - 1, -2], P[l, 0], k, X_plus)
        X_minus = M.connec.adjDxgeo(P[l - 1, -2], P[l, 0], k, X_plus)

        # Final gradients at p_l and p_l^-
        grad_E = grad_E.at[l - 1, -1].set(grad_E[l - 1, -1] + grad_E[l, 0] + X_l)
        grad_E = grad_E.at[l - 1, -2].set(grad_E[l - 1, -2] + X_minus)

    # Taking care for additional C1 conditions in the case of a closed curve.
    def do_cyclic(g):
        k = (B.degrees[-1] + B.degrees[0]) / B.degrees[-1]

        X_plus = g[0, 1]  # gradient w.r.t. p_0^+
        X_0 = M.connec.adjDxgeo(P[-1, -2], P[0, 0], k, X_plus)
        X_minus = M.connec.adjDxgeo(P[-1, -2], P[0, 0], k, X_plus)

        # Final gradients at p_l and p_l^-
        g = g.at[-1, -1].set(g[-1, -1] + g[0, 0] + X_0)
        g = g.at[-1, -2].set(g[-1, -2] + X_minus)

        return g

    return jax.lax.cond(B.iscycle, lambda g: do_cyclic(g), lambda g: g, grad_E)


def segments_from_data(Y: jnp.array, param: jnp.array) -> Tuple[List[jnp.array], List[jnp.array]]:
    """Divide data according to segments

    :param Y: array of values
    :param param: vector with corresponding nonnegative values
    :return: List of data arrays. The l-th entry contains data belonging to the l-th segment;
    list with corresponding parameter values. Data at a knot is assigned to the lower segment.
    """

    # sort data
    ind = jnp.argsort(param)
    param = param[ind]
    Y = Y[ind]

    # get the segments the data belongs to
    def segment(t):
        """Choose the correct segment and value for the parameter t
        :param t: scalar
        :return: index of segment, that is, i for t in (i,i+1] (0 if t=0)
        """
        # choose correct segment
        # if t == 0:
        #     ind = 0
        # elif t == jnp.round(t):
        #     ind = t - 1
        #     ind = ind.astype(int)
        # else:
        #     ind = jnp.floor(t).astype(int)
        # return ind
        return jnp.clip(jnp.ceil(t).astype(int) - 1, 0)

    # get indices where the new segment begins
    # s = jnp.zeros_like(param, int)
    # for i, t in enumerate(param):
    #     s = s.at[i].set(segment(t))
    s = jax.vmap(segment)(param)
    _, ind, count = jnp.unique(s, return_index=True, return_counts=True)

    data = []
    t = []
    for i, d in enumerate(ind):
        data.append(Y[d:d + count[i]])
        t.append(param[d:d + count[i]])

    return data, t
