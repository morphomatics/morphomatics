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

from morphomatics.geom import BezierSpline
from morphomatics.stats import ExponentialBarycenter

from morphomatics.manifold import Manifold
from morphomatics.manifold.ManoptWrap import ManoptWrap

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product
from pymanopt.optimizers import SteepestDescent


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

    def __init__(self, M: Manifold, Y, param, degrees, iscycle=False, P_init=None, verbosity=2, maxtime=100000,
                 maxiter=100, mingradnorm=1e-6, minstepsize=1e-10, maxcostevals=5000):
        """Compute regression with Bézier splines for data in a manifold M using pymanopt.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points in
        Y. The integer part determines the segment to which the data point belongs.
        :param degrees: vector of length L; the l-th entry is the degrees of the l-th segment of the spline. All entries must
        be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.
        :param iscycle: boolean that determines whether a closed curve C1 spline shall be modeled.
        :param P_init: initial guess
        :param verbosity: 0 is silent to gives the most information, see pymanopt's problem class

        :param maxtime: maximum time for steepest descent
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm
        :param minstepsize: stop iteration when step the stepsize is smaller than minstepsize
        :param maxcostevals: maximum number of allowed cost evaluations

        :return P: list of control points of the optimal Bézier spline
        """
        degrees = jnp.atleast_1d(degrees)

        # sort data
        ind = jnp.argsort(param)
        param = param[ind]
        Y = Y[ind]

        self._M = M
        self._Y = Y
        self._param = param

        pymanoptM = ManoptWrap(M)
        # Solve optimization problem with pymanopt by optimizing over independent control points
        if iscycle:
            N = Product([pymanoptM] * int(np.sum(degrees - 1)))
        else:
            N = Product([pymanoptM] * int(np.sum(degrees - 1) + 2))

        # Cost
        cost_ = jax.jit(lambda cp, Y_, t_: sumOfSquared(BezierSpline(M, cp, iscycle=iscycle), Y_, t_))
        @pymanopt.function.jax(N)
        def cost(*P):
            P = jnp.stack(list(P))
            control_points = full_set(M, P, degrees, iscycle)
            return cost_(control_points, Y, param)

        grad_ = jax.grad(cost, argnums=np.arange(len(N.point_layout)))
        @pymanopt.function.jax(N)
        def grad(*P):
            return jax.vmap(pymanoptM.euclidean_to_riemannian_gradient)(jnp.asarray(P), jnp.asarray(grad_(*P)))

        # # Gradient
        # grad_ = jax.jit(lambda cp, Y_, t_: self.gradSumOfSquared(BezierSpline(M, cp, iscycle=iscycle), Y_, t_))
        # def grad(P):
        #     control_points = self.full_set(M, P, degrees, iscycle)
        #     # grad_E = gradSumOfSquared(BezierSpline(M, control_points, iscycle=iscycle), Y, param)
        #     grad_E = grad_(control_points, Y, param)
        #     grad_E = indep_set(grad_E, iscycle)
        #
        #     return grad_E


        problem = Problem(manifold=N, cost=cost)# riemannian_gradient=grad)

        solver = SteepestDescent(max_time=maxtime, max_iterations=maxiter, min_gradient_norm=mingradnorm,
                                 min_step_size=minstepsize, max_cost_evaluations=maxcostevals, log_verbosity=2)

        if P_init is None:
            P_init = self.initControlPoints(M, Y, param, degrees, iscycle)

        P_init = indep_set(P_init, iscycle)

        opt = solver.run(problem, initial_point=P_init)
        P_opt = full_set(M, jnp.array(opt.point), degrees, iscycle)

        self._spline = BezierSpline(M, P_opt, iscycle=iscycle)
        self._unexplained_variance = opt.cost / len(Y)

    @property
    def trend(self):
        """
        :return: Estimated trajectory encoding relationship between
            explanatory and manifold-valued dependent variable.
        """
        return self._spline

    def unexplained_variance(self):
        """Variance in the data set that is not explained by the regressed Bézier spline.
        """
        return self._unexplained_variance

    @property
    def R2statistic(self):
        """ Computes Fletcher's generalized R2 statistic for Bézier spline regression. For the definition see
                        Fletcher, Geodesic Regression on Riemannian Manifolds (2011), Eq. 7.

        :return: generalized R^2 statistic (in [0, 1])
        """

        # total variance
        total_var = ExponentialBarycenter.total_variance(self._M, self._Y)

        return 1 - self.unexplained_variance() / total_var

    @staticmethod
    def initControlPoints(M: Manifold, Y, param, degrees, iscycle=False):
        """Computes an initial choice of control points for the gradient descent steps in non-cyclic Bézier spline
        regression.
        The control points are initialized "along geodesics" near the data
        points such that the differentiability conditions hold.

        :param M:  manifold
        :param Y:  array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points in
        Y. The integer part determines the segment to which the data point belongs.
        :param degrees: vector of length L; the l-th entry is the degrees of the l-th segment of the spline. All entries must
        be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.
        :param iscylce: boolean that determines whether a closed curve C1 spline shall be modeled.

        :return P: list of length L containing arrays of control points. The l-th entry is an
               array with degrees(l)+1 elements of M, that are ordered along the first dimension.
        """
        degrees = jnp.atleast_1d(degrees)

        # data, t = RiemannianRegression.segments_from_data(Y, param)
        data, t = segments_from_data(Y, param)

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


def sumOfSquared(B: BezierSpline, Y, param):
    """Computes sum of squared distances between the spline
    defined by P and data Y.
    :param B: Bézier spline
    :param Y: array with data points along first axis
    :param param: vector with corresponding parameter values
    :return: non-negative scalar
    """

    return jnp.sum(jax.vmap(lambda y, t: B._M.metric.squared_dist(B.eval(t), y))(Y, param))


def gradSumOfSquared(B: BezierSpline, Y, param):
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


def grad_constraints(B: BezierSpline, grad_E):
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


def segments_from_data(Y, param):
    """Divide data according to segments

    :param Y: array of values
    :param param: vector with corresponding nonnegative values sorted in ascending order
    :return: List of data arrays. The l-th entry contains data belonging to the l-th segment;
    list with corresponding parameter values. Data at a knot is assigned to the lower segment.
    """

    # get the segments the data belongs to
    def segment(t):
        """Choose the correct segment and value for the parameter t
        :param t: scalar
        :return: index of segment, that is, i for t in (i,i+1] (0 if t=0)
        """
        # choose correct segment
        if t == 0:
            ind = 0
        elif t == jnp.round(t):
            ind = t - 1
            ind = ind.astype(int)
        else:
            ind = jnp.floor(t).astype(int)
        return ind

    # get indices where the new segment begins
    s = jnp.zeros_like(param, int)
    for i, t in enumerate(param):
        s = s.at[i].set(segment(t))

    _, ind, count = jnp.unique(s, return_index=True, return_counts=True)

    data = []
    t = []
    for i, d in enumerate(ind):
        data.append(Y[d:d + count[i]])
        t.append(param[d:d + count[i]])

    return data, t


def full_set(M: Manifold, P, degrees, iscycle):
    """Compute all control points of a C^1 Bézier spline from the independent ones."""
    control_points = []
    start = 0
    for l, deg in enumerate(degrees):
        if l == 0:
            if not iscycle:
                # all control points of the first segment are independent
                control_points.append(P[:deg + 1])
                start = start + deg + 1
            else:
                # add first two control points
                C = jnp.vstack([jnp.expand_dims(P[-1], axis=0), jnp.expand_dims(M.connec.geopoint(P[-2], P[-1], 2),
                                                                                axis=0), P[:deg - 1]])
                control_points.append(C)
                start = start + deg - 1
        else:
            C = jnp.vstack([jnp.expand_dims(control_points[-1][-1], axis=0),
                jnp.expand_dims(M.connec.geopoint(control_points[-1][-2], control_points[-1][-1], 2), axis=0),
                P[start:start + deg - 1]])
            control_points.append(C)
            start = start + deg - 1

    return control_points


def indep_set(obj, iscycle):
    """Return array with independent control points or gradients from full set."""
    ind_pts = []
    for l in range(len(obj)):
        if l == 0 and not iscycle:
            ind_pts.append(obj[0])
        else:
            ind_pts.append(obj[l, 2:])
    return jnp.vstack(ind_pts)
