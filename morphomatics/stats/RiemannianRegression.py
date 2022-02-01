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
from morphomatics.geom import BezierSpline
from morphomatics.geom.BezierSpline import decasteljau
from morphomatics.stats import ExponentialBarycenter

from morphomatics.manifold import Manifold
from morphomatics.manifold.ManoptWrap import ManoptWrap
from pymanopt import Problem
from pymanopt.manifolds import Product
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.solvers import SteepestDescent, ConjugateGradient


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
        :param degrees: vector of length L; the l-th entry is the degree of the l-th segment of the spline. All entries must
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
        degrees = np.atleast_1d(degrees)

        self._M = M
        self._Y = Y
        self._param = param

        pymanoptM = ManoptWrap(M)

        # Cost
        def cost(P):
            P = np.stack(P)
            control_points = self.full_set(M, P, degrees, iscycle)

            return self.sumOfSquared(BezierSpline(M, control_points, iscycle=iscycle), Y, param)

        #MMM = Product([M for i in range(degrees[0])]) # for conjugated gradient

        # Gradient
        def grad(P):
            P = np.stack(P)
            control_points = self.full_set(M, P, degrees, iscycle)
            grad_E = self.gradSumOfSquared(BezierSpline(M, control_points, iscycle=iscycle), Y, param)
            grad_E = self.indep_set(grad_E, iscycle)

            # return _ProductTangentVector([grad_E[0][i] for i in range(len(grad_E[0]))]) # for conjugated gradient
            return np.concatenate(grad_E)

        # Solve optimization problem with pymanopt by optimizing over independent control points
        if iscycle:
            N = Product([pymanoptM] * np.sum(degrees - 1))
        else:
            N = Product([pymanoptM] * (np.sum(degrees - 1) + 2))

        problem = Problem(manifold=N, cost=cost, grad=grad, verbosity=verbosity)

        # solver = ConjugateGradient(maxtime=maxtime, maxiter=maxiter, mingradnorm=mingradnorm,
        #                            minstepsize=minstepsize, maxcostevals=maxcostevals, logverbosity=2)

        solver = SteepestDescent(maxtime=maxtime, maxiter=maxiter, mingradnorm=mingradnorm,
                                 minstepsize=minstepsize, maxcostevals=maxcostevals, logverbosity=2)

        if P_init is None:
            P_init = self.initControlPoints(M, Y, param, degrees, iscycle)
        P_init = self.indep_set(P_init, iscycle)

        P_opt, opt_log = solver.solve(problem, list(np.concatenate(P_init)))
        P_opt = self.full_set(M, np.stack(P_opt, axis=0), degrees, iscycle)

        self._spline = BezierSpline(M, P_opt, iscycle=iscycle)
        self._unexplained_variance = opt_log['final_values']["f(x)"] / len(Y)

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
        total_var = ExponentialBarycenter.total_variance(self._M, list(self._Y))

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
        :param degrees: vector of length L; the l-th entry is the degree of the l-th segment of the spline. All entries must
        be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.
        :param iscylce: boolean that determines whether a closed curve C1 spline shall be modeled.

        :return P: list of length L containing arrays of control points. The l-th entry is an
               array with degrees(l)+1 elements of M, that are ordered along the first dimension.
        """
        assert M.metric and M.connec
        degrees = np.atleast_1d(degrees)
        assert all(degrees >= 1)
        if iscycle:
            # check for minimum number of control points
            assert degrees.size > 1 and degrees[0] >= 3 and degrees[-1] >= 3

        # sort data
        ind = np.argsort(param)
        param[:] = param[ind]
        Y[:] = Y[ind]

        data, t = RiemannianRegression.segments_from_data(Y, param)

        assert len(data) == degrees.size

        P = []
        for l, d in enumerate(degrees):
            siz = np.array(data[l].shape)
            siz[0] = d + 1
            Pl = np.zeros(siz)

            # first segment
            if l == 0:
                for i in range(0, d + 1):
                    Pl[i] = M.connec.geopoint(data[0][0], data[0][-1], i / d)

            # initial values for the control points of the remaining segments
            else:
                # C^1 condition
                Pl[0] = P[l - 1][-1]
                Pl[1] = M.connec.geopoint(P[l - 1][-2], P[l - 1][-1], 2)
                # If there are remaining control points, they are free; we initialize them along a geodesic.
                if d > 1:
                    if l != degrees.size - 1 or not iscycle:
                        for i in range(2, d + 1):
                            Pl[i] = M.connec.geopoint(Pl[1], data[l][-1], i / d)
                    # last segment of closed spline
                    else:
                        # C^1 condition
                        Pl[-1] = P[0][0]
                        Pl[-2] = M.connec.geopoint(P[0][1], P[0][0], 2)
                        # d-3 free control points
                        for i in range(2, d - 1):
                            # on geodesic between neighbours
                            Pl[i] = M.connec.geopoint(Pl[1], Pl[-2], (i - 1) / (d - 2))

            P.append(Pl)

        return P

    @staticmethod
    def sumOfSquared(B: BezierSpline, Y, param):
        """Computes sum of squared distances between the spline
        defined by P and data Y.
        :param B: Bézier spline
        :param Y: array with data points along first axis
        :param param: vector with corresponding parameter values
        :return: non-negative scalar
        """
        s = 0
        for i, t in enumerate(param):
            s += B._M.metric.dist(B.eval(t), Y[i]) ** 2

        return s

    @staticmethod
    def gradSumOfSquared(B: BezierSpline, Y, param):
        """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
        points.
        :param B: Bézier spline with K segments
        :param Y: array that contains data in the manifold where B is defined (along first axis).
        :param param: vector with the sorted parameter values that correspond to the data in Y. All values must be
        in [0, B.nsegments].
        :return: gradients at the control points of B
        """
        assert all(0 <= param) and all(param <= B.nsegments)
        assert Y.shape[0] == param.shape[0]

        M = B._M

        P = B.control_points

        L = B.nsegments

        # sort data (maybe not necessary)
        ind = np.argsort(param)
        param[:] = param[ind]
        Y[:] = Y[ind]

        # Initiate gradients
        grad_E = []
        for l in range(L):
            grad_E.append(np.zeros_like(P[l]))

        # Distinct parameters in param with multiplicity
        u = np.unique(param)

        for t in u:
            # First, we sum up all gradients of tau_j(p) = d(p,y_j)^2 that live in the same tangent space; the value t
            # appears count[i] times.
            grad_dist = np.zeros_like(Y[0])  # would be cleaner with M.zerovec(decasteljau(M, P[ind], t_seg))

            ind, t_seg = B.segmentize(t)
            for jj in np.nonzero(param == t)[0]:
                grad_dist += -2 * M.connec.log(decasteljau(M, P[ind], t_seg), Y[jj])

            # add up adjointly transported contributions
            grad_E[ind] += B.adjDpB(t, grad_dist)

        # Taking care of C1 conditions
        for l in range(1, L):
            X_plus = grad_E[l][1]  # gradient w.r.t. p_l^+
            X_l = M.connec.adjDxgeo(P[l][0], P[l][1], 1, X_plus)
            X_minus = M.connec.adjDxgeo(P[l - 1][-2], P[l][1], 1, X_plus)

            # Final gradients at p_l and p_l^-
            grad_E[l - 1][-1] += grad_E[l][0] + X_l
            grad_E[l - 1][-2] += X_minus

            # Everything that is not needed anymore is set to 0 s.t. it cannot cause
            # bugs (here the gradient at p_l^+ and at p_l for the lower segment).
            grad_E[l][0] *= 0
            grad_E[l][1] *= 0

        # Taking care for additional C1 conditions in the case of a closed curve.
        if B.iscycle:
            X_plus = grad_E[0][1]  # gradient w.r.t. p_0^+
            X_l = M.connec.adjDxgeo(P[0][0], P[0][1], 1, X_plus)
            X_minus = M.connec.adjDxgeo(P[-1][-2], P[0][1], 1, X_plus)

            # Final gradients at p_l and p_l^-
            grad_E[-1][-1] += grad_E[0][0] + X_l
            grad_E[-1][-2] += X_minus

            # Everything that is not needed anymore is set to 0 s.t. it cannot cause bugs (here the gradient at p_0 and at
            # p_0^+ w.r.t the lower segment).
            grad_E[0][0] *= 0
            grad_E[0][1] *= 0

        return grad_E

    @staticmethod
    def segments_from_data(Y, param):
        """Divide data according to segments

        :param Y: array of values
        :param param: vector with corresponding nonnegative values sorted in ascending order
        :return: List of data arrays. The l-th entry contains data belonging to the l-th segment;
        list with corresponding parameter values. Data at a knot is assigned to the lower segment.
        """
        assert np.all(np.diff(param) >= 0) and np.all(param >= 0)
        assert Y.shape[0] == param.size

        # get the segments the data belongs to
        def segment(t):
            """Choose the correct segment and value for the parameter t
            :param t: scalar
            :return: index of segment, that is, i for t in (i,i+1] (0 if t=0)
            """
            # choose correct segment
            if t == 0:
                ind = 0
            elif t == np.round(t):
                ind = t - 1
                ind = ind.astype(int)
            else:
                ind = np.floor(t).astype(int)
            return ind

        # get indices where the new segment begins
        s = np.zeros_like(param, int)
        for i, t in enumerate(param):
            s[i] = segment(t)

        _, ind, count = np.unique(s, return_index=True, return_counts=True)

        data = []
        t = []
        for i, d in enumerate(ind):
            data.append(Y[d:d + count[i]])
            t.append(param[d:d + count[i]])

        return data, t

    @staticmethod
    def full_set(M: Manifold, P, degrees, iscycle):
        """Compute all control points of a C^1 Bézier spline from the independent ones."""
        control_points = []
        start = 0
        siz = np.array(P.shape)
        for i, d in enumerate(degrees):
            deg = degrees[i]
            if i == 0:
                if not iscycle:
                    # all control points of the first segment are independent
                    control_points.append(np.stack(P[:deg + 1]))
                    start += deg + 1
                else:
                    # add first two control points
                    siz[0] = deg + 1
                    C = np.zeros(siz)
                    C[0] = P[-1]
                    C[1] = M.connec.geopoint(P[-2], P[-1], 2)
                    C[2:] = P[:deg - 1]

                    control_points.append(C)
                    start += deg - 1
            else:
                # add first two control points
                siz[0] = deg + 1
                C = np.zeros(siz)
                C[0] = control_points[-1][-1]
                C[1] = M.connec.geopoint(control_points[-1][-2], control_points[-1][-1], 2)
                C[2:] = P[start:start + deg - 1]

                control_points.append(C)
                start += deg - 1

        return control_points

    @staticmethod
    def indep_set(obj, iscycle):
        """Return array with independent control points or gradients from full set."""
        for l in range(len(obj)):
            if l == 0:
                if iscycle:
                    obj[0] = obj[0][2:]
            else:
                obj[l] = obj[l][2:]

        return obj
