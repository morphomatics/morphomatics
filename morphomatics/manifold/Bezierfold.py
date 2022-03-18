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

import scipy.integrate as integrate

from morphomatics.manifold import Manifold

from joblib import Parallel, delayed

import time

import copy

from morphomatics.stats.RiemannianRegression import RiemannianRegression
from morphomatics.stats import ExponentialBarycenter
from morphomatics.geom.BezierSpline import BezierSpline
from morphomatics.manifold import Metric, Connection


class Bezierfold(Manifold):
    """Manifold of Bézier splines (of fixed degrees)

    """

    def __init__(self, M: Manifold, degrees, isscycle=False):
        """

        :arg M: base manifold in which the curves lie
        :arg degrees: array of degrees of the Bézier segments
        """
        assert M is not None

        self._M = M

        self._degrees = np.atleast_1d(degrees)

        if isscycle:
            name = 'Manifold of closed Bézier splines of degrees {d} through '.format(d=degrees) + M.__str__
            # number of independent control points
            K = np.sum(self._degrees) - 1
        else:
            name = 'Manifold of non-closed Bézier splines of degrees {d} through '.format(d=degrees) + M.__str__
            K = np.sum(self._degrees)

        dimension = (K + 1) * M.dim
        point_shape = [K, M.point_shape]
        self._K = K
        super().__init__(name, dimension, point_shape)

        self._iscycle=isscycle

    def initFunctionalBasedStructure(self):
        """
        Instantiate functional-based structure with discrete methods.
        """
        structure = Bezierfold.FunctionalBasedStructure(self)
        self._metric = structure
        self._connec = structure

    @property
    def M(self):
        """Return the underlying manifold
        """
        return self._M

    @property
    def degrees(self):
        """Return vector of segment degrees
        """
        return self._degrees

    @property
    def nsegments(self):
        """Returns the number of segments."""
        return len(self._degrees)

    @property
    def K(self):
        """Return the generalized degree of B, i.e., the number of independent control points - 1
        """
        return self._K

    @property
    def iscycle(self):
        """Return whether the Bezierfold consists of non-closed or closed splines
        """
        return self._iscycle

    def correct_type(self, B: BezierSpline):
        """Check whether B has the right segment degrees"""
        if B.nsegments != self.nsegments:
            return False
        else:
            if np.all(np.atleast_1d(B.degrees) == self.degrees):
                return True
            else:
                return False

    def rand(self):
        # TODO: sample control points from normal convex neighborhood
        return

    def randvec(self, B:BezierSpline):
        # TODO: sample random vectors at control points of B and compute corresponding generalized Jacobi field
        return

    def zerovec(self):
        # TODO: how to represent zero vector field?
        return

    class FunctionalBasedStructure(Metric, Connection):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product.
        """

        def __init__(self, Bf):
            """
            Constructor.
            """
            self._Bf = Bf

        @property
        def __str__(self):
            return "Bezierfold-functional-based structure"

        def inner(self, B:BezierSpline, X, Y):
            """Functional-based metric
            Vector fields must be given as functions.

            :arg B: Bézier spline in M
            :arg X: generalized Jacobi Field along B
            :arg Y: generalized Jacobi Field along B
            :return: inner product of X and Y at B
            """
            # TODO
            return

        def norm(self, B, X):
            """Norm of tangent vectors induced by the functional-based metric

            :arg B: Bézier spline in M
            :arg X: generalized Jacobi Field along B
            :return: norm of X
            """
            return np.sqrt(self.inner(B, X, X))

        @property
        def typicaldist(self):
            # approximations via control points
            return self._Bf.K * self._Bf.M.metric.typicaldist

        def dist(self, A, B, n=5):
            """Approximate the distance between two Bézier splines

            :param A: Bézier spline
            :param B: Bézier spline
            :param n: work with discrete n-geodesic
            :return: length of l-geodesic between alp and bet (approximation of the distance)
            """

            Gam = self.discgeodesic(A, B, n=n)

            d = 0
            for i in range(len(Gam) - 1):
                y, t = self.loc_dist(Gam[i], Gam[i + 1])
                d += integrate.simps(y, t)

            return d

        def discgeodesic(self, A, B, n=5, eps=1e-10, nsteps=30, verbosity=1):
            """Discrete shortest path through space of Bézier curves of same degrees

            :param A: Bézier spline in manifold M
            :param B: Bézier spline in manifold M
            :param n: create discrete n-geodesic
            :param eps: iteration stops when the difference in energy between the new and old iterate drops below eps
            :param nsteps : maximal number of steps
            :param verbosity: 0 (no text) or 1 (print information on convergence)
            :return: control points of the Bézier splines along the shortest path
            """
            assert self._Bf.correct_type(A) and self._Bf.correct_type(B)

            def init_disc_curve(A, B, n):
                """Initialize discrete curve by aligning control points along geodesics

                :param A: Bézier spline
                :param B: Bézier spline
                :param n: discretization parameter
                :return H: discrete curve from alp to bet as list of Bézier splines
                """

                # initial discrete curve
                H = [A]

                # initialize control points along geodesics
                for i in range(1, n):
                    # go through segments
                    P = []
                    for l in range(self._Bf.nsegments):
                        m = np.array(A.control_points[l].shape)
                        # m[0] = self._Bf.degrees + 1
                        # logs between corresponding control points
                        X = np.zeros(m)
                        for j in range(m[0]):
                            X[j] = self._Bf.M.connec.log(A.control_points[l][j], B.control_points[l][j])

                        P_l = np.zeros(m)
                        for j in range(m[0]):
                            P_l[j] = self._Bf.M.connec.exp(A.control_points[0][j], i / n * X[j])

                        P.append(P_l)
                    # i-th point (i.e., spline) of geodesic
                    H.append(BezierSpline(self._Bf.M, P))
                # add end point
                H.append(B)

                return H

            # initialize path
            H = init_disc_curve(A, B, n)
            # initialize parameters
            Eold = self.disc_path_energy(H)
            Enew = self.disc_path_energy(H)
            step = 0
            # optimize path
            while (np.abs(Enew - Eold) > eps and step < nsteps) or step == 0:
                step += 1
                Eold = Enew
                H_old = copy.deepcopy(H)

                # cycle through inner splines
                for i in range(1, n):
                    # sample splines at so many discrete points as we have free control points
                    #t = np.linspace(0, self._Bf.nsegments, num=int(np.sum(self._Bf.degrees + 1)))
                    t = np.linspace(0, self._Bf.nsegments, num=self._Bf.K + 1)
                    # we have two data sets with the same time points
                    double_t = np.concatenate((t, t))

                    # evaluate preceding and succeeding splines at time points
                    h1 = H[i - 1].eval(t)
                    h2 = H[i + 1].eval(t)
                    Y = np.concatenate((h1, h2))
                    # solve regression problem for "middle spline"
                    regression = RiemannianRegression(self._Bf.M, Y, double_t, self._Bf.degrees,
                                                      verbosity=11 * verbosity)

                    H[i] = regression.trend
                # update energy
                Enew = self.disc_path_energy(H)

                # check whether energy has increased
                if Enew > Eold:
                    print('Stopped computing discrete geodesic because the energy increased in step ' + str(step) + '.')
                    return H_old

                if np.isnan(Enew):
                    # repeat
                    H = H_old
                    print('Had to repeat because of Nan-value.')
                else:
                    if verbosity:
                        print('Disc-Geo-Step', step, 'Energy:', Enew, 'Enew - Eold:', Enew - Eold)

            return H

        def loc_dist(self, A: BezierSpline, B: BezierSpline, t=None):
            """ Evaluate distance between two Bézier splines in M at several points

            :param A: Bézier spline
            :param B: Bézier spline
            :param t: vector with time points

            :return: vector with point-wise distances
            """
            if t is None:
                # sample segments of higher degree more densely
                t = np.linspace(0, 1, num=self._Bf.degrees[0] + 1)
                for i in range(1, A.nsegments):
                    t = np.concatenate((t, np.linspace(i, i+1, num=self._Bf.degrees[i] + 1)[1:]))

            a_val = A.eval(t)
            b_val = B.eval(t)
            d_M = np.zeros(len(t))
            for i in range(len(t)):
                d_M[i] = self._Bf.M.metric.dist(a_val[i], b_val[i])
            return d_M, t

        def disc_path_energy(self, H):
            """Discrete path energy

            :param H: discrete path given as ordered list of Bézier splines of the same degrees
            :return: energy of H
            """
            d = 0
            for i in range(len(H) - 1):
                dh, _ = self.loc_dist(H[i], H[i + 1])
                d += np.sum(dh ** 2, axis=0)

            return d

        def mean(self, B, n=3, delta=1e-5, min_stepsize=1e-10, nsteps=20, eps=1e-5, n_stepsGeo=10, verbosity=1):
            """Discrete mean of a set of Bézier splines

            :param B: list of Bézier splines
            :param n: use discrete n-geodesics
            :param delta: iteration stops when the difference in energy between the new and old iterate drops below
            delta
            :param min_stepsize: iteration stops when the step length is smaller than the given value
            :param nsteps: maximal number of iterations
            :param eps: see eps in discgeodesic
            :param n_stepsGeo: maximal number of iterations when computating discrete geodesics
            :param verbosity: 0 (no text) or 1 (print information on convergence)
            :return: mean curve
            """
            for b in B:
                assert self._Bf.correct_type(b)

            def legs(meaniterate):
                """ Construct legs of polygonal spider, i.e., discrete n-geodesics between mean iterate and input curves
                """
                return Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(delayed(self.discgeodesic)
                                                                                  (meaniterate, b, n=n, eps=eps,
                                                                                   nsteps=n_stepsGeo, verbosity=0)
                                                                                  for b in B)

            def loss(FF):
                """Loss in optimization for mean
                """
                G = 0
                for HH in FF:
                    G = G + self.disc_path_energy(HH)
                return G

            # measure computation time
            begin_mean = time.time()

            # initialize i-th control point of the mean as the mean of the i-th control points of the data
            C = ExponentialBarycenter
            P = []
            # loop through segments
            for l in range(self._Bf.nsegments):
                # get shape of array of control points
                m = np.array(B[0].control_points[l].shape)
                # m[0] = self._Bf.degrees + 1

                P_l = np.zeros(m)
                for i in range(self._Bf.degrees[l] + 1):
                    D = []
                    # take the i-th control point of the l-th segment from all data curves
                    for bet in B:
                        D.append(bet.control_points[l][i])

                    # initialize i-th control point of the l-th segment as mean of corresponding data control points
                    P_l[i] = C.compute(self._Bf.M, D)

                P.append(P_l)


            # initial guess
            # bet_mean = B[0]
            bet_mean = BezierSpline(self._Bf.M, P)

            # initial legs
            F = legs(bet_mean)
            # initialize stopping parameters
            Eold = 10
            Enew = 1
            stepsize = 10
            step = 0
            while np.abs(Enew - Eold) > delta and stepsize > min_stepsize and step <= nsteps:
                # one more step
                step += 1
                # save old "stopping parameters"
                Eold = Enew
                F_old = F
                old_mean = BezierSpline(self._Bf.M, bet_mean.control_points)

                # new data for regression
                # sample splines at so many discrete points as we have free control points
                #t = np.linspace(0, self._Bf.nsegments, num=int(np.sum(self._Bf.degrees + 1)))
                t = np.linspace(0, self._Bf.nsegments, num=self._Bf.K + 1)
                Y = []

                for H in F:
                    # evaluate first "joint" at t
                    Y.append(H[1].eval(t))

                # make regression w.r.t. mean values -> faster
                mean_Y = np.zeros_like(Y[0])
                for i in range(len(mean_Y)):
                    dat = []
                    for j in range(len(Y)):
                        # take value of each curve at time t_i
                        dat.append(Y[j][i])
                    # mean at t_i if first joints
                    mean_Y[i] = C.compute(self._Bf.M, dat)

                if verbosity == 2:
                    print('Step ' + str(step) + ': Updating the mean...')

                regression = RiemannianRegression(self._Bf.M, mean_Y, t, self._Bf.degrees, iscycle=self._Bf.iscycle,
                                                  verbosity=2)
                bet_mean = regression.trend

                # update discrete paths
                if verbosity == 2:
                    print('Step ' + str(step) + ': Updating discrete paths...')
                    start = time.time()
                F = legs(bet_mean)

                if verbosity == 2:
                    end = time.time()
                    print('...took ' + "{:.2f}".format(end - start) + ' seconds to update the legs.')

                    print('Evaluating energy...')
                Enew = loss(F)

                # check for divergence
                if Enew > Eold:
                    print('Stopped because the energy increased.')
                    finish_mean = time.time()
                    print('Computing the mean took ' + "{:.2f}".format(finish_mean - begin_mean) + ' seconds.')
                    return old_mean, F_old

                # compute step size
                step_size = 0
                for i, p in enumerate(bet_mean.control_points[0]):
                    step_size += self._Bf.M.metric.dist(p, old_mean.control_points[0][i]) ** 2
                stepsize = np.sqrt(step_size)

                if verbosity > 0:
                    print('Mean-Comp-Step', step, 'Energy:', Enew, 'Enew - Eold:', Enew - Eold)

            finish_mean = time.time()
            print('Computing the mean took ' + "{:.2f}".format(finish_mean - begin_mean) + '.')

            return bet_mean, F

        def gram(self, B, B_mean=None, F=None, n=5, delta=1e-5, min_stepSize=1e-10, nsteps=20, eps=1e-5, n_stepsGeo=10,
                 verbosity=2):
            """Approximates the Gram matrix for a curve data set

            :param B: list of Bézier splines
            :param B_mean: mean of curves in B
            :param F: discrete spider, i.e, discrete paths from mean to data
            :param n: see mean method
            :param delta: see mean method
            :param min_stepSize: see mean method
            :param nsteps: see mean method
            :param eps: see mean method
            :param n_stepsGeo: see mean method
            :param verbosity: see mean method
            :return G: Gram matrix
            :return bet_mean: mean curve of data curves
            :return F: discrete geodesics from mean to data curves
            """

            if B_mean is None:
                B_mean, F = self.mean(B, n=n, delta=delta, min_stepsize=min_stepSize, nsteps=nsteps, eps=eps,
                                      n_stepsGeo=n_stepsGeo, verbosity=verbosity)

            if verbosity == 2:
                print('Computing Gram matrix...')

            n = len(F)
            G = np.zeros((n, n))
            for i, si in enumerate(F):
                for j, sj in enumerate(F[i:], start=i):
                    G[i, j] = n / 2 * (
                            self.dist(B_mean, si[1], n=1) ** 2 + self.dist(B_mean, sj[1], n=1) ** 2
                            - self.dist(si[1], sj[1], n=1) ** 2)
                    G[j, i] = G[i, j]

            return G, B_mean, F

        ### not imlemented ###

        def geopoint(self, alp, bet, t):
            return

        def pairmean(self, alp, bet):
            return

        def proj(self, X, H):
            #TODO
            return

        egrad2rgrad = proj

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            return

        def retr(self, R, X):
            # TODO
            return self.exp(R, X)

        def exp(self, R, X):
            return

        def log(self, R, Q):
            return

        def transp(self, R, Q, X):
            return

        def jacobiField(self, R, Q, t, X):
            return

        def adjJacobi(self, R, Q, t, X):
            return
