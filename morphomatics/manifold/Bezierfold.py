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

from ..stats.RiemannianRegression import RiemannianRegression
from ..stats import ExponentialBarycenter
from ..geom.BezierSpline import BezierSpline


class Bezierfold(Manifold):
    """Manifold of Bézier curves (of fixed degree)

    Only for single-segment curves for now.
    """

    def __init__(self, M: Manifold, degree):
        """

        :arg M: base manifold in which the curves lie
        :arg degree: degree of the Bézier curves
        """
        assert M is not None

        self._M = M

        self._degree = degree

        name = 'Manifold of Bézier curves of degree {d} through '.format(d=degree)+M.__str__
        K = np.sum(self._degree) - 1
        dimension = K * M.dim
        point_shape = [K, M.point_shape]
        super().__init__(name, dimension, point_shape)

    @property
    def typicaldist(self):
        return

    def inner(self, bet, X, Y):
        """Functional-based metric
        Vector fields must be given as functions.

        :arg bet: Bézier curve in M
        :arg X: generalized Jacobi Field along B
        :arg Y: generalized Jacobi Field along B
        :return: inner product of X and Y at B
        """
        assert bet.degrees == self._degree
        # TODO
        return

    def norm(self, bet, X):
        """Norm of tangent vectors induced by the functional-based metric

        :arg bet: Bézier curve in M
        :arg X: generalized Jacobi Field along B
        :return: norm of X
        """
        assert bet.degrees() == self._degree

        return np.sqrt(self.inner(bet, X, X))

    def proj(self, X, H):
        # TODO
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
        # TODO
        return

    def log(self, R, Q):
        # TODO
        return

    def geopoint(self, R, Q, t):
        # TODO
        return

    def discgeodesic(self, alp, bet, n=5, eps=1e-10, nsteps=30, verbosity=1):
        """Discrete shortest path through space of Bézier curves of same degree

        :param alp: Bézier curve in manifold M
        :param bet: Bézier curve in manifold M
        :param n: create discrete n-geodesic
        :param eps: iteration stops when the difference in energy between the new and old iterate drops below eps
        :param nsteps : maximal number of steps
        :param verbosity: 0 (no text) or 1 (print information on convergence)
        :return: control points of the Bézier curves along the shortest path
        """

        assert alp.degrees == self._degree and bet.degrees == self._degree

        def init_disc_curve(alp, bet, n):
            """Initialize discrete curve by aligning control points along geodesics
            """

            # initial discrete curve
            m = np.array(alp.control_points[0].shape)
            m[0] = self._degree + 1
            H = [alp]
            # logs between corresponding control points
            X = np.zeros(m)
            for j in range(self._degree + 1):
                X[j] = self._M.connec.log(alp.control_points[0][j], bet.control_points[0][j])
            # initialize control points along geodesics
            for i in range(1, n):
                P = np.zeros(m)
                for j in range(self._degree + 1):
                    P[j] = self._M.connec.exp(alp.control_points[0][j], i / n * X[j])

                H.append(BezierSpline(self._M, [P]))

            H.append(bet)

            return H

        # initialize path
        H = init_disc_curve(alp, bet, n)

        Eold = self.disc_path_energy(H)
        Enew = self.disc_path_energy(H)
        step = 0
        # optimize path
        while (np.abs(Enew - Eold) > eps and step < nsteps) or step == 0:
            step += 1
            Eold = Enew
            H_old = copy.deepcopy(H)

            for i in range(1, n):
                t = np.linspace(0, 1, num=self._degree + 1)
                double_t = np.concatenate((t, t))

                h1 = H[i - 1].eval(t)
                h2 = H[i + 1].eval(t)
                Y = np.concatenate((h1, h2))

                regression = RiemannianRegression(self._M, Y, double_t, self._degree, verbosity=11*verbosity)

                H[i] = regression.trend

            Enew = self.disc_path_energy(H)

            # check whether energy has increased
            if Enew > Eold:
                print('Stopped computing discrete geodesic because the energy increased in step '+str(step)+'.')
                return H_old

            if np.isnan(Enew):
                # repeat
                H = H_old
                print('Had to repeat because of Nan-value.')
            else:
                if verbosity:
                    print('Disc-Geo-Step', step, 'Energy:', Enew, 'Enew - Eold:', Enew - Eold)

        return H

    def loc_dist(self, alp: BezierSpline, bet: BezierSpline, t=np.array([0, 1 / 2, 1])):
        """ Evaluate distance between two Bézier curves in M at several points

        :param alp: Bézier curve
        :param bet: Bézier curve
        :param t: vector with elements in [0,1]

        :return: vector with point-wise distances
        """
        a_val = alp.eval(t)
        b_val = bet.eval(t)
        d_M = []
        for i in range(len(t)):
            d_M.append(self._M.metric.dist(a_val[i], b_val[i]))
        return np.array(d_M), t

    def disc_path_energy(self, H):
        """Discrete path energy

        :param H: discrete path given as ordered list of Bézier curves of the same degree
        :return: energy of H
        """
        # test ¨regression-conform¨ distance
        t = np.linspace(0, 1, num=self._degree + 1)
        d = 0
        for i in range(len(H) - 1):
            dh, _ = self.loc_dist(H[i], H[i + 1], t)
            d += np.sum(dh**2, axis=0)

        return d

    def rand(self):
        # TODO
        return

    def randvec(self, X):
        # TODO
        return

    def zerovec(self):
        # TODO
        return

    def transp(self, R, Q, X):
        # TODO
        return

    def pairmean(self, alp, bet):
        # TODO
        return

    def dist(self, alp, bet, l=5):
        """Approximate the distance between two Bézier splines

        :param alp: Bézier spline
        :param bet: Bézier spline
        :param l: work with discrete l-geodesic
        :return: length of l-geodesic between alp and bet (approximation of the distance)
        """

        Gam = self.discgeodesic(alp, bet, n=l)

        d = 0
        for i in range(len(Gam) - 1):
            y, t = self.loc_dist(Gam[i], Gam[i + 1])
            d += integrate.simps(y, t)

        return d

    def mean(self, B, n=3, delta=1e-5, min_stepsize=1e-10, nsteps=20, eps=1e-5, n_stepsGeo=10, verbosity=1):
        """Discrete mean of a set of Bézier curves

        :param B: list of Bézier curves
        :param n: use discrete n-geodesics
        :param delta: iteration stops when the difference in energy between the new and old iterate drops below eps
        :param min_stepsize: iteration stops when the step length is smaller than the given value
        :param nsteps: maximal number of iterations
        :param eps: see eps in discgeodesic
        :param n_stepsGeo: maximal number of iterations when computating discrete geodesics
        :param verbosity: 0 (no text) or 1 (print information on convergence)
        :return: mean curve
        """
        begin_mean = time.time()

        # get shape of array of control points
        m = np.array(B[0].control_points[0].shape)
        m[0] = self._degree + 1

        def legs(meaniterate):
            """ Construct legs of polygonal spider, i.e., discrete n-geodesics between mean iterate and input curves
            """
            return Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(delayed(self.discgeodesic)
                                                                              (meaniterate, b, n=n, eps=eps,
                                                                               nsteps=n_stepsGeo, verbosity=0)
                                                                              for b in B)

        def loss(FF):
            G = 0
            for HH in FF:
                G += self.disc_path_energy(HH)
            return G

        # initialize i-th control point of the mean as the mean of the i-th control points of the data
        C = ExponentialBarycenter
        P = np.zeros(m)
        for i in range(self._degree + 1):
            D = []
            for bet in B:
                D.append(bet.control_points[0][i])

            P[i] = C.compute(self._M, D)

        # initial guess
        bet_mean = B[0]

        # initial legs
        F = legs(bet_mean)
        # initialize stopping parameters
        Eold = 10
        Enew = 1
        stepsize = 10
        step = 0
        while np.abs(Enew - Eold) > delta and stepsize > min_stepsize and step <= nsteps:
            step += 1
            Eold = Enew
            F_old = F
            old_mean = BezierSpline(self._M, bet_mean.control_points)

            # new data for regression
            t = np.linspace(0, 1, num=self._degree + 1)
            Y = []

            for H in F:
                Y.append(H[1].eval(t))

            # make regression w.r.t. mean values -> faster
            mean_Y = np.zeros_like(Y[0])
            for i in range(len(mean_Y)):
                dat = []
                for j in range(len(Y)):
                    # take value of each curve at time t_i
                    dat.append(Y[j][i])

                mean_Y[i] = C.compute(self._M, dat)

            if verbosity == 2:
                print('Step '+str(step)+': Updating the mean...')

            regression = RiemannianRegression(self._M, mean_Y, t, self._degree, verbosity=2)
            bet_mean = regression.trend

            # update discrete paths
            if verbosity == 2:
                print('Step '+str(step)+': Updating discrete paths...')
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
                step_size += self._M.metric.dist(p, old_mean.control_points[0][i]) ** 2
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
                G[i, j] = n ** 2 / (2 * n) * (self.dist(B_mean, si[1], l=1) ** 2 + self.dist(B_mean, sj[1], l=1) ** 2
                                              - self.dist(si[1], sj[1], l=1) ** 2)
                G[j, i] = G[i, j]

        return G, B_mean, F

    def projToGeodesic(self, R, Q, P, max_iter=10):
        # TODO
        return
