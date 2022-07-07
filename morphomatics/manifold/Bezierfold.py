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

from morphomatics.manifold import Manifold
from morphomatics.stats.RiemannianRegression import full_set, indep_set, sumOfSquared
from morphomatics.manifold.ManoptWrap import ManoptWrap

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product
from pymanopt.optimizers import SteepestDescent

import time

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
        :arg degrees: array of degrees of the Bézier segments – every segment must have the same degree
        """

        self._M = M

        self._degrees = jnp.atleast_1d(degrees)

        if isscycle:
            name = 'Manifold of closed Bézier splines of degrees {d} through '.format(d=degrees) + str(M)
            K = jnp.sum(self._degrees - 1) - 1
        else:
            name = 'Manifold of non-closed Bézier splines of degrees {d} through '.format(d=degrees) + str(M)
            K = jnp.sum(self._degrees - 1) + 1

        dimension = (K + 1) * M.dim
        point_shape = [K, M.point_shape]
        self._K = K
        super().__init__(name, dimension, point_shape)

        self._iscycle = isscycle

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
        """Return the generalized degree if a Bezier spline, i.e., the number of independent control points - 1
        """
        return self._K

    @property
    def iscycle(self):
        """Return whether the Bezierfold consists of non-closed or closed splines
        """
        return self._iscycle

    def correct_type(self, B: BezierSpline):
        """Check whether B has the right segment degrees"""
        if jnp.all(jnp.atleast_1d(B.degrees) != self.degrees):
            return False
        else:
            return True

    def rand(self):
        """Return random Bézier spline"""
        return BezierSpline(self.M, full_set(self.M, jnp.array([self.M.rand() for k in self.K+1]),
                                             self.degrees, self.iscycle))

    def randvec(self, B: BezierSpline):
        """Return random vector for every independent control point"""
        return jnp.array([self.M.randvec(p) for p in indep_set(B, self.iscycle)])

    def zerovec(self):
        """Return zero vector for every independent control point"""
        return jnp.array([self.M.zerovec() for k in self.K+1])

    def indep_cp_path(self, P):
        """Returns concatenated sequence of independent control points of a discrete path
        """
        return jnp.concatenate(jnp.asarray([indep_set(p, self.iscycle) for p in P]))

    def full_cp_path(self, P):
        """Returns array of full control points of splines of a discrete path. The leading dimension enumerates the
        splines.
        """
        # # compute discretization parameter
        # if w_boundary:
        #     n = int(len(P) / (self.K+1)) - 1
        # else:
        #     n = int(len(P) / (self.K+1)) + 1
        #
        # P = jnp.asarray(P)
        # # reshape so that the splines are ordered along first axis
        # if w_boundary:
        #     P = P.reshape(n + 1, -1, *[dim for dim in self.M.point_shape])
        # else:
        #     P = P.reshape(n - 1, -1, *[dim for dim in self.M.point_shape])

        # compute number of splines
        l = int(len(P) / (self.K + 1))
        # reshape so that the splines are ordered along first axis
        P = P.reshape(l, self.K + 1, *[dim for dim in self.M.point_shape])

        # full set of control points
        P_ = []
        for p in P:
            P_.append(jnp.asarray(full_set(self.M, p, self.degrees, self.iscycle)))

        return jnp.asarray(P_)

    def grid(self, PP, t):
        """Evaluate discrete path (i.e., set of splines) at common parameters
        Parameters
        ----------
        PP: [k,..]-array of control points of k Bézier splines
        t: 1-D array of parameters

        Returns array of values of the splines corresponding to PP at times given in t
        -------

        """
        G = []
        for p in PP:
            G.append(jax.vmap(lambda time: BezierSpline(self.M, p).eval(time))(t))
        return jnp.asarray(G)

    class FunctionalBasedStructure(Metric, Connection):
        """
        Functional-based metric structure
        """

        def __init__(self, Bf):
            """
            Constructor.
            """
            self._Bf = Bf

        @property
        def __str__(self):
            return "Bezierfold-functional-based structure"

        def inner(self, B: BezierSpline, X, Y):
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
            return jnp.sqrt(self.inner(B, X, X))

        @property
        def typicaldist(self):
            # approximations via control points
            return self._Bf.K * self._Bf.M.metric.typicaldist

        def dist(self, A, B, n=5):
            """Approximate the distance between two Bézier splines

            :param A: Bézier spline
            :param B: Bézier spline
            :param n: work with discrete n-geodesic
            :return: length of n-geodesic between alp and bet (approximation of the distance)
            """

            Gam = self.discgeodesic(A, B, n=n)

            d = 0
            for i in range(len(Gam) - 1):
                y, t = self.loc_dist(Gam[i], Gam[i + 1])
                d = d + jnp.trapz(y, x=t)

            return d

        def discgeodesic(self, A, B, n=5, maxtime=1000, maxiter=30, mingradnorm=1e-6, minstepsize=1e-10,
                         maxcostevals=5000, verbosity=0):
            """Discrete shortest path through space of Bézier curves

            :param A: Bézier spline in manifold M or control points thereof
            :param B: Bézier spline in manifold M or control points thereof
            :param n: create discrete n-geodesic
            :param maxtime: see Pymanopt's solver class
            :param maxiter: see Pymanopt's solver class
            :param mingradnorm: see Pymanopt's solver class
            :param minstepsize: see Pymanopt's solver class
            :param maxcostevals: see Pymanopt's solver class
            :param logverbosity: see Pymanopt's solver class (0 to 2)
            :return: control points of the Bézier splines along the shortest path
            """

            PA = A.control_points if isinstance(A, BezierSpline) else A
            PB = B.control_points if isinstance(B, BezierSpline) else B

            def init_disc_curve():
                """Initialize discrete curve by aligning control points along geodesics
                :return: discrete curve as array of independent control points of Bézier splines (without A, B)
                """
                # only take independent control points
                pa = indep_set(PA, iscycle=self._Bf.iscycle)
                pb = indep_set(PB, iscycle=self._Bf.iscycle)

                # logs between corresponding control points of A and B (save repeated computations)
                X = jnp.zeros(pa.shape)
                for j in range(len(pa)):
                    X = X.at[j].set(self._Bf.M.connec.log(pa[j], pb[j]))

                H = []
                # initialize control points along geodesics
                for i in range(1, n):
                    P_i = jnp.zeros(pa.shape)
                    for j in range(len(pa)):
                        P_i = P_i.at[j].set(self._Bf.M.connec.exp(pa[j], i / n * X[j]))

                    # align all control points of splines along leading axis
                    H.append(jnp.asarray(P_i))

                return jnp.concatenate(H)

            # initialize inner splines of path
            H0 = init_disc_curve()

            # Solve optimization problem with pymanopt by optimizing over independent control points
            pymanoptM = ManoptWrap(self._Bf.M)
            # there are n-1 inner splines
            N = Product([pymanoptM] * (n-1) * (self._Bf.K + 1))

            @jax.jit
            def cost_(a, b):
                return jnp.sum(jax.vmap(self._Bf.M.metric.squared_dist, in_axes=(0, 0), out_axes=0)(a, b))

            @pymanopt.function.jax(N)
            def cost(*P):
                # TODO: I think we could jit the whole thing if we make the discretization parameter explicit here -> could be misused
                Pfull = jnp.concatenate((jnp.expand_dims(PA, axis=0), self._Bf.full_cp_path(jnp.asarray(P)),
                                         jnp.expand_dims(PB, axis=0)))

                # sample splines at as many discrete points as we have free control points
                t = jnp.linspace(0, self._Bf.nsegments, num=int(np.sum(self._Bf.degrees + 1)))

                H = self._Bf.grid(Pfull, t)

                c = 0
                for i in range(len(H)):
                    c = c + cost_(H[i], H[i+1])

                return c

            problem = Problem(manifold=N, cost=cost)

            solver = SteepestDescent(max_time=maxtime, max_iterations=maxiter, min_gradient_norm=mingradnorm,
                                     min_step_size=minstepsize, max_cost_evaluations=maxcostevals,
                                     log_verbosity=verbosity)

            opt = solver.run(problem, initial_point=H0)

            H_opt = self._Bf.full_cp_path(jnp.array(opt.point))

            return jnp.concatenate((jnp.expand_dims(PA, axis=0), H_opt, jnp.expand_dims(PB, axis=0)))

        def loc_dist(self, P, Q, t=None):
            """ Evaluate distance between two Bézier splines in M at several points

            :param P: array with control points of a Bézier spline
            :param Q: array with control points of Bézier spline
            :param t: vector with time points
            The control points have the size [k, l,...] where k is the number of segments and l-1 the degree.

            :return: vector with point-wise distances
            """
            A = BezierSpline(self._Bf.M, P, self._Bf.iscycle)
            B = BezierSpline(self._Bf.M, Q, self._Bf.iscycle)

            if t is None:
                # sample segments of higher degree more densely
                t = jnp.linspace(0, 1, num=P.shape[1])
                for i in range(1, A.nsegments):
                    t = jnp.concatenate((t, jnp.linspace(i, i + 1, num=P.shape[1])[1:]))

            a_val = [A.eval(time) for time in t]
            b_val = [B.eval(time) for time in t]
            d_M = jnp.zeros(len(t))
            for i in range(len(t)):
                d_M = d_M.at[i].set(self._Bf.M.metric.dist(a_val[i], b_val[i]))
            return d_M, t

        def disc_path_energy(self, H):
            """Discrete path energy

            :param H: discrete path given as ordered list of Bézier splines of the same degrees
            :return: energy of H
            """
            d = 0
            for i in range(len(H) - 1):
                # dh, _ = self.loc_dist(H[i], H[i + 1])
                # d = d + jnp.sum(dh ** 2, axis=0)
                y, t = self.loc_dist(H[i], H[i + 1])
                d = d + jnp.trapz(y, x=t)

            return d

        def mean(self, B, n=3, delta=1e-5, min_stepsize=1e-10, nsteps=20, eps=1e-5, n_stepsGeo=10, verbosity=1):
            """Discrete mean of a set of Bézier splines

            :param B: list of Bézier splines, every curve can have arbitrary many segments which must have the same degrees
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

            def legs(meaniterate):
                """ Construct legs of polygonal spider, i.e., discrete n-geodesics between mean iterate and input curves
                """
                # TODO: make parallel
                # def fun(b):
                #     return self.discgeodesic(meaniterate, b, n=n)

                F = []
                for b_ in B:
                    F.append(self.discgeodesic(meaniterate, b_, n=n))

                return jnp.array(F)
                #return jax.vmap(fun)(B)

            @jax.jit
            def loss(F):
                """Loss in optimization for mean
                """
                # G = 0
                # for HH in FF:
                #     G = G + self.disc_path_energy(HH)
                # return G
                return jnp.sum(jax.vmap(self.disc_path_energy)(F))

            # measure computation time
            begin_mean = time.time()

            B = extract_control_points(B)

            # initialize i-th control point of the mean as the mean of the i-th control points of the data
            C = ExponentialBarycenter
            P = []

            # loop through segments
            m = jnp.array(B[0, 0].shape)
            for l in range(self._Bf.nsegments):
                P_l = jnp.zeros(m)
                for i in range(self._Bf.degrees[0] + 1):
                    D = []
                    # take the i-th control point of the l-th segment from all data curves
                    for bet in B:
                        D.append(bet[l, i])

                    # initialize i-th control point of the l-th segment as mean of corresponding data control points
                    P_l = P_l.at[i].set(C.compute(self._Bf.M, jnp.array(D)))

                P.append(P_l)
            P_mean_iterate = jnp.array(P)

            # initial legs
            F = legs(P_mean_iterate)
            bet_mean = BezierSpline(self._Bf.M, P_mean_iterate, self._Bf.iscycle)
            # initialize stopping parameters
            Eold = 10
            Enew = 1
            stepsize = 10
            step = 0
            while jnp.abs(Enew - Eold) > delta and stepsize > min_stepsize and step <= nsteps:
                # one more step
                step += 1
                # save old "stopping parameters"
                Eold = Enew
                F_old = F
                old_mean = BezierSpline(self._Bf.M, bet_mean.control_points)

                # new data for regression
                # sample splines at so many discrete points as we have free control points
                # t = jnp.linspace(0, self._Bf.nsegments, num=int(np.sum(self._Bf.degrees + 1)))
                t = jnp.linspace(0, self._Bf.nsegments, num=self._Bf.K + 1)
                Y = []

                for H in F:
                    # evaluate first "joint" at t
                    bet = BezierSpline(self._Bf.M, H[1])
                    Y.append(jnp.array([bet.eval(time) for time in t]))

                # make regression w.r.t. mean values -> faster
                mean_Y = jnp.zeros_like(Y[0])
                for i in range(len(mean_Y)):
                    dat = []
                    for j in range(len(Y)):
                        # take value of each curve at time t_i
                        dat.append(Y[j][i])
                    # mean at t_i if first joints
                    mean_Y = mean_Y.at[i].set(C.compute(self._Bf.M, jnp.array(dat)))

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
                for l, P in enumerate(bet_mean.control_points):
                    for j, p in enumerate(P):
                        step_size = step_size + self._Bf.M.metric.dist(p, old_mean.control_points[l][j]) ** 2
                stepsize = jnp.sqrt(step_size)

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
            G = jnp.zeros((n, n))
            for i, si in enumerate(F):
                for j, sj in enumerate(F[i:], start=i):
                    G = G.at[i, j].set(n / 2 * (
                            self.dist(B_mean, si[1], n=1) ** 2 + self.dist(B_mean, sj[1], n=1) ** 2
                            - self.dist(si[1], sj[1], n=1) ** 2))
                    G = G.at[j, i].set(G[i, j])

            return G, B_mean, F

        ### not imlemented ###

        def geopoint(self, alp, bet, t):
            return

        def pairmean(self, alp, bet):
            return

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
            return

        def log(self, R, Q):
            return

        def transp(self, R, Q, X):
            return

        def jacobiField(self, R, Q, t, X):
            return

        def adjJacobi(self, R, Q, t, X):
            return


def extract_control_points(B):
    """ Extract array of control points from Bézier splines

    Parameters
    ----------
    B set of Bézier splines of same type

    Returns array of corresponding control points
    -------

    """
    P = []
    for b in B:
        P.append(b.control_points)

    return jnp.array(P)
