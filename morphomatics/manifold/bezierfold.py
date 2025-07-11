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

from __future__ import annotations
from functools import partial

from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.geom.bezier_spline import BezierSpline, full_set, indep_set
from morphomatics.manifold import Manifold, Metric, PowerManifold
from morphomatics.opt import RiemannianSteepestDescent, RiemannianNewtonRaphson
from morphomatics.stats import ExponentialBarycenter


class Bezierfold(Manifold):
    """Manifold of Bézier splines (of fixed degrees)

    """

    def __init__(self, M: Manifold, n_segments: int, degree: int, isscycle: bool=False,
                 n_steps: int=10, n_samples: int=None, structure='FunctionalBased'):
        """Manifold of Bézier splines of constant segment degrees

        :arg M: base manifold in which the curves lie
        :arg n_segments: number of spline segments
        :arg degree: degree of segment (same for each one)
        :arg iscycle: boolean indicating whether the splines are closed
        :arg n_steps: number of steps (i.e. segments) for approximation of geodesics in Bezierfold
        :arg n_samples: number of samples for quadrature of curve distance in L²(I, M)
        :arg structure: type of geometric structure
        """

        self._M = M
        self._degrees = np.full(n_segments, degree)
        self._nsteps = n_steps

        if isscycle:
            name = 'Manifold of closed Bézier splines of degree {d} through '.format(d=degree) + str(M)
            K = np.sum(self._degrees - 1) - 1
        else:
            name = 'Manifold of non-closed Bézier splines of degrees {d} through '.format(d=degree) + str(M)
            K = np.sum(self._degrees - 1) + 1

        self._nsamples = n_samples if n_samples else K+1
        assert self._nsamples > K

        dimension = (K + 1) * M.dim
        point_shape = (K+1, *M.point_shape)
        self._K = K
        super().__init__(name, dimension, point_shape)

        self._iscycle = isscycle

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        aux += (self.nsegments, self.degrees[0], self.iscycle, self.nsteps, self.nsamples)
        return children + (self.M,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, M = children
        *aux_data, n_seg, d, c, n_st, n_sam = aux_data
        obj = cls(M, n_seg, d, c, n_st, n_sam, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initFunctionalBasedStructure(self):
        """
        Instantiate the functional-based structure with discrete methods.
        """
        structure = Bezierfold.FunctionalBasedStructure(self)
        self._metric = structure
        self._connec = structure

    @property
    def M(self) -> Manifold:
        """Return the underlying manifold
        """
        return self._M

    @property
    def degrees(self) -> np.array:
        """Return vector of segment degrees
        """
        return self._degrees

    @property
    def nsegments(self) -> int:
        """Returns the number of spline segments."""
        return len(self._degrees)

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

    @property
    def nsamples(self):
        """Returns the number of samples for quadrature of curve distance in L²(I, M)."""
        return self._nsamples

    @property
    def nsteps(self):
        """Returns the number of steps (i.e. segments) for approximation of geodesics in Bezierfold"""
        return self._nsteps

    def correct_type(self, B: BezierSpline) -> bool:
        """Check whether B has the right segment degrees"""
        if jnp.all(jnp.atleast_1d(B.degrees) == self.degrees):
            return True
        else:
            return False

    def rand(self, key: jax.Array) -> BezierSpline:
        """Return random Bézier spline"""
        subkeys = jax.random.split(key, self.K + 1)
        return BezierSpline(self.M, full_set(self.M, jax.vmap(self.M.rand)(subkeys),
                                             self.degrees, self.iscycle))

    def randvec(self, B: BezierSpline, key: jax.Array) -> jnp.array:
        """Return random vector for every independent control point"""
        pts = indep_set(B, self.iscycle)
        subkeys = jax.random.split(key, len(pts))
        return jax.vmap(self.M.randvec)(pts, subkeys)

    def zerovec(self) -> jnp.array:
        """Return zero vector for every independent control point"""
        return jnp.tile(self.M.zerovec(), (self.K + 1,) + (1,)*len(self.M.point_shape))

    def to_coords(self, B: BezierSpline) -> jnp.array:
        """
        :param B: Bézier spline
        :return: Array of independent control points.
        """
        return indep_set(B.control_points, self.iscycle)

    def from_coords(self, pts: jnp.array) -> BezierSpline:
        """
        :param pts: independent control points
        :return: Bézier spline
        """
        pts = full_set(self.M, pts, self.degrees, self.iscycle)
        return BezierSpline(self.M, pts, self.iscycle)

    def proj(self, X, H):
        return H

    ############################## Functional-based structure ##############################
    class FunctionalBasedStructure(Metric):
        """
        Functional-based metric structure
        """

        def __init__(self, Bf: Bezierfold):
            """
            Constructor.
            """
            self._Bf = Bf

        @property
        def __str__(self):
            return "Bézierfold-functional-based structure"

        def inner(self, p: jnp.array, X: jnp.array, Y: jnp.array):
            """Functional-based metric, i.e. L²(I, TBM).

            :arg p: Bézier spline in M
            :arg X: tangent vector at p
            :arg Y: tangent vector at p
            :return: inner product of X and Y at p
            """

            M, deg, cyclic = self._Bf.M, self._Bf.degrees, self._Bf.iscycle

            def full(q, V):
                f = lambda pts: jnp.array(full_set(M, pts, deg, cyclic))
                # fwd-diff. of full_set
                q_full, V_full = jax.jvp(f, (q,), (V,))
                # proj. to tangent space
                vproj = jax.vmap(jax.vmap(M.proj))
                return q_full, vproj(q_full, V_full)

            # map p, X, Y to all control points
            p_full, X_full = full(p, X)
            _, Y_full = full(p, Y)

            # sample spline and generalized Jacobi fields for X, Y
            t = jnp.linspace(0., self._Bf.nsegments, self._Bf.nsamples)
            spln = BezierSpline(M, p_full, cyclic)
            vDpB = jax.vmap(spln.DpB, (0, None))
            B, Jx = vDpB(t, X_full)
            _, Jy = vDpB(t, Y_full)

            # eval inner products
            return jax.vmap(self._Bf.M.metric.inner)(B, Jx, Jy).sum()

        @property
        def typicaldist(self) -> float:
            # approximations via control points
            return self._Bf.K * self._Bf.M.metric.typicaldist

        def dist(self, a: jnp.array, b: jnp.array) -> float:
            """Approximate the distance between two Bézier splines

            :param a: independent control points of a Bézier spline
            :param b: independent control points of a Bézier spline
            :return: length of n-geodesic between A and B (approximation of the distance)
            """
            return jnp.sqrt(self.squared_dist(a, b))

        def squared_dist_extrinsic(self, p, q):
            t = jnp.linspace(0., self._Bf.nsegments, self._Bf.nsamples)
            d2 = jax.vmap(self._Bf.M.metric.squared_dist)
            return d2(sample(self._Bf, p, t), sample(self._Bf, q, t)).sum()

        def squared_dist(self, p, q):
            n = self._Bf.nsteps
            gamma = self.discgeodesic(self._Bf, p, q, n=n)
            return jax.vmap(self.squared_dist_extrinsic)(gamma[:-1], gamma[1:]).sum() * n

        @staticmethod
        @jax.jit
        def discexp(Bf, a: jnp.array, b: jnp.array):
            """
            Compute c such that [a,b,c] is a discrete 2-geodesic.
            :param Bf: Bezierfold a ang b live in
            :param a: Bézier spline in manifold M (i.e. independent control points thereof)
            :param b: Bézier spline in manifold M (i.e. independent control points thereof)
            :return: c
            """

            t = jnp.linspace(0., Bf.nsegments, Bf.nsamples)

            # initial guess for c
            c = jax.vmap(Bf.M.connec.geopoint, (0, 0, None))(a, b, 2.)

            # gradient of sum-of-squared-distances between samples along alpha and beta w.r.t. ctrl. pts. of alpha
            def G(alpha, beta):
                egrad = jax.grad(lambda x: jax.vmap(Bf.M.metric.squared_dist)(sample(Bf, x, t), sample(Bf, beta, t)).sum())
                return jax.vmap(Bf.M.metric.egrad2rgrad)(alpha, egrad(alpha))

            # gradient for b w.r.t. a
            G_a = G(b, a)

            # discrete Euler-Lagrange cnd. of path energy for [a,b,c]
            def F(x):
                return G(b, x) + G_a

            # solve F(x) = 0
            N = PowerManifold(Bf.M, Bf.K+1)
            return RiemannianNewtonRaphson.solve(N, F, c, stepsize=.1, maxiter=min(Bf.dim, 1000))

        def exp(self, p: jnp.array, X: jnp.array) -> jnp.array:
            n = self._Bf.nsteps

            def body(carry, _):
                a, b = carry
                # compute c s.t. [a,b,c] is discrete 2-geodesic
                c = self.discexp(self._Bf, a, b)
                return (b, c), None

            q = jax.vmap(self._Bf.M.connec.exp)(p, X/n)
            (_, q), _ = jax.lax.scan(body, (p, q), jnp.empty(n))

            return q

        def log(self, p: jnp.array, q: jnp.array) -> jnp.array:
            n = self._Bf.nsteps
            gamma = self.discgeodesic(self._Bf, p, q, n=n)
            return jax.vmap(self._Bf.M.connec.log)(p, gamma[1]) * n

        @staticmethod
        @partial(jax.jit, static_argnames=['n'])
        def discgeodesic(Bf: Bezierfold, p: jnp.array, q: jnp.array, n: int = 5, maxiter: int = 100, minchange: float = 1e-6) -> jnp.array:
            """Discrete shortest path through space of Bézier splines.

            :param Bf: Bezierfold p and q live in
            :param p: Bézier spline in manifold M (i.e. independent control points thereof)
            :param q: Bézier spline in manifold M (i.e. independent control points thereof)
            :param n: create discrete n-geodesic
            :param maxiter: max. number of iterations
            :param minchange: min. change in coordinates to declare convergence
            :return: control points of the Bézier splines along the shortest path
            """

            # Initialize inner splines of path

            # logs between corresponding control points of A and B (save repeated computations)
            X = jax.vmap(Bf.M.connec.log)(p, q)
            # exps
            t_exp = lambda t: jax.vmap(Bf.M.connec.exp)(p, t * X)
            H = jax.vmap(t_exp)(jnp.linspace(0., 1., n + 1)[1:-1])
            # add start-/endpt.
            H = jnp.concatenate((jnp.expand_dims(p, axis=0), H, jnp.expand_dims(q, axis=0)))

            # Discrete path shortening flow
            def body(args):
                x, _, i = args
                x, d = curve_shortening_step(Bf, x)
                # jax.debug.print("{}: {}", i, d)
                return x, d, i + 1

            # check convergence
            def cond(args):
                _, d, i = args
                c = jnp.array([d > minchange, i < maxiter])
                return jnp.all(c)

            H, *_ = jax.lax.while_loop(cond, body, (H, jnp.array(1.), jnp.array(0)))

            return H

        @staticmethod
        #@partial(jax.jit, static_argnames=['Bf'])
        def mean(Bf, B, maxiter: int = 500, minchange: float = 1e-5):
            """Discrete mean of a set of Bézier splines

            :param Bf: Bezierfold
            :param B: array of splines (i.e. independent control points thereof)
            :param maxiter: max. number of iterations
            :param minchange: min. change in coordinates to declare convergence
            :return: (independent control points of) mean curve
            """
            # times at which to sample splines
            t = jnp.linspace(0, Bf.nsegments, Bf.nsamples)

            # setup 'regression' problem for mean (where there are len(B) targets for each time pt.)

            # search space: k-fold product of M
            N = PowerManifold(Bf.M, Bf.K+1)

            # sum-of-squared-distances
            def ssd(pts, Y, param):
                x = sample(Bf, pts, param)
                d = jax.vmap(jax.vmap(Bf.M.metric.squared_dist), (None, 0))(x, Y)
                return jnp.sum(d) / np.prod(Y.shape[:2])

            # compute mean spline

            # initialize i-th control point of the mean as the mean of the i-th control points of the data
            mean = lambda b: ExponentialBarycenter.compute(Bf.M, b)
            init = jax.vmap(mean, 1)(B)

            # init legs, i.e. n-geodesics between mean and input curves B
            discgeodesic = Bezierfold.FunctionalBasedStructure.discgeodesic
            F_init = jax.vmap(discgeodesic, (None, None, 0, None))(Bf, init, B, Bf.nsteps)

            def body(args):
                x, F, change, i = args

                # update x via regression
                Y = jax.vmap(sample, (None, 0, None))(Bf, F[:, 1], t)
                opt = RiemannianSteepestDescent.fixedpoint(N, lambda a: ssd(a, Y, t), x)
                change = jnp.abs(opt - x).max()
                #change = jnp.linalg.norm((opt - x).ravel(), np.inf)

                # update legs of 'polygonal spider'
                F = F.at[:, 0].set(opt)
                F, d = jax.vmap(curve_shortening_step, (None, 0))(Bf, F)
                change = jnp.array([change, jnp.abs(d).max()]).max()
                #change = jnp.array([change, jnp.linalg.norm(d.ravel(), np.inf)]).max()

                jax.debug.print("{}: {}", i, change)
                return opt, F, change, i + 1

            def cond(args):
                _, _, change, i = args
                c = jnp.array([change > minchange, i < maxiter])
                return jnp.all(c)

            mu, F_mu, *_ = jax.lax.while_loop(cond, body, (init, F_init, 1., 0))

            return mu, F_mu

        def gram(self, B_mean: jnp.array, F: jnp.array):
            """Approximates the Gram matrix for a curve data set.

            :param B_mean: mean of curves in B (as returned by #mean)
            :param F: discrete spider, i.e, discrete paths from mean to data (as returned by #mean)
            :return G: Gram matrix
            """
            n = len(F)
            G = jnp.zeros((n, n))
            for i, si in enumerate(F):
                for j, sj in enumerate(F[i:], start=i):
                    G = G.at[i, j].set(n / 2 * (
                            self.squared_dist_extrinsic(B_mean, si[1])
                            + self.squared_dist_extrinsic(B_mean, sj[1])
                            - self.squared_dist_extrinsic(si[1], sj[1]))
                                       )
                    G = G.at[j, i].set(G[i, j])

            return G

        def egrad2rgrad(self, p: jnp.array, X: jnp.array) -> jnp.array:
            """
            :param p: Bézier spline in manifold M (i.e. independent control points thereof)
            :param X: tangent vector (i.e. tangent vectors at the independent control points)
            """
            return jax.vmap(self._Bf.M.metric.egrad2rgrad)(p, X)

        def retr(self, R, X):
            return self.exp(R, X)

        ### not imlemented ###

        def curvature_tensor(self, p, X, Y, Z):
            raise NotImplementedError('This function has not been implemented yet.')

        def transp(self, R, Q, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def jacobiField(self, R, Q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, R, Q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def flat(self, p, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, p, dX):
            raise NotImplementedError('This function has not been implemented yet.')


def sample(Bf: Bezierfold, pts: jnp.array, t: jnp.array) -> jnp.array:
    # vectorized methods for sampling of splines (from independent ctrl. pts.)
    return jax.vmap(lambda p, s: Bf.from_coords(p).eval(s), (None, 0))(pts, t)


def curve_shortening_step(Bf: Bezierfold, x: jnp.array) -> Tuple[jnp.array, float]:
    """Single step of discrete curve shortening flow: Replace inner node with
     average of its neighbors (s.t. it's the midpoint of the connecting 2-geodesic).

    :param Bf: Bezierfold
    :param x: Discrete path in Bf (i.e. independent control points of nodes)
    :return: updated nodes, inf-norm of update
    """
    # local import to avoid cyclic dependencies
    from morphomatics.stats.riemannian_regression import RiemannianRegression

    deg = Bf.degrees[0]
    nseg = Bf.nsegments

    t = jnp.linspace(0., nseg, Bf.nsamples)
    tt = jnp.concatenate([t, t])

    def body(carry, cur_post):
        pre, d = carry
        cur, post = cur_post
        # sample pre & post
        pre = sample(Bf, pre, t)
        post = sample(Bf, post, t)
        # update (fit cur to pre & post)
        Y = jnp.concatenate([pre, post])
        opt = RiemannianRegression.fit(Bf.M, Y, tt, cur, deg, nseg, maxiter=1, iscycle=Bf.iscycle)
        # update inf-norm
        d = jnp.array([d, jnp.abs(opt - cur).max()]).max()
        #d = jnp.array([d, jnp.linalg.norm(jnp.ravel(opt - cur), ord=jnp.inf)]).max()
        return (opt, d), opt

    # stack each node with its successor
    stacked = jnp.stack([x[1:-1], x[2:]], axis=1)

    # update nodes one-by-one
    (_, d), inner_nodes = jax.lax.scan(body, (x[0], 0.), stacked)

    return jnp.concatenate((x[None, 0], inner_nodes, x[None, -1])), d
