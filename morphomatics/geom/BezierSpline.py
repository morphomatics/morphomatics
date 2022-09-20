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
import jax.numpy as jnp
import jax.lax as lax


class BezierSpline:
    """Manifold-valued spline that consists of Bézier curves"""

    def __init__(self, M, control_points, iscycle=False):
        """
        :arg M: manifold in which the curve lies
        :arg control_points: array of control points of the Bézier spline, the L >= 1 segments must be sorted along the
        first axis and all segments must have the same degree k; that is, the input must be an [L, k, M.point_shape] array
        entry of the list belongs to the l-th segment of the spline.
        :arg iscycle: boolean indicating whether B is a closed curve
        """
        assert M is not None

        self._M = M

        self.control_points = jnp.array(control_points)

        self.iscycle = iscycle

    def __str__(self):
        return 'Bézier spline through ' + str(self._M)

    @property
    def nsegments(self):
        """Returns the number of segments."""
        return len(self.control_points)

    @property
    def degrees(self):
        """Returns the degrees of the spline segments."""
        L = len(self.control_points)
        n_seg = np.zeros(L, dtype=int)
        for i in range(L):
            n_seg[i] = np.shape(self.control_points[i])[0] - 1
        return n_seg

    def length(self):
        # TODO
        return

    def energy(self):
        # TODO
        return

    def tangent(self, t):
        """
        Compute the tangent vector at the point of the spline corresponding to t.
        """

        def bezier_tangent(bet:BezierSpline, s):
            """
            Compute the tangent vector at the point of a (single) Bèzier curve corresponding to t in [0, 1].
            """

            def single_layer(A, r, X=None):
                """
                Single layer of the computation consisting of a single step of the de Casteljau algorithm
                plus additinal vectors transport/computation.
                """
                if X is None:
                    # averaging of a single layer in de Casteljau algorithm
                    size = np.array(np.shape(A))
                    # give back one point less
                    size[0] = size[0] - 1
                    B = np.zeros(size)
                    for i in range(size[0]):
                        B[i] = self._M.exp(A[i], self._M.log(A[i], A[i + 1]) * r)
                    return B

                else:
                    # averaging of a single layer in de Casteljau algorithm
                    size = np.array(np.shape(A))
                    # give back one point less
                    size[0] = size[0] - 1
                    B = np.zeros(size)
                    for i in range(size[0]):
                        B[i] = self._M.exp(A[i], self._M.log(A[i], A[i + 1]) * r)

                    # calculate updates of tangent vectors
                    X_shape = X.shape
                    X_shape[0] -= 1
                    Y = np.zeros(X_shape)
                    for i in range(len(Y)):
                        # new point is on geodesic between old control points -> log to endpoint shortened tangent vector
                        v = self._M.connec.log(B[ii], A[ii+1])
                        # rescale
                        v = v / self._M.metric.norm(B[ii], v) * self._M.metric.dist(bet.control_points[ii],
                                                                                    bet.control_points[ii + 1])
                        Y[i] += v
                        # add transported old vectors
                        # X[i] 'forward' X[i+1] 'backward'
                        Y[i] += self._M.connec.DxGeo(A[i], A[i+1], r, X[i])
                        Y[i] += self._M.connec.DyGeo(A[i], A[i + 1], r, X[i+1])

                    return B, Y

            k = bet.degrees[0]
            if s == 0:
                return bet.eval(0), k * self._M.connec.log(bet.control_points[0][0], bet.control_points[0][1])
            elif s == 1:
                return bet.eval(1), -k * self._M.connec.log(bet.control_points[0][-1], bet.control_points[0][-2])
            else:
                P_old = bet.control_points[0]
                P = single_layer(P_old, s)

                X = np.zeros(k, self._M.zerovec().shape)
                for ii in range(len(P)):
                    # new point is on geodesic between old control points -> log to endpoint shortened tangent vector
                    v = self._M.connec.log(P[ii], P_old[ii+1])
                    # rescale
                    X[ii] = v / self._M.connec.norm(P[ii], v) * self._M.metric.dist(P_old[ii], P_old[ii + 1])

                # there are k+1 control points
                for l in range(k):
                    P, X = single_layer(P, s, X)

                return P, X

        # get segment and local parameter
        ind, t = segmentize(t)

        return bezier_tangent(BezierSpline(self._M, [self.control_points[ind]]), t)

    def isC1(self, eps=1e-5):
        """
        Check whether the spline is (approximately) continuously differentible. For this, all control points that connect
        two segments must be in the middle of their neighbours.
        """
        cp = self.control_points

        # trivial case: only one segment -> infinitly often differentible
        if len(cp) == 1:
            return True

        for i, seg in enumerate(cp[1:]):
            p = self._M.connec.geopoint(cp[i-1][-2], seg[1], 1/2)
            # if midpoint and connecting control point are further apart than epsilon return False
            if self._M.metric.dist(p, seg[0]) > eps:
                return False

        return True

    def geoshaped(self, eps=1e-7):
        """
        Return whether the spline is a reparametrized geodesic. For this we test if all tangent vectors from the first
        control point to the other control points are parallel (within a tolerance of epsilon).
        """
        cp = self.control_points.copy()

        # trivial case
        if len(cp) == 1 and len(cp[0]) == 2:
            return True

        c = cp[0][0]
        v0 = self._M.connec.log(c, cp[0][1])
        cp[0] = cp[0][2:]
        # check whether the logs at c to all other control points are parallel to v0
        for seg in cp:
            for cc in seg:
                # ignore almost equal points---the test is unstable for them and their influence in non-geodecity is
                # negligable
                if self._M.metric.dist(c, cc) > 1e-7:
                    v = self._M.connec.log(c, cc)
                    par = self._M.metric.inner(c, v0, v) / (self._M.metric.norm(c, v0) * self._M.metric.norm(c, v))

                    if -1 + eps < par < 1 - eps:
                        # v and v0 are not parallel
                        return False
        # all vectors were (almost) parallel
        return True

    def eval(self, t: float):
        """Evaluates the Bézier spline at time t."""

        # choose correct control points
        ind, t = segmentize(t)
        P = self.control_points[ind]

        return decasteljau(self._M, P, t)[0]

    def adjDpB(self, t, X):
        """Compute the value of the adjoint derivative of a Bézier curve B with respect to its control points applied
        to the vector X.
        :param t: scalar in [0, nSegments]
        :param X: tangent vector at B(t)
        :return: vectors at the control points
        """

        M = self._M
        siz = list(X.shape)
        # insert 1 in front
        siz.insert(0, 1)

        # t indicates which element of P to choose
        ind, t = segmentize(t)
        P = self.control_points[ind]

        # number of control points of corresponding segment
        k = len(P)

        b, B = decasteljau(M, P, t)
        # want to go backwards from B(t) to control points
        B.reverse()

        # initialize list for intermediate vectors
        D = []
        s = siz.copy()
        for i in range(1, len(B) + 1):
            s[0] = i + 1
            D.append(jnp.zeros(s))

        # transport X backwards along the "tree of geodesics" defined by the generalized de Casteljau algorithm.
        # We iterate over the depth of the tree and add vectors from the same tangent space.
        for i in range(k-1):
            if i == 0:
                D_old = jnp.zeros(siz)
                D_old = D_old.at[0].set(X)
            else:
                D_old = D[i - 1]

            siz = np.array(D_old.shape)
            siz[0] *= 2
            D_tilde = jnp.zeros(siz)
            for jj in range(siz[0] // 2):
                # transport to starting point of the geodesic
                D_tilde = D_tilde.at[2 * jj].set(M.connec.adjDxgeo(B[i][jj], B[i][jj + 1], t, D_old[jj]))
                # and to the endpoint
                D_tilde = D_tilde.at[2 * jj + 1].set(M.connec.adjDygeo(B[i][jj], B[i][jj + 1], t, D_old[jj]))

            D[i] = D[i].at[0].set(D_tilde[0])
            D[i] = D[i].at[-1].set(D_tilde[-1])

            # add up vectors
            for jj in range(1, D[i].shape[0] - 1):
                D[i] = D[i].at[jj].set(D_tilde[2 * jj - 1] + D_tilde[2 * jj])

        # return D[-1]

        grad = jnp.zeros_like(self.control_points)

        # update the entries corresponding to the ind-th segment
        grad = grad.at[ind].set(D[-1])

        return grad

def segmentize(t):
    """Choose the correct segment and value for the parameter t
    :param t: scalar in [0, nsegments]
    :return: index of corresponding control points in self.control_points and the adjusted value of t in [0,1]
    """

    def startpoint(t):
        return int(0), t

    def connecting_point(t):
        return t.astype(int) - 1, 1.

    def inner_point(t):
        return jnp.floor(t).astype(int), t - jnp.floor(t)

    return lax.cond(t == 0, startpoint, lambda s: lax.cond(t == jnp.round(t), connecting_point, inner_point, s), t)


def decasteljau(M, P, t):
    """Generalized de Casteljau algorithm
    :param M: manifold
    :param P: control points of curve beta
    :param t: scalar in [0,1]
    :return  beta(t), (B): result of the de Casteljau algorithm with control points P, (intermediate points B in the algorithm)
    """
    # number of control points
    k = len(P)

    # init linearized tree of control points
    B = jnp.concatenate([jnp.asarray(P)[i:] for i in range(k)])
    # for lower-level control points: indices of parent ones w.r.t B
    offset = [(2*k*n - n*n + n)//2 for n in range(k-1)]
    idx = np.concatenate([np.arange(k-1-i)+o for i, o in enumerate(offset)])
    # compute lower-level points
    f = lambda B, io: (B.at[io[1]].set(M.connec.geopoint(B[io[0]], B[io[0]+1], t)), None)
    B = lax.scan(f, B, np.c_[idx, k+np.arange(len(idx))])[0]

    return B[-1], [B[o:o+k-i] for i, o in enumerate(offset)]
