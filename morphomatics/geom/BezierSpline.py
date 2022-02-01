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

class BezierSpline:
    """Manifold-valued spline that consists of Bézier curves"""

    def __init__(self, M, control_points, iscycle=False):
        """
        :arg M: manifold in which the curve lies
        :arg control_points: list of arrays of control points of the Bézier spline sorted along the first axis. The l-th
        entry of the list belongs to the l-th segment of the spline.
        :arg iscycle: boolean indicating whether B is a closed curve
        """
        assert M is not None

        self._M = M

        self.control_points = control_points

        self.iscycle = iscycle

    def __str__(self):
        return 'Bézier spline through ' + self._M.__str__()

    @property
    def nsegments(self):
        """Returns the number of segments."""
        return len(self.control_points)

    @property
    def degrees(self):
        """Returns the degrees of the spline segments."""
        K = len(self.control_points)
        n_seg = np.zeros(K, dtype=int)
        for i in range(K):
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
                Single layer of the computation consisting of a a single step of the de Casteljau algorithm
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
                    X[ii] = v / self._M.connec.norm(P[ii], v) * self._M.metric.dist(P_old[ii], P_old[ii+1])

                # there are k+1 control points
                for l in range(k):
                    P, X = single_layer(P, s, X)

                return P, X

        # get segment and local parameter
        ind, t = self.segmentize(t)

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
            p = self.Mgeopoint(cp[i-1][-2], seg[1], 1/2)
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

    def Mgeopoint(self, p, q, t):
        """Evaluates the geodesic from p to q at time t in [0,1].
        Some manifolds allow faster implementations than this generic one; e.g., Sym+."""
        return self._M.connec.exp(p, self._M.connec.log(p, q) * t)

    def eval(self, time=np.linspace(0, 1, 100)):
        """Evaluates the Bézier spline at one or more points."""
        time = np.atleast_1d(time)
        # Spline is defined for t in [0, nsegments]
        assert all(0 <= time) and all(time <= self.nsegments)

        l = len(time)
        size = np.shape(self.control_points[0][0])
        Q = np.zeros(np.append(l, size))
        for i, t in enumerate(time):
            # choose correct control points
            ind, t = self.segmentize(t)
            P = self.control_points[ind]

            Q[i] = decasteljau(self._M, P, t)

        return Q if l > 1 else Q[0]

    def adjDpB(self, t, X):
        """Compute the value of the adjoint derivative of a Bézier curve B with respect to its control points applied
        to the vector X.
        :param t: scalar in [0, nSegments]
        :param X: tangent vector at B(t)
        :return: vectors at the control points
        """
        assert X.shape == self.control_points[0][0].shape and np.isscalar(t)

        M = self._M
        siz = list(X.shape)
        siz.insert(0, 1)

        # t indicates which element of P to choose
        ind, t = self.segmentize(t)
        P = self.control_points[ind]

        k = len(P)

        b, B = decasteljau(M, P, t, return_intermediate=True)
        # want to go backwards from B(t) to control points
        B.reverse()

        # initialize list for intermediate vectors
        D = []
        s = siz.copy()
        for i in range(1, len(B) + 1):
            s[0] = i + 1
            D.append(np.zeros(s))

        # transport X backwards along the "tree of geodesics" defined by the generalized de Casteljau algorithm.
        # We iterate over the depth of the tree and add vectors from the same tangent space.
        for i in range(k-1):
            if i == 0:
                D_old = np.zeros(siz)
                D_old[0] = X
            else:
                D_old = D[i - 1]

            siz = np.array(D_old.shape)
            siz[0] *= 2
            D_tilde = np.zeros(siz)
            for jj in range(siz[0] // 2):
                # transport to starting point of the geodesic
                D_tilde[2 * jj] = M.connec.adjDxgeo(B[i][jj], B[i][jj + 1], t, D_old[jj])
                # and to the endpoint
                D_tilde[2 * jj + 1] = M.connec.adjDygeo(B[i][jj], B[i][jj + 1], t, D_old[jj])

            D[i][0] = D_tilde[0]
            D[i][-1] = D_tilde[-1]

            # add up vectors
            for jj in range(1, D[i].shape[0] - 1):
                D[i][jj] = D_tilde[2 * jj - 1] + D_tilde[2 * jj]

        return D[-1]

    def segmentize(self, t):
        """Choose the correct segment and value for the parameter t
        :param t: scalar in [0, nsegments]
        :return: index of corresponding control points in self.control_points and the adjusted value of t in [0,1]
        """
        assert 0 <= t <= self.nsegments

        # choose correct control points
        if t == 0:
            ind = 0
        elif t == np.round(t):
            ind = t - 1
            ind = ind.astype(int)
            t = 1
        else:
            ind = np.floor(t).astype(int)
            t = t - np.floor(t)

        return ind, t


def decasteljau(M, P, t, return_intermediate=False):
    """Generalized de Casteljau algorithm
    :param M: manifold
    :param P: control points
    :param t: scalar in [0,1]
    :param return_intermediate: If True, return return intermediate points of the de Casteljau algorithm.
    :return  P, (B): result of the de Casteljau algorithm with control points P, (intermediate points B in the algorithm)
    """
    def single_layer(A, t):
        # averaging of a single layer in de Casteljau algorithm
        size = np.array(np.shape(A))
        # give back one point less
        size[0] = size[0] - 1
        Bl = np.zeros(size)
        for i in range(size[0]):
            # B[i] = self.geopoint(A[..., i], A[..., i + 1], t)
            Bl[i] = M.connec.exp(A[i], M.connec.log(A[i], A[i + 1]) * t)
        return Bl

    # number of control points
    k = len(P)
    B = [np.array(P)]

    # easy cases
    if t == 0:
        if return_intermediate:
            for l in range(1, k-1):
                # do not take uppermost control point
                B.append(P[:-l])
            return P[0], B
        else:
            return P[0]
    elif t == 1:
        if return_intermediate:
            for l in range(1, k-1):
                # do not take lowermost control point
                B.append(P[l:])
            return P[-1], B
        else:
            return P[-1]

    # computations are neccessary
    for l in range(k - 1):
        P = single_layer(P, t)
        B.append(P)

    if return_intermediate:
        # last entry is B(t) = P
        B.pop(-1)
        return P[0], B
    else:
        return P[0]
