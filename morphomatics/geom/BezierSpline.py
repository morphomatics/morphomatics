################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2021 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see /LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np
from pymanopt.manifolds.manifold import Manifold


class BezierSpline:
    """Manifold-valued spline that consists of Bézier curves"""

    def __init__(self, M: Manifold, control_points, iscycle=False):
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
        """Returns the deegres of the spline segments."""
        K = len(self.control_points)
        n_seg = np.zeros(K, dtype=int)
        for i in range(K):
            n_seg[i] = np.shape(self.control_points[i])[0] - 1
        return n_seg

    @property
    def length(self):
        # TODO
        return

    @property
    def energy(self):
        # TODO
        return

    def isc1(self):
        # TODO: check C1 conditions approximately
        return

    def geoshaped(self):
        return  # TODO: test whether self is actually a (reparametrized) geodesic (unit tangent parallel transported?)

    def Mgeopoint(self, p, q, t):
        """Evaluates the geodesic from p to q at time t in [0,1].
        Some manifolds allow faster implementations than this generic one; e.g., Sym+."""
        return self._M.exp(p, self._M.log(p, q) * t)

    def eval(self, time=np.linspace(0, 1, 100)):
        """Evaluates the Bézier spline at one ore more points."""
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

    def tangent_vector(self, time=0):
        # TODO:  calculate tangent vectors at curve
        return

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
                D_tilde[2 * jj] = M.adjDxgeo(B[i][jj], B[i][jj + 1], t, D_old[jj])
                # and to the endpoint
                D_tilde[2 * jj + 1] = M.adjDygeo(B[i][jj], B[i][jj + 1], t, D_old[jj])

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


def decasteljau(M : Manifold, P, t, return_intermediate=False):
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
            Bl[i] = M.exp(A[i], M.log(A[i], A[i + 1]) * t)
        return Bl

    # number of control points
    k = len(P)
    B = [np.array(P)]

    for l in range(k - 1):
        P = single_layer(P, t)
        B.append(P)

    if return_intermediate:
        # last entry is B(t) = P
        B.pop(-1)
        return P[0], B
    else:
        return P[0]
