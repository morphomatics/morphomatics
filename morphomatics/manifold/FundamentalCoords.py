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

import os
import numpy as np

from scipy import sparse

try:
    from sksparse.cholmod import cholesky as direct_solve
except:
    from scipy.sparse.linalg import factorized as direct_solve

from ..geom import Surface
from . import SO3
from . import SPD
from . import ShapeSpace, Metric, Connection


class FundamentalCoords(ShapeSpace, Metric, Connection):
    """
    Shape space based on fundamental coordinates.

    See:
    Felix Ambellan, Stefan Zachow, and Christoph von Tycowicz.
    A Surface-Theoretic Approach for Statistical Shape Modeling.
    Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI), LNCS, 2019.
    """

    def __init__(self, reference: Surface, structure='product', metric_weights=(1.0, 1.0)):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        :arg metric_weights: weights (rotation, stretch) for commensuration between rotational and stretch parts
        """
        assert reference is not None
        self.ref = reference
        self.init_face = int(os.getenv('FCM_INIT_FACE', 0))                    # initial face for spanning tree path
        self.init_vert = int(os.getenv('FCM_INIT_VERT', 0))                    # id of fixed vertex

        self.integration_tol = float(os.getenv('FCM_INTEGRATION_TOL', 1e-05))  # integration tolerance local/global solver
        self.integration_iter = int(os.getenv('FCM_INTEGRATION_ITER', 3))      # max iteration local/global solver

        omega_C = float(os.getenv('FCM_WEIGHT_ROTATION', metric_weights[0]))
        omega_U = float(os.getenv('FCM_WEIGHT_STRETCH', metric_weights[1]))
        self.metric_weights = (omega_C, omega_U)

        self.spanning_tree_path = self.setup_spanning_tree_path()

        # rotation and stretch manifolds
        self.SO = SO3(int(0.5 * self.ref.inner_edges.getnnz()))        # relative rotations (transition rotations)
        self.SPD = SPD(self.ref.f.shape[0], 2)                         # stretch w.r.t. tangent space

        self.update_ref_geom(self.ref.v)

        name = f'Fundamental Coordinates Shape Space ({structure})'
        dimension = self.SO.dim + self.SPD.dim
        point_shape = [2, self.ref.f.shape[0], 3, 3]
        super().__init__(name, dimension, point_shape, self, self, None)

    @property
    def __str__(self):
        return self._name

    @property
    def n_triangles(self):
        """Number of triangles of the reference surface
        """
        return self.ref.f.shape[0]

    def update_ref_geom(self, v):
        self.ref.v=v

        # center of gravity
        self.CoG = self.ref.v.mean(axis=0)

        # setup Poisson system
        S = self.ref.div @ self.ref.grad
        # add soft-constraint fixing translational DoF
        S += sparse.coo_matrix(([1.0], ([0], [0])), S.shape)  # make pos-def
        self.poisson = direct_solve(S.tocsc())

        self.ref_frame_field = self.setup_frame_field()

        edgeAreaFactor = np.divide(self.ref.edge_areas, np.sum(self.ref.edge_areas))
        faceAreaFactor = np.divide(self.ref.face_areas, np.sum(self.ref.face_areas))

        # setup mass matrix (weights for each triangle and inner edge)
        diag = np.concatenate((self.metric_weights[0] * np.repeat(edgeAreaFactor, 9), self.metric_weights[1] * np.repeat(faceAreaFactor, 4)), axis=None)
        self.mass = sparse.diags(diag, 0)

        self._identity = self.to_coords(self.ref.v)

    def disentangle(self, c):
        """
        :arg c: vectorized fundamental coords. (tangent vectors)
        :returns: de-vectorized tuple of rotations and stretches (skew-sym. and sym. matrices)
        """
        # 2xkx3x3 array, rotations are stored in [0, :, :, :] and stretches in [1, :, :, :]
        e = int(0.5 * self.ref.inner_edges.getnnz())
        return c[:9*e].reshape(-1, 3, 3), c[9*e:].reshape(-1, 2, 2)

    def to_coords(self, v):
        """
        :arg v: #v-by-3 array of vertex coordinates
        :return: fundamental coords.
        """
        # compute gradients
        D = self.ref.grad @ v

        # decompose...
        U, S, Vt = np.linalg.svd(D.reshape(-1, 3, 3))

        # D holds transpose of def. grads.
        # -> compute left polar decomposition for right stretch tensor

        # ...rotation
        R = np.einsum('...ij,...jk', U, Vt)
        W = np.ones_like(S)
        W[:, -1] = np.linalg.det(R)
        R = np.einsum('...ij,...j,...jk', U, W, Vt)

        # ...stretch
        S[:, -1] = 1  # no stretch (=1) in normal direction
        # for degenerate triangles
        # TODO: check which direction is normal in degenerate case
        S[S < 1e-6] = 1e-6
        U = np.einsum('...ij,...j,...kj', U, S, U)

        # frame field on actual shape pushed over from reference shape
        frame = np.einsum('...ji,...jk', R, self.ref_frame_field)

        # setup ...transition rotations for every inner edge
        e = sparse.triu(self.ref.inner_edges).tocoo()
        C = np.zeros((e.getnnz(), 3, 3))
        C[e.data[:]] = np.einsum('...ji,...jk', frame[e.row[:]], frame[e.col[:]])

        # transform ...stretch from gobal (standard) coordinates to tangential Ulocal
        # frame.T * U * frame
        Ulocal = np.einsum('...ji,...jk,...kl', self.ref_frame_field, U, self.ref_frame_field)
        Ulocal = Ulocal[:,0:-1, 0:-1]

        return np.concatenate([np.ravel(C), np.ravel(Ulocal)]).reshape(-1)

    def from_coords(self, c):
        """
        :arg c: fundamental coords.
        :returns: #v-by-3 array of vertex coordinates
        """
        ################################################################################################################
        # initialization with spanning tree path #######################################################################
        C, Ulocal = self.disentangle(c)

        eIds = self.spanning_tree_path[:,0]
        fsourceId = self.spanning_tree_path[:, 1]
        ftargetId = self.spanning_tree_path[:, 2]

        # organize transition rotations along the path
        CoI = C[eIds[:]]
        CC = np.zeros_like(CoI)
        BB = (fsourceId < ftargetId)
        CC[BB] = CoI[BB]
        CC[~BB] = np.einsum("...ij->...ji", CoI[~BB])

        R= np.repeat(np.eye(3)[np.newaxis, :, :], len(self.ref.f), axis=0)

        # walk along path and initialize rotations
        CC = np.einsum('...jk,...kl,...ml', self.ref_frame_field[fsourceId], CC, self.ref_frame_field[ftargetId])
        for l in range(eIds.shape[0]):
            R[ftargetId[l]] = R[fsourceId[l]] @ CC[l]

        # transform (tangential) Ulocal to gobal (standard) coordinates
        U = np.zeros_like(R)
        U[:, 0:-1, 0:-1] = Ulocal
        # frame * U * frame.T
        U = np.einsum('...ij,...jk,...lk', self.ref_frame_field, U, self.ref_frame_field)

        idx_1, idx_2, idx_3, n_1, n_2, n_3 = self.ref.neighbors

        e = sparse.triu(self.ref.inner_edges).tocoo(); f = sparse.tril(self.ref.inner_edges).tocoo()

        e.data += 1; f.data += 1

        CC = np.zeros((C.shape[0] + 1, 3, 3)); CCt = np.zeros((C.shape[0] + 1, 3, 3))
        CC[e.data] = C[e.data - 1]; CCt[f.data] = np.einsum("...ij->...ji", C[f.data - 1])

        e = e.tocsr(); f = f.tocsr()

        Dijk = R.copy()
        n_iter = 0
        v = np.asarray(self.ref.v.copy())
        vk = np.asarray(self.ref.v.copy())
        sqrt_tol = np.sqrt(self.integration_tol)
        while n_iter < self.integration_iter:
        ################################################################################################################
        # global step ##################################################################################################

            # setup gradient matrix and solve Poisson system
            D = np.einsum('...ij,...kj', U, R)  # <-- from left polar decomp.
            rhs = self.ref.div @ D.reshape(-1, 3)
            vk = v
            v = self.poisson(rhs)
            v += self.CoG - v.mean(axis=0)
            errCoord = np.amax(np.abs((v - vk)))
            errCoordTol = sqrt_tol * (1.0 + np.amax(np.abs((vk))))

        ################################################################################################################
        # local step ###################################################################################################
            if (n_iter + 1 == self.integration_iter) or (errCoord < errCoordTol):
                break

            # compute gradients again
            D = (self.ref.grad @ v).reshape(-1, 3, 3)

            Dijk[idx_1] = np.einsum('...ji,...jk,...kl,...lm,...nm', D[n_1[:, 0]], U[n_1[:, 0]], self.ref_frame_field[n_1[:, 0]], CCt[e[idx_1, n_1[:, 0]]] + CC[f[idx_1, n_1[:, 0]]], self.ref_frame_field[idx_1])
            if n_2.shape[0] > 0 :
                Dijk[idx_2] = Dijk[idx_2] + np.einsum('...ji,...jk,...kl,...lm,...nm', D[n_2[:, 1]], U[n_2[:, 1]], self.ref_frame_field[n_2[:, 1]], CCt[e[idx_2, n_2[:, 1]]] + CC[f[idx_2, n_2[:, 1]]], self.ref_frame_field[idx_2])
            if n_3.shape[0] > 0 :
                Dijk[idx_3] = Dijk[idx_3] + np.einsum('...ji,...jk,...kl,...lm,...nm', D[n_3[:, 2]], U[n_3[:, 2]], self.ref_frame_field[n_3[:, 2]], CC[f[idx_3, n_3[:, 2]]] + CCt[e[idx_3, n_3[:, 2]]], self.ref_frame_field[idx_3])

            Uijk, Sijk, Vtijk = np.linalg.svd(Dijk)
            R = np.einsum('...ij,...jk', Uijk, Vtijk)
            Wijk = np.ones_like(Sijk)
            Wijk[:, -1] = np.linalg.det(R)
            R = np.einsum('...ij,...j,...jk', Uijk, Wijk, Vtijk)

            n_iter += 1

        # orient w.r.t. fixed frame and move to fixed node
        v[:] = (self.ref_frame_field[self.init_face] @ FundamentalCoords.frame_of_face(v, self.ref.f, [self.init_face]).T @ v[:].T).T
        v += self.ref.v[self.init_vert] - v[self.init_vert]
        # print("v:\n", v)
        return v

    @property
    def ref_coords(self):
        return self._identity

    def rand(self):
        R = self.SO.rand()
        U = self.SPD.rand()
        return self.entangle(R, U)

    def zerovec(self):
        """Returns the zero vector in any tangent space."""
        return self.entangle(self.SO.zerovec(), self.SPD.zerovec())

    def projToGeodesic(self, X, Y, P, max_iter = 10):
        '''
        Project P onto geodesic from X to Y.

        See:
        Felix Ambellan, Stefan Zachow, Christoph von Tycowicz.
        Geodesic B-Score for Improved Assessment of Knee Osteoarthritis.
        Proc. Information Processing in Medical Imaging (IPMI), LNCS, 2021.

        :arg X, Y: manifold coords defining geodesic X->Y.
        :arg P: manifold coords to be projected to X->Y.
        :returns: manifold coords of projection of P to X->Y
        '''

        assert X.shape == Y.shape
        assert Y.shape == P.shape

        # all tagent vectors in common space i.e. algebra
        v = self.connec.log(X, Y)
        v /= self.metric.norm(X, v)

        # initial guess
        Pi = X

        # solver loop
        for _ in range(max_iter):
            w = self.connec.log(Pi, P)
            d = self.metric.inner(Pi, v, w)

            # print(f'|<v, w>|={d}')
            if abs(d) < 1e-6: break

            Pi = self.connec.exp(Pi, d * v)

        return Pi

    ##########################################################
    # Implement Metric interface
    ##########################################################

    def dist(self, X, Y):
        """Returns the geodesic distance between two points p and q on the
        manifold."""
        return self.norm(X, self.log(X, Y))

    @property
    def typicaldist(self):
        return np.sqrt(self.SO.metric.typicaldist()**2 + self.SPD.metric.typicaldist()**2)

    def inner(self, X, G, H):
        """
        :arg G: (list of) tangent vector(s) at X
        :arg H: (list of) tangent vector(s) at X
        :returns: inner product at X between G and H, i.e. <G,H>_X
        """
        return G @ self.mass @ np.asanyarray(H).T

    def proj(self, X, A):
        """orthogonal (with respect to the euclidean inner product) projection of ambient
        vector (vectorized (2,k,3,3) array) onto the tangentspace at X"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(A)

        # project in each component
        r = self.SO.metric.proj(R, r)
        u = self.SPD.metric.proj(U, u)

        return np.concatenate([r, u]).reshape(-1)

    def egrad2rgrad(self, X, D):
        """converts euclidean gradient(vectorized (2,k,3,3) array))
        into riemannian gradient, vectorized inputs!"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(D)

        # componentwise
        r = self.SO.metric.egrad2rgrad(R, r)
        u = self.SPD.metric.egrad2rgrad(U, u)
        grad = np.concatenate([r, u]).reshape(-1)

        # multiply with inverse of metric
        grad /= self.mass.diagonal()

        return grad

    def ehess2rhess(self, p, G, H, X):
        """Converts the Euclidean gradient G and Hessian H of a function at
        a point p along a tangent vector X to the Riemannian Hessian
        along X on the manifold.
        """
        return

    ##########################################################
    # Implement Connection interface
    ##########################################################

    def exp(self, X, G):
        # disentangle coords. into rotations and stretches
        C, U = self.disentangle(X)
        c, u = self.disentangle(G)

        # alloc coords.
        Y = np.zeros_like(X)
        Cy, Uy = self.disentangle(Y)

        # exp C
        Cy[:] = self.SO.connec.exp(C, c)
        # exp U (avoid additional exp/log)
        Uy[:] = self.SPD.connec.exp(U, u)

        return Y

    def geopoint(self, X, Y, t):
        return self.exp(X, t * self.log(X, Y))

    retr = exp

    def log(self, X, Y):
        # disentangle coords. into rotations and stretches
        Cx, Ux = self.disentangle(X)
        Cy, Uy = self.disentangle(Y)

        # alloc tangent vector
        y = np.zeros(9 * int(0.5 * self.ref.inner_edges.getnnz()) + 4 * len(self.ref.f))
        c, u = self.disentangle(y)

        # log R1
        c[:] = self.SO.connec.log(Cx, Cy)
        # log U (avoid additional log/exp)
        u[:] = self.SPD.connec.log(Ux, Uy)

        return y

    def transp(self, X, Y, G):
        """
        :param X: element of the space of fundamental coordinates
        :param Y: element of the space of fundamental coordinates
        :param G: tangent vector at X
        :return: parallel transport of G along the geodesic from X to Y
        """
        # disentangle coords. into rotations and stretches
        Cx, Ux = self.disentangle(X)
        Cy, Uy = self.disentangle(Y)
        cx, ux = self.disentangle(G)

        # alloc coords.
        Y = np.zeros_like(X)
        cy, uy = self.disentangle(Y)

        cy[:] = self.SO.connec.transp(Cx, Cy, cx)
        uy[:] = self.SPD.connec.transp(Ux, Uy, ux)

        return Y

    def jacobiField(self, p, q, t, X):
        """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: tangent vector at gam(t)
        """
        raise NotImplementedError()

    def adjJacobi(self, X, Y, t, G):
        """
        Evaluates an adjoint Jacobi field along the geodesic gam from X to Z at X.
        :param X: element of the space of fundamental coordinates
        :param Y: element of the space of fundamental coordinates
        :param t: scalar in [0,1]
        :param G: tangent vector at gam(t)
        :return: tangent vector at X
        """

        # assert X.shape == Y.shape and X.shape == G.shape

        if t == 0:
            return G
        elif t == 1:
            return np.zeros_like(G)

        # disentangle coords. into rotations and stretches
        Cx, Ux = self.disentangle(X)
        Cy, Uy = self.disentangle(Y)

        c, u = self.disentangle(G)

        j = np.zeros_like(G)
        jr, js = self.disentangle(j)

        # SO(3) part
        jr[:] = self.SO.connec.adjJacobi(Cx, Cy, t, c)
        # Sym+(3) part
        js[:] = self.SPD.connec.adjJacobi(Ux, Uy, t, u)

        return j

    def adjDxgeo(self, X, Y, t, G):
        """Evaluates the adjoint of the differential of the geodesic gamma from X to Y w.r.t the starting point X at G,
        i.e, the adjoint  of d_X gamma(t; ., Y) applied to G, which is en element of the tangent space at gamma(t).
        """
        return self.adjJacobi(X, Y, t, G)

    def adjDygeo(self, X, Y, t, G):
        """Evaluates the adjoint of the differential of the geodesic gamma from X to Y w.r.t the endpoint Y at G,
        i.e, the adjoint  of d_Y gamma(t; X, .) applied to G, which is en element of the tangent space at gamma(t).
        """
        return self.adjJacobi(Y, X, 1 - t, G)

    # def jacop(self, X, Y, r):
    #     """ Evaluate the Jacobi operator along the geodesic from X to Y at r.
    #
    #     For the definition of the Jacobi operator see:
    #         Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.
    #
    #     :param X: element of the space of fundamental coordinates
    #     :param Y: element of the space of fundamental coordinates
    #     :param r: tangent vector at the rotational part of X
    #     :returns: skew-symmetric part of J_G(H)
    #     """
    #     v, w = self.disentangle(self.log(X, Y))
    #     w[:] = 0 * w
    #     v = 1 / 4 * (-np.einsum('...ij,...jk,...kl', v, v, r) + 2 * np.einsum('...ij,...jk,...kl', v, r, v)
    #                  - np.einsum('...ij,...jk,...kl', r, v, v))
    #
    #     return v
    #
    # def jacONB(self, X, Y):
    #     """
    #     Let J be the Jacobi operator along the geodesic from X to Y. This code diagonalizes J. Note that J restricted
    #     to the Sym+ part is the zero operator.
    #     :param X: element of the space of fundamental coordinates
    #     :param Y: element of the space of fundamental coordinates
    #     :returns lam, G: eigenvalues and orthonormal eigenbasis of  the rotational part of J at X
    #     """
    #     Cx, Ux = self.disentangle(X)
    #     Cy, Uy = self.disentangle(Y)
    #     return self.SO.jacONB(Cx, Cy)

    def setup_spanning_tree_path(self):
        """
        Setup a path across spanning tree of the refrence surface beginning at self.init_face.
        :return: n x 3 - array holding column wise an edge id and the respective neighbouring faces.
        """
        depth =[-1]*(len(self.ref.f))

        depth[self.init_face] = 0
        idcs = []
        idcs.append(self.init_face)

        spanningTreePath = []
        while idcs:
            idx = idcs.pop(0)
            d = depth[idx] + 1
            neighs = self.ref.inner_edges.getrow(idx).tocoo()

            for neigh, edge in zip(neighs.col, neighs.data):
                if depth[neigh] >= 0:
                    continue
                depth[neigh] = d
                idcs.append(neigh)

                spanningTreePath.append([edge, idx, neigh])
        return np.asarray(spanningTreePath)

    def setup_frame_field(self):
        """
        Compute frames for every face of the surface with some added pi(e).
        :return: n x 3 x 3 - array holding one frame for every face, column wise organized with c1, c2 tangential and c3 normal..
        """
        v1 = self.ref.v[self.ref.f[:, 2]] - self.ref.v[self.ref.f[:, 1]]
        v2 = self.ref.v[self.ref.f[:, 0]] - self.ref.v[self.ref.f[:, 2]]

        # orthonormal basis for face plane
        proj = np.divide(np.einsum('ij,ij->i', v2, v1), np.einsum('ij,ij->i', v1, v1))
        proj = sparse.diags(proj)

        v2 = v2 - proj @ v1

        # normalize and calculation of normal
        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        v3 = np.cross(v1, v2, axisa=1, axisb=1, axisc=1)

        # shape as n x 3 x 3 with basis vectors as cols
        frame = np.reshape(np.concatenate((v1, v2, v3), axis=1), [-1, 3, 3])
        frame = np.einsum('ijk->ikj', frame)

        return frame

    @staticmethod
    def frame_of_face(v, f, fId : int):
        """
        :arg fId: id of face to caluclate frame for
        :return: frame (colunm wise) with c1, c2 tangential and c3 normal.
        """
        v1 = v[f[fId, 2]] - v[f[fId, 1]]
        v2 = v[f[fId, 0]] - v[f[fId, 2]]

        # orthonormal basis for face plane
        v2 = v2 - (np.dot(v2, v1.T) / np.dot(v1, v1.T)) * v1

        # normalize and calculation of normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)

        return np.column_stack((v1.T, v2.T, v3.T))

    def flatCoords(self, X):
        """
        Project shape X isometrically to flat configuration.
        :param X: element of the space of fundamental coordinates
        :returns: Flattened configuration.
        """
        _, Ulocal = self.disentangle(X)

        inner_edge = sparse.triu(self.ref.inner_edges)

        C = np.zeros((self.SO.k,3,3))

        for l in range(inner_edge.data.shape[0]):
            # transition rotations are directed from triangle with lower to triangle with higher id
            i = inner_edge.row[l]
            j = inner_edge.col[l]

            ###### calc quaternion representing rotation from nj to ni ######

            ni = self.ref_frame_field[i][:, 2]
            nj = self.ref_frame_field[j][:, 2]

            lni_lnj = np.sqrt(np.dot(ni,ni)*np.dot(nj,nj))
            qw = lni_lnj + np.dot(ni,nj)

            # check for anti-parallelism of ni and nj
            if (qw < 1.0e-7 * lni_lnj):
                qw=0.0
                if(np.abs(ni[0]) > np.abs(ni[2])):
                    qxyz = np.array([ -ni[1],  ni[0], 0.0  ])
                else:
                    qxyz = np.array([    0.0, -ni[2], ni[1]])
            else:
                qxyz = np.cross(nj, ni)

            # normalize quaternion
            lq = np.sqrt(qw*qw + np.dot(qxyz, qxyz))
            qw = qw / lq
            qxyz = qxyz / lq

            ########## get rotation matrix from (unit) quarternion ##########

            Rninj = np.eye(3)

            qwqw = qw * qw
            qxqx = qxyz[0] * qxyz[0]
            qyqy = qxyz[1] * qxyz[1]
            qzqz = qxyz[2] * qxyz[2]
            qxqy = qxyz[0] * qxyz[1]
            qzqw = qxyz[2] * qw
            qxqz = qxyz[0] * qxyz[2]
            qyqw = qxyz[1] * qw
            qyqz = qxyz[1] * qxyz[2]
            qxqw = qxyz[0] * qw

            Rninj[0, 0] =  qxqx - qyqy - qzqz + qwqw
            Rninj[1, 1] = -qxqx + qyqy - qzqz + qwqw
            Rninj[2, 2] = -qxqx - qyqy + qzqz + qwqw
            Rninj[1, 0] = 2.0 * (qxqy + qzqw)
            Rninj[0, 1] = 2.0 * (qxqy - qzqw)
            Rninj[2, 0] = 2.0 * (qxqz - qyqw)
            Rninj[0, 2] = 2.0 * (qxqz + qyqw)
            Rninj[2, 1] = 2.0 * (qyqz + qxqw)
            Rninj[1, 2] = 2.0 * (qyqz - qxqw)

            #################################################################

            # update transition rotations
            C[inner_edge.data[l]] = self.ref_frame_field[i].T @ Rninj @ self.ref_frame_field[j]

        return np.concatenate([np.ravel(C), np.ravel(Ulocal)]).reshape(-1)
