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

from scipy import sparse

try:
    from sksparse.cholmod import cholesky as direct_solve
except:
    from scipy.sparse.linalg import factorized as direct_solve

from ..geom import Surface
from . import SO3
from . import SPD
from . import ShapeSpace
from .util import align


class DifferentialCoords(ShapeSpace):
    """
    Shape space based on differential coordinates.

    See:
    Christoph von Tycowicz, Felix Ambellan, Anirban Mukhopadhyay, and Stefan Zachow.
    An Efficient Riemannian Statistical Shape Model using Differential Coordinates.
    Medical Image Analysis, Volume 43, January 2018.
    """

    def __init__(self, reference: Surface, metric_weights=(1.0, 1.0)):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        :arg metric_weights: weights (rotation, stretch) for commensuration between rotational and stretch parts
        """
        assert reference is not None
        self.ref = reference

        self.metric_weights = metric_weights

        self.update_ref_geom(self.ref.v)

        # rotation and stretch manifolds
        self.SO = SO3(self.ref.f.shape[0])
        self.SPD = SPD(self.ref.f.shape[0])

        # tensor mapping versor to skew-sym. matrix
        I = np.eye(3)
        O = np.zeros(3)
        self.versor2skew = np.array([[O, -I[2], I[1]], [I[2], O, -I[0]], [-I[1], I[0], O]])
        # ... and the opposite direction
        self.skew2versor = .5 * self.versor2skew

    def __str__(self):
        return 'Differential Coordinates Shape Space'

    @property
    def dim(self):
        return self.SO.dim() + self.SPD.dim()

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def update_ref_geom(self, v):
        self.ref.v=v

        # center of gravity
        self.CoG = self.ref.v.mean(axis=0)

        # setup Poisson system
        S = self.ref.div @ self.ref.grad
        # add soft-constraint fixing translational DoF
        S += sparse.coo_matrix(([1.0], ([0], [0])), S.shape)  # make pos-def
        self.poisson = direct_solve(S.tocsc())

        # setup metric
        diag = np.repeat(self.ref.face_areas, 18).reshape(-1, 2) @ np.diag(self.metric_weights)
        self.metric = sparse.diags(diag.T.flatten(), 0)

    def disentangle(self, c):
        """
        :arg c: vetorized differential coords. (or tangent vectors)
        :returns: de-vectorized tuple of rotations and stretches (skew-sym. and sym. matrices)
        """
        # 2xkx3x3 array, rotations are stored in [0, :, :, :] and stretches in [1, :, :, :]
        m = len(self.ref.f)
        return c.reshape(-1, 3, 3)[:m], c.reshape(-1, 3, 3)[m:]

    def entangle(self, R, U):
        """
        Inverse of #disentangle().
        :arg R: rotational components
        :arg U: stretch components
        :returns: concatenated and vectorized version
        """
        return np.concatenate([R, U]).reshape(-1)

    def to_coords(self, v):
        """
        :arg v: #v-by-3 array of vertex coordinates
        :return: differentical coords.
        """

        # align
        v = align(v, self.ref.v)

        # compute gradients
        D = self.ref.grad @ v

        # D holds transpose of def. grads.
        # -> compute left polar decomposition for right stretch tensor

        # decompose...
        U, S, Vt = np.linalg.svd(D.reshape(-1, 3, 3))

        # ...rotation
        R = np.einsum('...ij,...jk', U, Vt)
        W = np.ones_like(S)
        W[:, -1] = np.linalg.det(R)
        R = np.einsum('...ij,...j,...jk', U, W, Vt)

        # ...stretch
        S[:, -1] = 1  # no stretch (=1) in normal direction
        U = np.einsum('...ij,...j,...kj', U, S, U)

        return self.entangle(R, U)

    def from_coords(self, c):
        """
        :arg c: differentical coords.
        :returns: #v-by-3 array of vertex coordinates
        """
        # compose
        R, U = self.disentangle(c)
        D = np.einsum('...ij,...jk', U, R)  # <-- from left polar decomp.

        # solve Poisson system
        rhs = self.ref.div @ D.reshape(-1, 3)
        v = self.poisson(rhs)
        # move to CoG
        v += self.CoG - v.mean(axis=0)

        return v

    @property
    def identity(self):
        return np.tile(np.eye(3), (2*len(self.ref.f), 1)).reshape(-1)

    def inner(self, X, G, H):
        """
        :arg G: (list of) tangent vector(s) at X
        :arg H: (list of) tangent vector(s) at X
        :returns: inner product at X between G and H, i.e. <G,H>_X
        """
        return G @ self.metric @ np.asanyarray(H).T

    def proj(self, X, A):
        """orthogonal (with respect to the euclidean inner product) projection of ambient
        vector (vectorized (2,k,3,3) array) onto the tangentspace at X"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(A)

        # project in each component
        r = self.SO.proj(R, r)
        u = self.SPD.proj(U, u)

        return np.concatenate([r, u]).reshape(-1)

    def egrad2rgrad(self, X, D):
        """converts euclidean gradient(vectorized (2,k,3,3) array))
        into riemannian gradient, vectorized inputs!"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(D)

        # componentwise
        r = self.SO.egrad2rgrad(R, r)
        u = self.SPD.egrad2rgrad(U, u)
        grad = np.concatenate([r, u]).reshape(-1)

        # multiply with inverse of metric
        grad /= self.metric.diagonal()

        return grad

    def ehess2rhess(self, X, H):
        # TODO
        return ehess

    def exp(self, X, G):
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(G)

        # alloc coords.
        Y = np.zeros_like(X)
        Ry, Uy = self.disentangle(Y)

        # exp R1
        Ry[:] = self.SO.exp(R, r)
        # exp U (avoid additional exp/log)
        Uy[:] = self.SPD.exp(U, u)

        return Y

    def geopoint(self, X, Y, t):
        return self.exp(X, t * self.log(X, Y))

    retr = exp

    def log(self, X, Y):
        # disentangle coords. into rotations and stretches
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)

        # alloc tangent vector
        y = np.zeros(X.size)
        r, u = self.disentangle(y)

        # log R1
        r[:] = self.SO.log(Rx, Ry)
        # log U (avoid additional log/exp)
        u[:] = self.SPD.log(Ux, Uy)

        return y

    def rand(self):
        R = self.SO.rand()
        U = self.SPD.rand()
        return self.entangle(R, U)

    def transp(self, X, Y, G):
        """
        :param X: element of the space of differential coordinates
        :param Y: element of the space of differential coordinates
        :param G: tangent vector at X
        :return: parallel transport of G along the geodesic from X to Y
        """
        # disentangle coords. into rotations and stretches
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)
        rx, ux = self.disentangle(G)

        # alloc coords.
        Y = np.zeros_like(X)
        ry, uy = self.disentangle(Y)

        ry[:] = self.SO.transp(Rx, Ry, rx)
        uy[:] = self.SPD.transp(Ux, Uy, ux)

        return Y

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
        v = self.log(X, Y)
        v /= self.norm(X, v)

        # initial guess
        Pi = X

        # solver loop
        for _ in range(max_iter):
            w = self.log(Pi, P)
            d = self.inner(Pi, v, w)

            # print(f'|<v, w>|={d}')
            if abs(d) < 1e-6: break

            Pi = self.exp(Pi, d * v)

        return Pi

    def jacop(self, X, Y, r):
        """ Evaluate the Jacobi operator along the geodesic from X to Y at r.

        For the definition of the Jacobi operator see:
            Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

        :param X: element of the space of differential coordinates
        :param Y: element of the space of differential coordinates
        :param r: tangent vector at the rotational part of X
        :returns: skew-symmetric part of J_G(H)
        """
        v, w = self.disentangle(self.log(X, Y))
        w[:] = 0 * w
        v = 1 / 4 * (-np.einsum('...ij,...jk,...kl', v, v, r) + 2 * np.einsum('...ij,...jk,...kl', v, r, v)
                     - np.einsum('...ij,...jk,...kl', r, v, v))

        return v

    def jacONB(self, X, Y):
        """
        Let J be the Jacobi operator along the geodesic from X to Y. This code diagonalizes J. Note that J restricted
        to the Sym+ part is the zero operator.
        :param X: element of the space of differential coordinates
        :param Y: element of the space of differential coordinates
        :returns lam, G: eigenvalues and orthonormal eigenbasis of  the rotational part of J at X
        """
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)
        return self.SO.jacONB(Rx, Ry)

    def adjJacobi(self, X, Y, t, G):
        """
        Evaluates an adjoint Jacobi field along the geodesic gam from X to Z at X.
        :param X: element of the space of differential coordinates
        :param Y: element of the space of differential coordinates
        :param t: scalar in [0,1]
        :param G: tangent vector at gam(t)
        :return: tangent vector at X
        """

        assert X.shape == Y.shape and X.shape == G.shape

        # disentangle coords. into rotations and stretches
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)

        r, u = self.disentangle(G)

        j = np.zeros_like(G)
        jr, js = self.disentangle(j)

        # SO(3) part
        jr[:] = self.SO.adjJacobi(Rx, Ry, t, r)
        # Sym+(3) part
        js[:] = self.SPD.adjJacobi(Ux, Uy, t, u)

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
