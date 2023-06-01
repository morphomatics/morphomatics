################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2023 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np
import jax
import jax.numpy as jnp

from scipy import sparse

try:
    from sksparse.cholmod import cholesky as direct_solve
except:
    from scipy.sparse.linalg import factorized as direct_solve

from ..geom import Surface
from . import SO3
from . import SPD
from . import ShapeSpace, Metric
from .util import align


class DifferentialCoords(ShapeSpace, Metric):
    """
    Shape space based on differential coordinates.

    See:
    Christoph von Tycowicz, Felix Ambellan, Anirban Mukhopadhyay, and Stefan Zachow.
    An Efficient Riemannian Statistical Shape Model using Differential Coordinates.
    Medical Image Analysis, Volume 43, January 2018.
    """

    def __init__(self, reference: Surface, commensuration_weights=(1.0, 1.0)):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        :arg commensuration_weights: weights (rotation, stretch) for commensuration between rotational and stretch parts
        """
        assert reference is not None
        self.ref = reference

        self.commensuration_weights = commensuration_weights

        self.update_ref_geom(self.ref.v)

        # rotation and stretch manifolds
        self.SO = SO3(len(self.ref.f))
        self.SPD = SPD(len(self.ref.f))

        name = f'Differential Coordinates Shape Space'
        dimension = self.SO.dim + self.SPD.dim
        point_shape = (2, len(self.ref.f), 3, 3)
        super().__init__(name, dimension, point_shape, self, self, None)

    def update_ref_geom(self, v):
        self.ref.v=v

        # center of gravity
        self.CoG = self.ref.v.mean(axis=0)

        # setup Poisson system
        S = self.ref.div @ self.ref.grad
        # add soft-constraint fixing translational DoF
        S += sparse.coo_matrix(([1.0], ([0], [0])), S.shape)  # make pos-def
        self.poisson = direct_solve(S.tocsc())

        # # setup mass matrix (weights for each triangle)
        # diag = np.outer(np.repeat(self.ref.face_areas, 9), self.commensuration_weights)
        # self.mass = sparse.diags(diag.T.flatten(), 0)
        # -> single weight for each triangle -> use broadcasting instead of sparse matrix product
        mass = np.outer(self.commensuration_weights, self.ref.face_areas)
        self.mass = jnp.array(mass).reshape(mass.shape+(1, 1))

    def disentangle(self, c):
        """
        :arg c: vectorized differential coords. (or tangent vectors)
        :returns: de-vectorized tuple of rotations and stretches (skew-sym. and sym. matrices)
        """
        # 2xkx3x3 array, rotations are stored in [0, :, :, :] and stretches in [1, :, :, :]
        # m = len(self.ref.f)
        # return c.reshape(-1, 3, 3)[:m], c.reshape(-1, 3, 3)[m:]
        return c[0], c[1]

    def entangle(self, R, U):
        """
        Inverse of #disentangle().
        :arg R: rotational components
        :arg U: stretch components
        :returns: concatenated and vectorized version
        """
        # return np.concatenate([R, U]).reshape(-1)
        return jnp.array([R, U])

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
        # for degenerate triangles
        # TODO: check which direction is normal in degenerate case
        S[S < 1e-6] = 1e-6
        U = np.einsum('...ij,...j,...kj', U, S, U)

        return self.entangle(R, U)

    def from_coords(self, c):
        """
        :arg c: differentical coords.
        :returns: #v-by-3 array of vertex coordinates
        """
        # compose
        R, U = self.disentangle(c)
        D = jnp.einsum('...ij,...jk', U, R)  # <-- from left polar decomp.

        # solve Poisson system
        rhs = self.ref.div @ D.reshape(-1, 3)
        v = self.poisson(rhs)
        # move to CoG
        v += self.CoG - v.mean(axis=0)

        return v

    @property
    def ref_coords(self):
        return jnp.tile(jnp.eye(3), (2*len(self.ref.f), 1)).reshape(self.point_shape)

    def rand(self, key: jax.random.KeyArray):
        k1, k2 = jax.random.split(key)
        return self.entangle(self.SO.rand(k1), self.SPD.rand(k2))

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

        # all tagent vectors in common space i.e. algebra
        v = self.connec.log(X, Y)
        v = v / self.metric.norm(X, v)

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

    def proj(self, X, A):
        """orthogonal (with respect to the euclidean inner product) projection of ambient
        vector (i.e. (2,k,3,3) array) onto the tangentspace at X"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(A)

        # project in each component
        r = self.SO.proj(R, r)
        u = self.SPD.proj(U, u)

        return self.entangle(r, u)

    ##########################################################
    # Implement Metric interface
    ##########################################################

    def dist(self, X, Y):
        """Returns the geodesic distance between two points p and q on the
        manifold."""
        return self.norm(X, self.log(X, Y))

    def squared_dist(self, X, Y):
        """Returns the geodesic distance between two points p and q on the
        manifold."""
        d = self.log(X, Y)
        return self.inner(X, d, d)

    @property
    def typicaldist(self):
        return jnp.sqrt(self.SO.metric.typicaldist()**2 + self.SPD.metric.typicaldist()**2)

    def inner(self, X, G, H):
        """
        :arg G: tangent vector at X
        :arg H: tangent vector at X
        :returns: inner product at X between P_G and H, i.e. <P_G,H>_X
        """
        # return P_G @ self.mass @ np.asanyarray(H).T
        return (self.mass * H).reshape(-1) @ G.reshape(-1)

    def flat(self, X, G):
        raise NotImplementedError('This function has not been implemented yet.')

    def sharp(self, G, dG):
        raise NotImplementedError('This function has not been implemented yet.')

    def egrad2rgrad(self, X, D):
        """converts Euclidean gradient (i.e. (2,k,3,3) array))
        into Riemannian gradient"""
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(D)

        # componentwise
        r = self.SO.metric.egrad2rgrad(R, r)
        u = self.SPD.metric.egrad2rgrad(U, u)
        grad = self.entangle(r, u)

        # multiply with inverse mass matrix
        grad = grad / self.mass

        return grad

    def ehess2rhess(self, p, G, H, X):
        """Converts the Euclidean gradient P_G and Hessian H of a function at
        a point p along a tangent vector X to the Riemannian Hessian
        along X on the manifold.
        """
        raise NotImplementedError('This function has not been implemented yet.')

    ##########################################################
    # Implement Connection interface
    ##########################################################

    def exp(self, X, G):
        # disentangle coords. into rotations and stretches
        R, U = self.disentangle(X)
        r, u = self.disentangle(G)
        return self.entangle(self.SO.connec.exp(R, r), self.SPD.connec.exp(U, u))

    retr = exp

    def log(self, X, Y):
        # disentangle coords. into rotations and stretches
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)
        return self.entangle(self.SO.connec.log(Rx, Ry), self.SPD.connec.log(Ux, Uy))

    def curvature_tensor(self, p, X, Y, Z):
        """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
        covariant derivative of Y in direction X and [] being the Lie bracket, the convention
            R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
        is used.
        """
        R, U = self.disentangle(p)
        r_x, u_x = self.disentangle(X)
        r_y, u_y = self.disentangle(Y)
        r_z, u_z = self.disentangle(Z)
        return self.entangle(self.SO.connec.curvature_tensor(R, r_x, r_y, r_z), self.SPD.connec.curvature_tensor(U, u_x, u_y, u_z))

    def transp(self, X, Y, G):
        """
        :param X: element of the space of differential coordinates
        :param Y: element of the space of differential coordinates
        :param G: tangent vector at X
        :return: parallel transport of P_G along the geodesic from X to Y
        """
        # disentangle coords. into rotations and stretches
        Rx, Ux = self.disentangle(X)
        Ry, Uy = self.disentangle(Y)
        rx, ux = self.disentangle(G)
        return self.entangle(self.SO.connec.transp(Rx, Ry, rx), self.SPD.connec.transp(Ux, Uy, ux))

    def jacobiField(self, p, q, t, X):
        """Evaluates a Jacobi field (with boundary conditions gam'(0) = X, gam'(1) = 0) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: [gam(t), J]
        """
        # disentangle coords. into rotations and stretches
        Rp, Up = self.disentangle(p)
        Rq, Uq = self.disentangle(q)
        r, u = self.disentangle(X)

        Jr = self.SO.connec.jacobiField(Rp, Rq, t, r)
        Ju = self.SPD.connec.jacobiField(Up, Uq, t, u)
        return [self.entangle(Jr[0], Ju[0]), self.entangle(Jr[1], Ju[1])]

    def adjJacobi(self, p, q, t, X):
        """Evaluates an adjoint Jacobi field (with boundary conditions gam'(0) = 0, gam'(t) = X) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at gam(t)
        :return: tangent vector at p
        """
        # disentangle coords. into rotations and stretches
        Rp, Up = self.disentangle(p)
        Rq, Uq = self.disentangle(q)
        r, u = self.disentangle(X)
        return self.entangle(self.SO.metric.adjJacobi(Rp, Rq, t, r), self.SPD.metric.adjJacobi(Up, Uq, t, u))

    def coords(self, X):
        """Coordinate map for the tangent space at the identity"""
        X = self.disentangle(X)
        x, y = self.SO.connec.coords(X[0]), self.SPD.connec.coords(X[1])
        return np.hstack((x, y))

    def coords_inverse(self, c):
        """Inverse of coords (SO coordinates come first)"""
        d = self.SO.dim
        a, b = self.SO.connec.coords_inverse(c[:d]), self.SPD.connec.coords_inverse(c[d:])
        return self.entangle(a, b)
