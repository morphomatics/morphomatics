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
from . import GLpn
from . import ShapeSpace
from .util import align


class GLpCoords(ShapeSpace):
    """
    Shape space based the group of matrices with positive determinant.

    See:
    Felix Ambellan, Stefan Zachow and Christoph von Tycowicz.
    An as-invariant-as-possible GL+(3)-based statistical shape model.
    Proc. 7th MICCAI workshop on Mathematical Foundations of Computational Anatomy, pp. 219--228, 2019.
    """

    def __init__(self, reference: Surface):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        """
        assert reference is not None
        self.ref = reference

        self.update_ref_geom(self.ref.v)

        self.GLp = GLpn(self.ref.f.shape[0], structure='AffineGroup')

        name = 'Shape Space based on the orientation preserving component of the general linear group'
        dimension = self.ref.f.shape[0] * 9
        point_shape = [self.ref.v.shape[0], 3, 3]
        super().__init__(name, dimension, point_shape, self.GLp.connec, None, self.GLp.group)

    @property
    def n_triangles(self):
        return self.ref.f.shape[0]

    def update_ref_geom(self, v):
        self.ref.v = v

        # center of gravity
        self.CoG = self.ref.v.mean(axis=0)

        # setup Poisson system
        S = self.ref.div @ self.ref.grad
        # add soft-constraint fixing translational DoF
        S += sparse.coo_matrix(([1.0], ([0], [0])), S.shape)  # make pos-def
        self.poisson = direct_solve(S.tocsc())

    def to_coords(self, v):
        """
        :arg v: #v-by-3 array of vertex coordinates
        :return: GLp coords.
        """

        # align
        v = align(v, self.ref.v)

        # compute gradients
        D = self.ref.grad @ v

        # D holds transpose of def. grads.
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
        return np.einsum('...ij,...jl', R, U)

    def from_coords(self, D):
        """
        :arg D: GLp coords.
        :returns: #v-by-3 array of vertex coordinates
        """
        # solve Poisson system
        rhs = self.ref.div @ D.reshape(-1, 3)
        v = self.poisson(rhs)
        # move to CoG
        v += self.CoG - v.mean(axis=0)

        return v

    @property
    def ref_coords(self):
        """ Identity coordinates (i.e., the reference shape).
        """
        return self.group.identity()

    def rand(self):
        """Random set of coordinates () won't represent a 'nice' shape).
        """
        return self.GLp.rand()

    def randvec(self, A):
        return self.GLp.randvec(A)

    def zerovec(self):
        """Zero tangent vector in any tangent space.
        """
        return self.GLp.zerovec()

    def geopoint(self, A, B, t):
        return self.GLp.connec.geopoint(A, B, t)
