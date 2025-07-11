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

import numpy as np
import jax
import jax.numpy as jnp

from scipy import sparse

try:
    from sksparse.cholmod import cholesky as direct_solve
except:
    from scipy.sparse.linalg import factorized as direct_solve

from ..geom import Surface
from . import SO3, SPD
from . import ProductManifold, PowerManifold
from . import ShapeSpace, Metric
from .util import align


class DifferentialCoords(ShapeSpace):
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

        # rotation and stretch manifolds
        self.SPD = PowerManifold(SPD(3), len(self.ref.f))
        self.SO = PowerManifold(SO3(), len(self.ref.f))
        self._M = ProductManifold([self.SO, self.SPD], jnp.asarray(commensuration_weights))

        self.update_ref_geom(self.ref.v)

        name = f'Differential Coordinates Shape Space'
        super().__init__(name, self.M.dim, self.M.point_shape, self.M.connec, self.M.metric, None)

    def tree_flatten(self):
        return (self.M,), (self.ref.v.tolist(), self.ref.f.tolist())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        M = children[0]
        obj = cls(Surface(*aux_data))
        obj._M = M
        obj.SO, obj.SPD = M.manifolds
        return obj

    @property
    def M(self):
        return self._M

    def update_ref_geom(self, v):
        self.ref.v=v

        # center of gravity
        self.CoG = self.ref.v.mean(axis=0)

        # setup Poisson system
        S = self.ref.div @ self.ref.grad
        # add soft-constraint fixing translational DoF
        S += sparse.coo_matrix(([1.0], ([0], [0])), S.shape)  # make pos-def
        self.poisson = direct_solve(S.tocsc())

        # set metric weights
        w = jnp.asarray(self.ref.face_areas)
        self.SO.metric_weights = self.SPD.metric_weights = w


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

        return self.M.entangle([R, U])

    def from_coords(self, c):
        """
        :arg c: differentical coords.
        :returns: #v-by-3 array of vertex coordinates
        """
        # compose
        R, U = self.M.disentangle(c)
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

    def rand(self, key: jax.Array):
        return self.M.rand(key)

    def zerovec(self):
        """Returns the zero vector in any tangent space."""
        return self.M.zerovec()

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
