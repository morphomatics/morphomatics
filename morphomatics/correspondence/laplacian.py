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
from scipy import sparse
from scipy.sparse import linalg
from ..geom import Surface

try:
    from robust_laplacian import mesh_laplacian
except ImportError:
    def mesh_laplacian(v, f):
        s = Surface(v,f)
        M = sparse.diags(s.vertex_areas_barycentric)
        W = s.div @ s.grad
        return W, M


class LaplaceBeltrami(object):

    def __init__(self, surf):
        self.surf = surf
        self.mass = None
        self.n_eig = None
        self.evals = None
        self.evecs = None

    @property
    def evecs_inv(self):
        return self.evecs.T @ self.mass

    def mass_fe(self):
        n = len(self.surf.v)
        faces = self.surf.f
        face_areas = self.surf.face_areas
        Me1 = sparse.csr_matrix((face_areas / 12, (faces[:, 1], faces[:, 2])), (n, n))
        Me2 = sparse.csr_matrix((face_areas / 12, (faces[:, 2], faces[:, 0])), (n, n))
        Me3 = sparse.csr_matrix((face_areas / 12, (faces[:, 0], faces[:, 1])), (n, n))
        ind = np.hstack(faces.T)
        Mii = sparse.csr_matrix((np.concatenate((face_areas, face_areas, face_areas)) / 6, (ind, ind)), (n, n))
        self.mass = Me1 + Me1.T + Me2 + Me2.T + Me3 + Me3.T + Mii

    def eig(self):
        W, M = mesh_laplacian(np.asarray(self.surf.v), np.asarray(self.surf.f))
        if self.mass:
            self.evals, self.evecs = sparse.linalg.eigsh(W, self.n_eig, self.mass, sigma=-1e-8)
        else:
            self.mass = M
            # solve S * v = lambda * M * v with change of variables u = M^.5 * v
            sqrtMinv = sparse.diags(1 / np.sqrt(M.data))
            self.evals, evecs = sparse.linalg.eigsh(sqrtMinv @ W @ sqrtMinv, self.n_eig, sigma=-1e-8)
            self.evecs = sqrtMinv @ evecs
