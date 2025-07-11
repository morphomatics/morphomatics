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

from scipy import sparse
from scipy import spatial
from sklearn.neighbors import NearestNeighbors, KDTree
from .util import *


def to_hat(bases, C):
    Phi1, Phi2 = bases[0], bases[1]
    n1, n2 = Phi1.shape[0], Phi2.shape[0]
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Phi1)
    inds = nn.kneighbors(Phi2 @ C, return_distance=False)
    inds = inds.squeeze()
    return sparse.csr_matrix((np.ones(n2), (np.arange(n2), inds)), shape=(n2, n1))


def to_hat_iso(bases, C):
    Phi1, Phi2 = bases[0], bases[1]
    n1, n2 = Phi1.shape[0], Phi2.shape[0]
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Phi1 @ C.T)
    inds = nn.kneighbors(Phi2, return_distance=False)
    inds = inds.squeeze()
    return sparse.csr_matrix((np.ones(n2), (np.arange(n2), inds)), shape=(n2, n1))


def to_precise(ops, C):
    """ Convert map of functionwise correspondence to precise map """
    s1, s2 = ops[0].surf, ops[1].surf
    n1 = s1.v.shape[0]
    n2 = s2.v.shape[0]
    Phi1, Phi2 = ops[0].evecs, ops[1].evecs
    Phi1 = Phi1[:, :C.shape[1]]
    Phi2 = Phi2[:, :C.shape[0]]
    Phi1_c0 = Phi1[s1.f[:, 0], :]
    Phi1_c1 = Phi1[s1.f[:, 1], :]
    Phi1_c2 = Phi1[s1.f[:, 2], :]
    # compute l_max
    norm_c1c0 = np.linalg.norm(Phi1_c1 - Phi1_c0, axis=1, keepdims=True)
    norm_c2c1 = np.linalg.norm(Phi1_c2 - Phi1_c1, axis=1, keepdims=True)
    norm_c0c2 = np.linalg.norm(Phi1_c0 - Phi1_c2, axis=1, keepdims=True)
    l_max = np.max(np.hstack((norm_c1c0, norm_c2c1, norm_c0c2)), axis=1)
    # compute Delta_min
    tree = KDTree(Phi1)
    dists = tree.query(Phi2 @ C)[0]
    Delta_min = dists.flatten()
    # compute delta_min
    distmat1 = spatial.distance.cdist(Phi1_c0, Phi2 @ C)
    distmat2 = spatial.distance.cdist(Phi1_c1, Phi2 @ C)
    distmat3 = spatial.distance.cdist(Phi1_c2, Phi2 @ C)
    delta_min_all = np.min((distmat1, distmat2, distmat3), axis=0)
    # compute barycentric coordinates
    face_match = np.zeros(n2, dtype=int)
    bary_coord = np.zeros((n2, 3))
    for vertind in range(n2):
        query_faceinds = np.where(delta_min_all[:, vertind] - l_max < Delta_min[vertind])[0]  # (npfi,)
        query_triangles = Phi1[s1.f[query_faceinds], :]  # (npfi, 3)
        query_point = Phi2[vertind, :] @ C  # (npfi, 3, k2)
        dists, proj, bary_coords = project_p2t(query_triangles, query_point, return_bary=True)
        min_ind = dists.argmin()
        face_match[vertind] = query_faceinds[min_ind]
        bary_coord[vertind] = bary_coords[min_ind]
    # build precise map
    v0 = s1.f[face_match, 0]
    v1 = s1.f[face_match, 1]
    v2 = s1.f[face_match, 2]
    ii = np.arange(n2)
    In = np.concatenate([ii, ii, ii])
    Jn = np.concatenate([v0, v1, v2])
    Sn = np.concatenate([bary_coord[:, 0], bary_coord[:, 1], bary_coord[:, 2]])
    return sparse.csr_matrix((Sn, (In, Jn)), shape=(n2, n1))
