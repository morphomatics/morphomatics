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

from tqdm import tqdm
from .convert import *


def icp(surfs, bases, C12_init, n_iter):
    s1, s2 = surfs
    Phi1, Phi2 = bases
    M2 = sparse.diags(s2.vertex_areas_barycentric)
    Phi2_pseudoinv = Phi2.T @ M2
    n1, k1 = Phi1.shape
    n2, k2 = Phi2.shape
    C12 = C12_init
    inds = None
    for i in range(n_iter):
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Phi1[:, :k1])
        inds = nn.kneighbors(Phi2[:, :k2] @ C12, return_distance=False)
        inds = inds.squeeze()
        P12 = sparse.csr_matrix((np.ones(n2), (np.linspace(0, n2 - 1, n2), inds)), shape=(n2, n1))
        U, _, VT = np.linalg.svd(Phi2_pseudoinv @ P12 @ Phi1)
        C12 = U @ sparse.eye(k2, k1) @ VT
    return C12, inds


def zoomout(ops, C, n_steps, steps, convert=False):
    k_t, k_s = C.shape
    step_s, step_t = steps
    for i in range(n_steps):
        if convert == 'iso':
            P = to_hat_iso([ops[0].evecs[:, :k_s], ops[1].evecs[:, :k_t]], C)
        elif convert == 'precise':
            P = to_precise(ops, C)
        else:
            P = to_hat([ops[0].evecs[:, :k_s], ops[1].evecs[:, :k_t]], C)
        k_s, k_t = k_s + step_s, k_t + step_t
        C = ops[1].evecs_inv[:k_t, :] @ P @ ops[0].evecs[:, :k_s]
    return C


def zoomout_consistent(ops, adj, C, steps, p_lat, tol_lat=-1e-6):
    n = len(ops)
    n_edges = len(C)
    k = len(C[0])
    for step in tqdm(steps):
        Id = np.tile(np.eye(k), (n_edges, 1, 1))
        weights = adj.data[:, None, None]
        n_lat = int(p_lat * k)
        W = sparse.bsr_matrix((weights * C, adj.row, np.arange(n_edges + 1)))\
            - sparse.bsr_matrix((weights * Id, adj.col, np.arange(n_edges + 1)))
        W = W.T @ W
        eigs, Y = sparse.linalg.eigsh(W, k=n_lat, sigma=tol_lat)
        Y = Y.reshape((adj.shape[0], k, n_lat))
        E = np.einsum('hji,hj,hjk', Y, [ops[i].evals[:k] for i in range(n)], Y)
        Lambda_0, U = np.linalg.eigh(E)
        Y = np.einsum('...ij,jk', Y, U)
        k_new = k + step
        for e, (i, j) in enumerate(zip(adj.row, adj.col)):
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(ops[i].evecs[:, :k] @ Y[i])
            inds_ij = nn.kneighbors(ops[j].evecs[:, :k] @ Y[j], return_distance=False)
            inds_ij = inds_ij.squeeze()
            ni, nj = len(ops[i].evecs), len(ops[j].evecs)
            Pij = sparse.csr_matrix((np.ones(nj), (np.arange(nj), inds_ij)), shape=(nj, ni))
            Cij = ops[j].evecs_inv[:k_new, :] @ Pij @ ops[i].evecs[:, :k_new]
            C[e] = Cij
        k = k_new
    Id = np.tile(np.eye(k), (n_edges, 1, 1))
    weights = adj.data[:, None, None]
    n_lat = int(p_lat * k)
    W = sparse.bsr_matrix((weights * C, adj.row, np.arange(n_edges + 1))) \
        - sparse.bsr_matrix((weights * Id, adj.col, np.arange(n_edges + 1)))
    W = W.T @ W
    eigs, Y = sparse.linalg.eigsh(W, k=n_lat, sigma=tol_lat)
    Y = Y.reshape((adj.shape[0], k, n_lat))
    E = np.einsum('hji,hj,hjk', Y, [ops[i].evals[:k] for i in range(n)], Y)
    Lambda_0, U = np.linalg.eigh(E)
    Y = np.einsum('...ij,jk', Y, U)
    return Lambda_0, Y, C
