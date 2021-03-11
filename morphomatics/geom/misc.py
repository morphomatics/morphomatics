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

import functools

import numpy as np
from scipy import sparse

def memoize(cache_name):
    """Helper decorator memoizes the given zero-argument function.
    Really helpful for memoizing properties so they don't have to be recomputed
    dozens of times.
    """
    def memo_decorator(fn):
        @functools.wraps(fn)
        def memofn(self, *args, **kwargs):
            cache = getattr(self, cache_name, None)
            if id(fn) not in cache:
                cache[id(fn)] = fn(self)
            return cache[id(fn)]

        return memofn
    return memo_decorator

def gradient_matrix_ambient(verts, cells):
    """
    Compute gradient (represented in ambient space) matrix for Lagrange basis
    on k-manifold simplicial geom with vertices \a verts and k-simplices \a cells
    :return: sparse (d*m)-by-n gradient matrix, where d is dim. of vertices,
        and m (n) is the number of triangles (vertices).
    """
    n = len(verts)
    m = len(cells)
    d = verts.shape[1]
    k = cells.shape[1]-1

    E = [verts[cells[:,i]] - verts[cells[:,k]] for i in range(k)]
    M = np.matmul(np.stack(E, axis=1), np.stack(E, axis=2))
    # TODO: use solve() instead of inv()
    Minv = np.linalg.inv(M)
    EMinv = np.matmul(np.stack(E, axis=2), Minv)
    partials = np.zeros((k,k+1))
    partials[:k,:k] = np.eye(k)
    partials[:,k] = -1
    # TODO: use np.einsum s.t. we don't need np.tile
    D = np.matmul(EMinv, np.tile(partials, (m,1,1))).ravel()

    I = np.repeat(np.arange(d*m), k+1)
    J = np.repeat(cells, d, axis=0).ravel()
    return sparse.csr_matrix((D, (I, J)), shape=(d*m, n))

def gradient_matrix_local(verts, cells):
    """
    Compute gradient matrix for Lagrange basis on d-manifold simplicial geom
    with vertices \a verts and d-simplices \a cells.
    Gradients will be represented in (d-dim.) local chart of each simplex.
    :return: sparse (d*m)-by-n gradient matrix, where m (n) is the number of triangles (vertices),
             and volumes of d-simplices
    """
    n = len(verts)
    m = len(cells)
    d = cells.shape[1] - 1

    E = [verts[cells[:, i]] - verts[cells[:, d]] for i in range(d)]
    # metric
    M = np.matmul(np.stack(E, axis=1), np.stack(E, axis=2))
    # (lower) cholesky factor of M
    L = np.linalg.cholesky(M)

    # partial derivatives for reference simplex
    partials = np.zeros((d, d + 1))
    partials[:d, :d] = np.eye(d)
    partials[:, d] = -1

    # gradient = inv(M)*partials
    # change of variables: x -> L^T*x (s.t. M-inner product becomes standard one)
    # togehter: L^T * inv(M) = inv(L)

    # unroll forward substitution (no array-wise solve in numpy)
    D = np.tile(partials, (m, 1))
    for i in range(d):
        for j in range(i):
            D[i::2] -= D[j::d] * L.ravel()[i * d + j::d ** 2, None]
        D[i::d] /= L.ravel()[i * d + i::d ** 2, None]

    # set up gradient matrix
    I = np.repeat(np.arange(d * m), d + 1)
    J = np.repeat(cells, d, axis=0).ravel()
    grad = sparse.csr_matrix((D.ravel(), (I, J)), shape=(d * m, n))

    # volumes of d-dimplices (computing sqrt. of det(M) re-using L)
    factorial = lambda d: np.prod(range(1, d + 1))
    vol = np.diagonal(L, axis1=1, axis2=2).prod(axis=1) / factorial(d)

    return grad, vol
