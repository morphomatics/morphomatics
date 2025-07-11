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

from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp
import numpy as np

from morphomatics.geom.surface import Surface

class LazyKernel(NamedTuple):
    """Lazy operations with the kernel matrix."""
    p: jax.Array
    q: jax.Array
    kernel: Callable[[jax.Array, jax.Array], float]

    def __matmul__(self, v: jax.Array) -> jax.Array:
        """:return product of kernel matrix with v"""
        init = jnp.zeros((len(self.p),) + jnp.atleast_2d(v).shape[1:])
        Ki = jax.vmap(self.kernel, (0, None))
        f = lambda carry, qv: (carry + jnp.einsum('i,...->i...', Ki(self.p, qv[0]), qv[1]), None)
        return jax.lax.scan(f, init, (self.q, v))[0]

def align(src, ref):
    """ (Constrained) Procrustes alignment of src to ref using Kabsch algorithm
    :arg src: n-by-3 array of vertex coordinates of source object
    :arg ref: n-by-3 array of vertex coordinates of reference
    :returns: aligned coords.
    """
    n = len(ref)
    # cross-covariance matrix
    c_s = src.mean(axis=0)
    c_r = ref.mean(axis=0)
    xCov = (ref.T @ src) / n - jnp.outer(c_r, c_s)
    # optimal rotation
    U, S, Vt = jnp.linalg.svd(xCov)
    R = Vt.T @ U.T
    if jnp.linalg.det(R) < 0:
        R = Vt.T @ np.diag([1, 1, -1]) @ U.T
    # return aligned coords.
    return src @ R + (c_r - c_s @ R)


def generalized_procrustes(surf):
    """ Generalized Procrustes analysis.
    :arg surf: list of surfaces to be aligned. The meshes must be in correspondence.
    """
    ref = Surface(jnp.copy(surf[0].v), jnp.copy(surf[0].f))
    old_ref = Surface(jnp.copy(ref.v), jnp.copy(ref.f))

    n_steps = 0
    # do until convergence
    while (jnp.linalg.norm(ref.v - old_ref.v) > 1e-11 and 1000 > n_steps) or n_steps == 0:
        n_steps = n_steps + 1
        old_ref = Surface(jnp.copy(ref.v), jnp.copy(ref.f))
        # align meshes to reference
        for i, s in enumerate(surf):
            s.v = align(s.v, ref.v)

        # compute new reference
        v_ref = jnp.mean(jnp.array([s.v for s in surf]), axis=0)
        ref.v = v_ref


def preshape(v):
    """ Center point cloud at origin and normalize its size
    :arg v: n-by-3 array of vertex coordinates
    :returns: n-by-3 array of adjusted vertex coordinates
    """
    # center
    v = v - 1 / v.shape[0] * jnp.tile(jnp.sum(v, axis=0), (v.shape[0], 1))
    # normalize
    v /= jnp.linalg.norm(v)
    return v


def multiprod(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    # vectorized matrix - matrix multiplication
    return jnp.einsum('...ij,...jk', A, B)


def multitransp(A):
    # vectorized matrix transpose
    return jnp.einsum('...ij->...ji', A)


def multiskew(A):
    return 0.5 * (A - jnp.einsum('...ij->...ji', A))


def multisym(A):
    return 0.5 * (A + jnp.einsum('...ij->...ji', A))


def vectime3d(x, A):
    """
    :param x: vector of length k
    :param A: array of size k x n x m
    :return: k x n x m array such that the j-th n x m slice of A is multiplied with the j-th element of x

    In case of k=1, x * A is returned.
    """
    if jnp.isscalar(x) and A.ndim == 2:
        return x * A

    x = jnp.atleast_2d(x)
    assert x.ndim <= 2 and jnp.size(A.shape) == 3
    assert x.shape[0] == 1 or x.shape[1] == 1
    assert x.shape[0] == A.shape[0] or x.shape[1] == A.shape[0]

    if x.shape[1] == 1:
        x = x.T

    A = jnp.einsum('kij->ijk', A)
    return jnp.einsum('ijk->kij', x * A)


def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A = A.at[:, j].set(A[:, j] - jnp.dot(A[:, k], A[:, j]) * A[:, k])
        A = A.at[:, j].set(A[:, j] / jnp.linalg.norm(A[:, j]))

    return A


def projToGeodesic_flat(metric, X, Y, P, max_iter=10):
    '''
    Specialized version of Metric#projToGeodesic for flat spaces.

    :arg X, Y: start- and endpoint of geodesic X->Y.
    :arg P: point to be projected to X->Y.
    :returns: projection of P to X->Y
    '''

    v = metric.log(X, Y)
    v = v / metric.norm(X, v)

    w = metric.log(X, P)
    d = metric.inner(X, v, w)

    return metric.exp(X, d * v)

def projToGeodesic_group(metric, X, Y, P, max_iter=10):
    '''
    Specialized version of Metric#projToGeodesic for Lie groups, which represent
    tangent vectors in algebra.

    :arg X, Y: start- and endpoint of geodesic X->Y.
    :arg P: point to be projected to X->Y.
    :returns: projection of P to X->Y
    '''

    # all tagent vectors in common tangent space i.e. algebra
    v = metric.log(X, Y)
    v = v / metric.norm(X, v)

    # initial guess
    Pi = X.copy()

    # solver loop
    for _ in range(max_iter):
        w = metric.log(Pi, P)
        d = metric.inner(Pi, v, w)

        # print(f'|<v, w>|={d}')
        if abs(d) < 1e-6: break

        Pi = metric.exp(Pi, d * v)

    return Pi
