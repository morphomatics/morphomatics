################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2026 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from __future__ import annotations

import functools
from typing import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

import jax
import jax.numpy as jnp

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['p', 'q'],
                   meta_fields=['kernel'])
@dataclass
class LazyKernel(object):
    """Lazy operations with the kernel matrix."""
    p: jax.Array
    q: jax.Array
    kernel: Callable[[jax.Array, jax.Array], float]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the kernel matrix."""
        return (len(jax.tree.leaves(self.p)[0]), len(jax.tree.leaves(self.q)[0]))

    @property
    def T(self) -> LazyKernel:
        """Transpose of the kernel matrix."""
        return LazyKernel(self.q, self.p, lambda q, p: self.kernel(p, q))

    @functools.partial(jax.jit, static_argnames=['axis'])
    def sum(self, axis: int | None = None) -> jax.Array:
        """Sum of the kernel matrix along the specified axis."""
        K = self.T if axis == 0 else self
        s = LazyKernel.row_sum(K)
        return jax.tree.map(jnp.sum, s) if axis is None else s

    @jax.jit
    def __matmul__(self, v: jax.Array) -> jax.Array:
        return LazyKernel.matmul(self, v)

    @jax.custom_vjp
    @staticmethod
    def row_sum(K: LazyKernel) -> jax.Array:
        """Row-wise sum of the kernel matrix."""
        # tree.map allows use with gradients of kernel functions
        sum = lambda a: jnp.sum(a, axis=-1)
        K_i = lambda p: jax.tree.map(sum, jax.vmap(K.kernel, (None, 0), -1)(p, K.q))
        return jax.lax.map(K_i, K.p, batch_size=1000)

    @staticmethod
    def row_sum_fwd(K: LazyKernel):
        """Forward pass of the row-wise sum of the kernel matrix."""
        return LazyKernel.row_sum(K), K

    @staticmethod
    def row_sum_bwd(K: LazyKernel, g: jax.Array):
        """Backward pass of the row-wise sum of the kernel matrix."""

        pg = {'p': K.p, 'g': g}

        dKp_g = lambda a, q: jax.vjp(K.kernel, a['p'], q)[1](a['g'])[0]
        dp = LazyKernel(pg, K.q, dKp_g).sum(axis=1)

        dKq_g = lambda a, q: jax.vjp(K.kernel, a['p'], q)[1](a['g'])[1]
        dq = LazyKernel(pg, K.q, dKq_g).sum(axis=0)

        return (LazyKernel(dp, dq, K.kernel),)

    @jax.custom_vjp
    @staticmethod
    def matmul(K: LazyKernel, v: jax.Array) -> jax.Array:
        """:return product of kernel matrix with v"""
        # tree.map allows use with gradients of kernel functions
        Kv_i = lambda p: jax.tree.map(lambda a: a @ v, jax.vmap(K.kernel, (None, 0), -1)(p, K.q))
        return jax.lax.map(Kv_i, K.p, batch_size=1000)

    @staticmethod
    def matmul_fwd(K: LazyKernel, v: jax.Array):
        """Forward pass of the kernel matrix multiplication."""
        return K @ v, (K, v)

    @staticmethod
    def matmul_bwd(res, g: jax.Array):
        """Backward pass of the kernel matrix multiplication."""
        K, v = res

        dv = K.T @ g

        pg = {'p': K.p, 'g': g}
        qv = {'q': K.q, 'v': v}

        dKp_gv = lambda a, b: jax.tree.map(
            lambda d: d * jnp.sum(a['g'] * b['v']),
            jax.grad(K.kernel, argnums=0)(a['p'], b['q']))
        dp = LazyKernel(pg, qv, dKp_gv).sum(axis=1)

        dKq_gv = lambda a, b: jax.tree.map(
            lambda d: d * jnp.sum(a['v'] * b['g']),
            jax.grad(K.kernel, argnums=1)(b['p'], a['q']))
        dq = LazyKernel(qv, pg, dKq_gv).sum(axis=1)

        return LazyKernel(dp, dq, K.kernel), dv

LazyKernel.matmul.defvjp(LazyKernel.matmul_fwd, LazyKernel.matmul_bwd)
LazyKernel.row_sum.defvjp(LazyKernel.row_sum_fwd, LazyKernel.row_sum_bwd)

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
    # together: L^T * inv(M) = inv(L)

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
