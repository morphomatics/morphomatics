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

import jax
import jax.numpy as jnp
import numpy as np

from morphomatics.manifold import Manifold


def pole_ladder(M: Manifold, p: jnp.array, q: jnp.array, v: jnp.array, n_step: int = 1) -> jnp.array:
    """Pole Ladder algorithm to approximate parallel transport along geodesics in affine manifolds
        See

        Numerical Accuracy of Ladder Schemes for Parallel Transport on Manifolds, Nicolas Guigui, Xavier Pennec
        Foundations of Computational Mathematics (2022) 22:757–790,

        for details. The method is exact in Symmetric Spaces.

        :param M: Manifold
        :param p: Point in M
        :param q: Point in M
        :param v: Vector in the tangent space at p
        :param n_step: Number of steps
        :return: Vector in the tangent space at q
    """

    # scaling speeds up convergence
    v = v / n_step**2

    def body(carry, _):
        _P, _p_pr, _i = carry
        _m = _P[_i]
        _q_pr = M.connec.exp(_m, -M.connec.log(_m, _p_pr))

        return (_P, _q_pr, _i+1), None

    U = M.connec.log(p, q)
    t = np.array([i/(2*n_step) for i in range(1, 2*n_step, 2)])
    tU = t.reshape((-1,) + (1,)*U.ndim) * U[None]

    P = jax.vmap(M.connec.exp, (None, 0))(p, tU)
    p_pr = M.connec.exp(p, v)

    (_, q_pr, _), _ = jax.lax.scan(body, (P, p_pr, 0), None, length=n_step)

    return (-1)**n_step * n_step**2 * M.connec.log(q, q_pr)
