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

# postponed evaluation of annotations to circumvent cyclic dependencies (will be default behavior in Python 4.0)
from __future__ import annotations

import jax
import jax.numpy as jnp


class GeometricMedian(object):
    """
    Geometric median (a robust estimator of centrality for data) on Riemannian manifolds.
    """

    @staticmethod
    @jax.jit
    def compute(mfd: Manifold, data: jnp.ndarray, x=None, w: jnp.ndarray=None, max_iter: int=10):
        """
        Compute the geometric median of data points on a Riemannian manifold.

        Uses Weizfeld's algorithm for manifolds as proposed in:
        Fletcher, P. T., Venkatasubramanian, S., & Joshi, S. (2009).
        The geometric median on Riemannian manifolds with application to robust atlas estimation.
        NeuroImage, 45(1), S143-S152.

        :arg mfd: data space in which mean is computed
        :arg data: array of data points
        :arg x: initial guess
        :arg w: weights
        :arg max_iter: maximal number of iterations
        :returns: median of data
        """
        assert mfd.connec

        # initial guess
        if x is None:
            x = data[0].copy()

        # init weights
        if w is None:
            w = jnp.ones(len(data))
        # ensure sum(w) = 1
        w = w / jnp.sum(w)

        # Weizfeld's algorithm for manifolds

        # step size in (0, 2)
        alpha = 1.0

        def body(args):
            x, g_sqnorm, i = args

            logs = jax.vmap(mfd.connec.log, (None, 0))(x, data)
            sq_dists = jax.vmap(mfd.metric.inner, (None, 0, 0))(x, logs, logs)
            # caution: jnp.where not NaN-safe with reverse-mode AD
            # anyway: while_loop supports only forward-mode AD
            a = jnp.where(sq_dists > 1e-6, w / jnp.sqrt(sq_dists), 0)

            # gradient
            g = jnp.sum(a.reshape((-1,)+(1,)*x.ndim) * logs, axis=0)
            g_sqnorm = mfd.metric.inner(x, g, g)

            # step size (incl. gradient scaling)
            delta = alpha / jnp.sum(a)

            # update
            x = mfd.connec.exp(x, delta * g)
            return (x, g_sqnorm, i+1)

        def cond(args):
            _, g_sqnorm, i = args
            # gradient norm always <= 1
            c = jnp.array([g_sqnorm > 1e-6, i < max_iter])
            return jnp.all(c)

        x, g_norm, i = jax.lax.while_loop(cond, body, (x, jnp.array(1.), jnp.array(0)))
        # print(g_norm, i)

        return x
