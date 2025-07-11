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


class ExponentialBarycenter(object):
    """
    Exponential barycenter, see e.g.
    Pennec and Arsigny (2012): Exponential Barycenters of the Canonical Cartan Connection and Invariant Means on Lie Groups.

    The barycenter will be a bi-invariant notion of mean in the Lie group setting and the Frechét mean for Riemamnnian manifolds.
    (For the special case of a bi-invariant metric, both notions will agree.)
    """

    @staticmethod
    @jax.jit
    def compute(mfd: Manifold, data, x=None, max_iter=10):
        """
        :arg mfd: data space in which mean is computed
        :arg data: array of data points
        :arg x: initial guess
        :arg max_iter: maximal number of iterations
        :returns: mean of data, i.e. exp. barycenter thereof
        """
        assert mfd.connec

        # initial guess
        if x is None:
            x = data[0].copy() #TODO: better guess -> choose most central sample

        # Newton-type fixed point iteration

        def body(args):
            x, g_norm, i = args
            g = jnp.sum(jax.vmap(mfd.connec.log, (None, 0))(x, data), axis=0) / len(data)
            g_norm = jax.numpy.linalg.norm(g) if mfd.metric is None else mfd.metric.norm(x, g)
            x = mfd.connec.exp(x, g)
            return (x, g_norm, i+1)

        def cond(args):
            _, g_norm, i = args
            c = jnp.array([g_norm > 1e-6, i < max_iter])
            return jnp.all(c)

        x, g_norm, i = jax.lax.while_loop(cond, body, (x, jnp.array(1.), jnp.array(0)))
        # print(g_norm, i)

        return x

    @staticmethod
    def total_variance(mfd: Manifold, data, x=None):
        """
        :arg mfd: data space in which mean is computed
        :arg data: samples
        :arg x: center
        :returns: total variance
        """
        assert mfd.connec and mfd.metric

        if x is None:
            x = ExponentialBarycenter.compute(mfd, data)

        #return np.sum([mfd.metric.dist(x, y) ** 2 for y in data]) / len(data)
        return jnp.sum(jax.vmap(lambda y: mfd.metric.dist(x, y) ** 2)(data)) / len(data)
