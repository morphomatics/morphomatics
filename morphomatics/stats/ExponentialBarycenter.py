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


import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold

class ExponentialBarycenter(object):
    """
    Exponential barycenter, see e.g.
    Pennec and Arsigny (2012): Exponential Barycenters of the Canonical Cartan Connection and Invariant Means on Lie Groups.

    The barycenter will be a bi-invariant notion of mean in the Lie group setting and the Frechét mean for Riemamnnian manifolds.
    (For the special case of a bi-invariant metric, both notions will agree.)
    """

    @staticmethod
    def compute(mfd: Manifold, data, x=None, max_iter=10):
        """
        :arg mfd: data space in which mean is computed
        :arg data: array of data points
        :arg x: initial guess
        :returns: mean of data, i.e. exp. barycenter thereof
        """
        assert mfd.connec

        # initial guess
        if x is None:
            x = data[0].copy() #TODO: better guess -> choose most central sample

        # Newton-type fixed point iteration
        grad = jax.jit(lambda a: jnp.sum(jax.vmap(lambda b: mfd.connec.log(a, b))(data), axis=0) / len(data))
        for _ in range(max_iter):
            g = grad(x)
            if mfd.metric:
                g_norm = mfd.metric.norm(x, -g)
            else:
                g_norm = jax.numpy.linalg.norm(-g)
            print(f'|grad|={g_norm}')
            if g_norm < 1e-12: break
            x = mfd.connec.exp(x, g)

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
