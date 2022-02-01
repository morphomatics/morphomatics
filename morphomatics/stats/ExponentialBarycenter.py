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


import numpy as np
from joblib import Parallel, delayed
from joblib import parallel_backend


from morphomatics.manifold import Manifold
#from pymanopt.solvers import SteepestDescent
#from pymanopt import Problem


class ExponentialBarycenter(object):
    """
    Exponential barycenter, see e.g.
    Pennec and Arsigny (2012): Exponential Barycenters of the Canonical Cartan Connection and Invariant Means on Lie Groups.

    The barycenter will be a bi-invariant notion of mean in the Lie group setting and the Frechét mean for Riemamnnian manifolds.
    (For the special case of a bi-invariant metric, both notions will agree.)
    """

    @staticmethod
    def compute(mfd: Manifold, data, x=None, max_iter=10, n_jobs=-1):
        """
        :arg mfd: data space in which mean is computed
        :arg data: list of data points
        :arg x: initial guess
        :returns: mean of data, i.e. exp. barycenter thereof
        """
        assert mfd.connec

        # initial guess
        if x is None:
            x = data[0].copy() #TODO: better guess -> choose most central sample

        # compute intrinsic mean
        #cost = lambda a: 0.5 / len(data) * np.sum([mfd.metric.dist(a, b) ** 2 for b in data])
        #grad = lambda a: np.sum([mfd.connec.log(a, b) for b in data], axis=0) / len(data)
        # hess = lambda a, b: b
        # problem = Problem(manifold=mfd, cost=cost, grad=grad, hess=hess, verbosity=2)
        # x = SteepestDescent(maxiter=max_iter).solve(problem, x=x)

        # Newton-type fixed point iteration
        with Parallel(n_jobs=n_jobs, prefer='threads', verbose=0) as parallel:
            grad = lambda a: np.sum(parallel(delayed(mfd.connec.log)(a, b) for b in data), axis=0) / len(data)
            for _ in range(max_iter):
                g = grad(x)
                if mfd.metric:
                    g_norm = mfd.metric.norm(x, -g)
                else:
                    g_norm = np.linalg.norm(-g)
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

        return np.sum([mfd.metric.dist(x, y) ** 2 for y in data]) / len(data)
