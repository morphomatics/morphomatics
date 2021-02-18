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

from pymanopt.manifolds.manifold import Manifold
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
    def compute(mfd: Manifold, data, x=None, max_iter=10):
        """
        :arg mfd: data space in which mean is computed
        :arg data: list of data points
        :arg x: initial guess
        :returns: mean of data, i.e. exp. barycenter thereof
        """
        # initial guess
        if x is None:
            x = data[0].copy() #TODO: better guess -> choose most central sample

        # compute intrinsic mean
        cost = lambda a: 0.5 / len(data) * np.sum([mfd.dist(a, b) ** 2 for b in data])
        grad = lambda a: np.sum([mfd.log(a, b) for b in data], axis=0) / len(data)
        # hess = lambda a, b: b
        # problem = Problem(manifold=mfd, cost=cost, grad=grad, hess=hess, verbosity=2)
        # x = SteepestDescent(maxiter=max_iter).solve(problem, x=x)

        # Gauss-Newton type solver
        for _ in range(max_iter):
            g = grad(x)
            g_norm = mfd.norm(x, -g)
            print(f'|grad|={g_norm}')
            if g_norm < 1e-6: break
            x = mfd.exp(x, g)

        return x

    @staticmethod
    def total_variance(mfd: Manifold, data, x=None):
        """
        :arg mfd: data space in which mean is computed
        :arg data: samples
        :arg x: center
        :returns: total variance
        """
        if x is None:
            x = ExponentialBarycenter.compute(mfd, data)

        return np.sum([mfd.dist(x, y) ** 2 for y in data]) / len(data)
