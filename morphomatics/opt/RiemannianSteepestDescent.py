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

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold

class RiemannianSteepestDescent(object):

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'f'])
    def fixedpoint(M: Manifold, f: Callable[[jnp.array], float], init: jnp.array,
                   stepsize=1., maxiter=100, mingradnorm=1e-6) -> jnp.array:
        """
        Compute minimizer of f.
        :param M: manifold search space
        :param f: objective function mapping from M to R
        :param init: initial guess in M
        :param stepsize: fixed length of step in steepest descent direction
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm
        :return: Minimizer of f.
        """

        # Gradient
        grad_ = jax.grad(f)
        def grad(x):
            return M.metric.egrad2rgrad(x, grad_(x))

        # optimize
        def body(args):
            x, _, i = args
            g = grad(x)
            g_norm = M.metric.norm(x, g)
            # steepest descent
            x = M.connec.exp(x, -stepsize * g)
            return x, g_norm, i + 1

        def cond(args):
            _, g_norm, i = args
            c = jnp.array([g_norm > mingradnorm, i < maxiter])
            return jnp.all(c)

        opt, *_ = jax.lax.while_loop(cond, body, (init, jnp.array(1.), jnp.array(0)))

        return opt
