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

from functools import partial
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab


class RiemannianNewtonRaphson(object):

    @staticmethod
    @partial(jax.jit, static_argnames=['F'])
    def solve(M: Manifold, F: Callable[[jnp.array], jnp.array], init: jnp.array,
              stepsize=1., maxiter=100, minnorm=1e-6) -> jnp.array:
        """
        Newton-Raphson iteration for solving F(x) = 0.

        :param M: manifold domain
        :param F: fwd.-differentiable (cf. jax.jvp) function mapping x in M to w in TyM (tangent space at y in M)
        :param init: initial guess in M
        :param stepsize: length of step to take towards root of linear model in each iteration
        :param maxiter: maximum number of iterations
        :param minnorm: stop iteration when inf-norm of F(x) is below this value
        :return: Root of F.
        """

        def body(args):
            x, Fx, i = args

            # solve for update direction: v = -J⁻¹F(x)
            J = lambda v: jax.jvp(F, (x,), (v,))[1]
            v, _ = bicgstab(J, -Fx)

            # step
            x = M.connec.exp(x, stepsize*v)

            return x, F(x), i + 1

        def cond(args):
            x, Fx, i = args
            F_norm = jnp.linalg.norm(Fx, np.inf)
            jax.debug.print("F_norm ({}): {}", i, F_norm)
            cnds = jnp.array([F_norm > minnorm, i < maxiter])
            return jnp.all(cnds)

        opt, *_ = jax.lax.while_loop(cond, body, (init, F(init), jnp.array(0)))

        return opt
