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

from typing import List

import jax
import jax.numpy as jnp

try:
    import optax
    def optimize(init_params, fun, max_iter, tol):
        opt = optax.lbfgs(memory_size=100)

        value_and_grad_fun = optax.value_and_grad_from_state(fun)

        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = opt.update(
                grad, state, params, value=value, grad=grad, value_fn=fun
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = optax.tree_utils.tree_get(state, 'count')
            jax.debug.print('Iteration {i}', i=iter_num)
            grad = optax.tree_utils.tree_get(state, 'grad')
            err = optax.tree_utils.tree_linf_norm(grad)
            diff = optax.tree_utils.tree_get(state, 'updates')
            step = optax.tree_utils.tree_linf_norm(diff)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol**.5) & (step >= tol))

        init_carry = (init_params, opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )
        return final_params#, final_state

except ImportError:
    from morphomatics.opt import RiemannianSteepestDescent
    def optimize(x0, fun, max_iter, tol):
        return RiemannianSteepestDescent.fixedpoint(
            Euclidean(x0.shape), fun, x0, stepsize=1e-3, maxiter=max_iter, mingradnorm=tol**.5)

from morphomatics.geom import Surface
from morphomatics.geom.similarity import varifold
from morphomatics.manifold import Diffeomorphism, ProductManifold, Euclidean


class DiffeomorphicRegression:
    """
    A class for performing diffeomorphic regression on a set of shapes.

    Attributes:
        shapes (list): A list of shapes to be regressed.
        parameters (dict): Parameters for the regression.

    Reference:
    Fishbaugh, J., Durrleman, S., Prastawa, M., & Gerig, G. (2017).
    Geodesic shape regression with multiple geometries and sparse parameters.
    Medical image analysis, 39, 1-17.
    """

    def __init__(self, S: Surface, t: jnp.array, shapes: List[Surface], diffeo: Diffeomorphism, momenta: jnp.array):
        """
        Initializes the DiffeomorphicRegression with shapes and optional parameters.

        Args:
            S (Surface): Baseline shape for regression.
            t (jnp.array): Time points for observed shapes.
            shapes (list): Observed shapes to be regressed.
            diffeo (Diffeomorphism): Diffeomorphism specifying deformation model.
            momenta (jnp.array): Momentum vectors for the diffeomorphism.
        """
        self.S = S
        self.t = t
        self.shapes = shapes
        self.diffeo = diffeo
        self.momenta = momenta

    @staticmethod
    def fit(S: Surface, t: jnp.array, shapes: List[Surface], ctrl_pts: jnp.array, momenta: jnp.array,
            scale: float = 0.1, sigma: float = 0.1, gamma: float = 1e-3, max_num_iter: int = 100, tol: float = 1e-6) -> DiffeomorphicRegression:
        """
        Initializes the DiffeomorphicRegression with shapes and optional parameters.

        Args:
            S (Surface): Initial guess for baseline shape for regression.
            t (jnp.array): Time points for observed shapes (must be strictly increasing).
            shapes (list): Observed shapes to be regressed.
            ctrl_pts (jnp.array): Initial guess for control points for the diffeomorphism.
            momenta (jnp.array): Initial guess for  momentum vectors for the diffeomorphism.
            scale (float): Scale of the LDDMM kernel , default is 0.1.
            gamma (float): Regularization parameter for the momenta, default is 1e-3.
        Returns:
            DiffeomorphicRegression: An instance of the DiffeomorphicRegression class.
        """

        # setup search space
        M = ProductManifold([Euclidean(a.shape) for a in (ctrl_pts, momenta, S.v)])

        # regression objective
        def loss(x):
            p, m, v = M.disentangle(x)

            D = Diffeomorphism(p, scale)

            # regularization term (norm of the momenta)
            d = D.metric.inner(None, m, m) * gamma
            jax.debug.print("Regularization term: {d}", d=d)

            # varifold attachment terms
            c = D.group.action(m, v, t)
            for c_, t_, s in zip(c, t, shapes):
                a = varifold(Surface(c_, S.f), s, sigma=sigma)
                jax.debug.print("Varifold distance at time {t_}: {a}", t_=t_, a=a**.5)
                d = d + a

            return d

        # solve regression problem
        x0 = M.entangle((ctrl_pts, momenta, S.v))
        x = optimize(x0, loss, max_num_iter, tol)
        p, m, v = M.disentangle(x)

        return DiffeomorphicRegression(Surface(v, S.f), t[1:], shapes, Diffeomorphism(p, scale), m)
