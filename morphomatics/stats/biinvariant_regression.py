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

from __future__ import annotations

import jax
import jax.numpy as jnp
from morphomatics.geom import BezierSpline
from morphomatics.stats import BiinvariantDissimilarityMeasures


class BiinvariantRegression(object):
    """
    Regression for the estimation of a relationship between single explanatory and Lie group-valued dependent variable.
    The result is  equivariant under left and right translation of the data.
    The relationship is modeled via geodesics of the canoncial Cartan-Schouten connection.

    For details about the method see

                    Johannes Schade, Christoph von Tycowicz, Martin Hanik:
                    Bi-invariant Geodesic Regression with Data from the Osteoarthritis Initiative.
                    Proc. of Information Processing in Medical Imaging (IPMI), 2025.

    """

    def __init__(self, G: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array = None,
                 residual_scale: jnp.array = None, max_iter: int = 100, min_norm: float = 1e-6, step_size: float = 0.1,
                 verbose: bool = True):
        """
        :param G: Lie group
        :param Y: array containing G-valued data
        :param param: vector of scalar parameters corresponding to the data in Y
        :param P_init: initial guess for the start and endpoint of the geodesic
        :param residual_scale: optional rescaling factors
        :param max_iter: see the fit-method
        :param min_norm: see the fit-method
        :param step_size: see the fit-method
        :param verbose: warns that min_norm is not reached if true
        """
        assert Y.shape[0] == param.shape[0]

        self._G = G
        self._Y = Y
        self._param = param

        # initial guess
        if P_init is None:
            P_init = jnp.array([Y[0], Y[-1]])

        if residual_scale is None:
            residual_scale = jnp.ones_like(param)

        # fit geodesic to data
        P, self.conv, e = BiinvariantRegression.fit(G, Y, param, P_init, residual_scale,
                                                    max_iter, min_norm, step_size)

        # jax.debug.print("final error: {}".format(e))
        if verbose and not self.conv:
            print(f'No convergence WARNING: final error {e:.2E} is higher than min_norm {min_norm:.2E}.')

        self._geodesic = BezierSpline(G, P[None])


    @staticmethod
    @jax.jit
    def fit(G: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array, residual_scale: jnp.array, max_iter: int,
            min_norm: float, step_size: float) -> jnp.array:
        """Fit a geodesic to given data.

        :param G: Lie group
        :param Y: array containing G-valued data
        :param param: vector of scalar parameters corresponding to the data in Y
        :param P_init: initial guess for the start and endpoint of the geodesic
        :param residual_scale: re-scaling factors to weigh contributions a an update vector
        :param max_iter: maximum number of iterations
        :param min_norm: threshold determining the convergence of the optimization
        :param step_size: factor with which the update direction is weighted
        :return P: array containing the start and endpoint of the optimal geodesic
        """

        # jax.debug.print("param: {}", param)
        # jax.debug.print("Y: {}", jnp.sum(Y))
        # jax.debug.print("residual_scale: {}", residual_scale)

        reparam_start = jnp.where(jnp.isclose(param, jnp.ones_like(param)), 0., 1 / (1 - param))
        reparam_end = jnp.where(jnp.isclose(param, jnp.zeros_like(param)), 0., 1 / param)

        def step(args):
            P, _, i = args

            # evaluate the current geodesic at the parameters in param and compute the residuals
            gamma_t = jax.vmap(G.connec.geopoint, in_axes=(None, None, 0))(P[0], P[1], param)
            eps = jax.vmap(G.connec.log)(gamma_t, Y)

            # compute start- and endpoint derivatives at points gamma_t in directions of eps at times param
            J_s = jax.vmap(G.connec.dygeo, in_axes=(None, 0, 0, 0))(P[1], gamma_t, reparam_start, eps)
            J_e = jax.vmap(G.connec.dygeo, in_axes=(None, 0, 0, 0))(P[0], gamma_t, reparam_end, eps)

            # compute update vectors
            update_start = jnp.einsum('i...,i', J_s, (1 - param) ** 2 * residual_scale)
            update_end = jnp.einsum('i...,i', J_e, param ** 2 * residual_scale)

            # update endpoints
            P = P.at[0].set(G.connec.exp(P[0], step_size * update_start))
            P = P.at[1].set(G.connec.exp(P[1], step_size * update_end))

            return P, jnp.sqrt(jnp.linalg.norm(update_start)**2 + jnp.linalg.norm(update_end)**2), i + 1

        def condition(args):
            _, err, i = args
            c = jnp.array([i < max_iter, err > min_norm])
            return jnp.all(c)

        opt, err, iter = jax.lax.while_loop(condition, step, (P_init, jnp.inf, 0))

        conv = err <= min_norm

        return opt, conv, err


    @property
    def trend(self) -> BezierSpline:
        """
        :return: estimated geodesic encoding the relationship between the explanatory and manifold-valued dependent
        variable.
        """
        return self._geodesic


class BiinvariantLocalGeodesicRegression(object):

    def __init__(self, G: Manifold, Y: jnp.array, param: jnp.array, kernel: str = "gauss", bandwidth: float = 0.2,
                 max_iter: int = 500, min_norm: float = 1e-6, step_size: float = 0.1):

        self._G = G
        self._Y = Y
        self._param = param

        if kernel == "gauss":
            self._kernel = lambda s: gauss_kernel(s/bandwidth)
        elif kernel == "cauchy":
            self._kernel = lambda s: cauchy_kernel(s/bandwidth)
        elif kernel == "picard":
            self._kernel = lambda s: picard_kernel(s/bandwidth)

        # hyperparameters for geodesic regression
        self.max_iter = max_iter
        self.min_norm = min_norm
        self.step_size = step_size

    def eval(self, t: float) -> jnp.array:
        res_factors = self._kernel(jnp.abs(t - self._param))
        geodesic_regression = BiinvariantRegression(self._G, self._Y, self._param, None,
                                                    res_factors,
                                                    self.max_iter, self.min_norm, self.step_size, False)

        return geodesic_regression.trend.eval(t)


def R2statistic(reg: BiinvariantRegression | BiinvariantLocalGeodesicRegression) -> float:
    """ Prototype of a bi-invariant R2 statistic

    :return: generalized R^2 statistic
    """
    BDM = BiinvariantDissimilarityMeasures(reg._G, variant="left")

    C, mean = BDM.centralized_sample_covariance(reg._Y)

    def squared_mahalanobis(v):
        v = reg._G.group.coords(v)
        Cinv_y = jnp.linalg.solve(C, v)
        d = v.T @ Cinv_y
        return d[0,0]

    evaluate = reg.eval if isinstance(reg, BiinvariantLocalGeodesicRegression) else reg._geodesic.eval

    gam_j = jax.vmap(evaluate)(reg._param)
    residuals = jax.vmap(BDM.diff_at_e)(reg._Y, gam_j)

    residual_var = jax.vmap(squared_mahalanobis)(residuals).mean()
    total_var = jax.vmap(squared_mahalanobis)(reg._Y).mean()

    return float(1 - residual_var / total_var)

# kernels functions

def gauss_kernel(par: jnp.array) -> jnp.array:
    return jax.vmap(lambda t: 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-t ** 2 / 2))(par)

def cauchy_kernel(par: jnp.array) -> jnp.array:
    return jax.vmap(lambda t: 1 / (jnp.pi * (1 + t ** 2)))(par)

def picard_kernel(par: jnp.array) -> jnp.array:
    return jax.vmap(lambda t: 1 / 2 * jnp.exp(-jnp.abs(t)))(par)
