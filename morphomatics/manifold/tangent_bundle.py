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

from typing import Callable, List


from morphomatics.manifold import Manifold, Metric


class TangentBundle(Manifold):
    """Tangent Bundle TM of a smooth manifold M

        Elements of TM are modelled as arrays of the form [2, M.point_shape]
    """

    def __init__(self, M: Manifold, structure: str = 'Sasaki'):
        point_shape = tuple([2, *M.point_shape])
        name = 'Tangent bundle of ' + M.__str__() + '.'
        dimension = 2 * M.dim
        super().__init__(name, dimension, point_shape)
        self._base_manifold = M
        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children+(self.base_manifold,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, M = children
        obj = cls(M, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    @property
    def base_manifold(self) -> Manifold:
        """Return the base manifold of whose tangent bundle is TM """
        return self._base_manifold

    def bundle_projection(self, pu: jnp.array) -> jnp.array:
        """Canonical projection of the tangent bundle onto its base manifold"""
        return pu[0]

    def initSasakiStructure(self):
        """
        Instantiate the tangent bundle with Sasaki structure.
        """
        structure = TangentBundle.SasakiStructure(self)
        self._metric = structure
        self._connec = structure

    def rand(self, key: jax.Array) -> jnp.array:
        """Random element of TM"""
        k1, k2 = jax.random.split(key, 2)
        p = self._base_manifold.rand(k1)
        u = self._base_manifold.randvec(k2)
        return jnp.stack((p, u))

    def randvec(self, pu: jnp.array, key: jax.Array) -> jnp.array:
        """Random vector in the tangent space of the point pu

        :param pu: element of TM
        :return: tangent vector at pu
        """
        k1, k2 = jax.random.split(key, 2)
        p = self.bundle_projection(pu)
        return jnp.stack((self._base_manifold.randvec(p, k1), self._base_manifold.randvec(p, k2)))

    def zerovec(self) -> jnp.array:
        """Zero vector in any tangen space
        """
        return jnp.zeros(self.point_shape)

    def proj(self, pu: jnp.array, vw: jnp.array) -> jnp.array:
        raise NotImplementedError('This function has not been implemented yet.')

    class SasakiStructure(Metric):
        """
        This class implements the Sasaki metric: The natural metric on the tangent bundle TM of a Riemannian manifold M.

        The Sasaki metric is characterized by the following three properties:
         * the canonical projection of TM becomes a Riemannian submersion,
         * parallel vector fields along curves are orthogonal to their fibres, and
         * its restriction to any tangent space is Euclidean.

        Geodesic computations are realized via a discrete formulation of the geodesic equation on TM that involve
        geodesics, parallel translation, and the curvature tensor on the base manifold M; for details see
            Muralidharan, P., & Fletcher, P. T. (2012, June).
            Sasaki metrics for analysis of longitudinal data on manifolds.
            In 2012 IEEE conference on computer vision and pattern recognition (pp. 1027-1034). IEEE.
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4270017/
        """

        def __init__(self, TM: Manifold, Ns: int = 10):
            """
            Constructor.

            :param TM: TangentBundle object
            :param Ns: scalar that determines the number of discretization steps used in the approximation of the
            exponential and logarithm maps
            """
            self._TM = TM
            self.Ns = Ns

        def __str__(self) -> str:
            return "Sasaki structure"

        @property
        def typicaldist(self):
            raise NotImplementedError('This function has not been implemented yet.')

        def inner(self, pu: jnp.array, vw: jnp.array, xy: jnp.array) -> float:
            """Inner product between two tangent vectors at point in TM.

            :param pu: element of _TM
            :param vw: tangent vector at pv
            :param xy: tangent vector at pv
            :return: inner product (scalar) of vw and xy
            """
            p = self._TM.bundle_projection(pu)
            # compute Sasaki inner product via metric of the underlying manifold
            base_metric = self._TM.base_manifold.metric.inner
            return base_metric(p, vw[0], xy[0]) + base_metric(p, vw[1], xy[1])

        def flat(self, pu: jnp.array, xy: jnp.array) -> jnp.array:
            """Lower vector xy at pu with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, pu: jnp.array, d_xy: jnp.array) -> jnp.array:
            """Raise covector d_xy at pu with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def egrad2rgrad(self, pu: jnp.array, vw: jnp.array) -> jnp.array:
            raise NotImplementedError('This function has not been implemented yet.')

        def geodesic_discrete(self, pu: jnp.array, qr: jnp.array) -> List[jnp.array]:
            """
            Compute Sasaki geodesic employing a variational time discretization.

            :param pu: element of TM
            :param qr: element of TM
            :return: array-like, shape=[n_steps + 1, 2, M.shape]
                Discrete geodesic x(s)=(p(s), u(s)) in Sasaki metric connecting
                pu = x(0) and qr = x(1).
            """

            Ns = self.Ns
            base_connection = self._TM.base_manifold.connec
            par_trans = base_connection.transp
            p0, u0 = pu[0], pu[1]
            pL, uL = qr[0], qr[1]

            def grad(c):
                """ Gradient of path energy for discrete geodesic c """
                def grad_i(puP, pu, puN):
                    v, w = base_connection.log(pu[0], puN[0]), par_trans(puN[0], pu[0], puN[1]) - pu[1]
                    gp = .5 * (v + base_connection.log(pu[0], puP[0])) \
                              + base_connection.curvature_tensor(pu[0], pu[1], w, v)
                    gu = w + par_trans(puP[0], pu[0], puP[1]) - pu[1]
                    return jnp.array([gp, gu])
                return -Ns * jax.vmap(grad_i)(c[:-2], c[1:-1], c[2:])

            # Initial guess for gradient_descent
            vw = base_connection.log(p0, pL)
            s = jnp.linspace(0., 1., Ns + 1)
            def init(t):
                p_ini = base_connection.exp(p0, t * vw)
                u_ini = (1 - t) * par_trans(p0, p_ini, u0) + t * par_trans(pL, p_ini, uL)
                return jnp.array([p_ini, u_ini])
            pu_ini = jax.vmap(init)(s[1:-1])
            pu_ini = jnp.vstack([pu[None], pu_ini, qr[None]])

            # Minimization by gradient descent
            x = _gradient_descent(self._TM, pu_ini, grad)
            # x = _gradient_descent(pu_ini, grad, self.exp)

            return x

        def exp(self, pu: jnp.array, vw: jnp.array) -> jnp.array:
            """Compute the exponential of the Levi-Civita connection of the Sasaki metric.

            Exponential map at pv of uw computed by
            shooting a Sasaki geodesic using an Euler integration in TM.

            :param pu: element of TM
            :param vw: tangent vector at pv
            :return: point at time 1 of the geodesic that starts in pu with initial velocity vw
            """

            base_connection = self._TM.base_manifold.connec
            par_trans = self._TM.base_manifold.connec.transp
            Ns = self.Ns
            eps = 1 / Ns

            def body(carry, _):
                p, u, v, w = carry
                p_ = base_connection.exp(p, eps * v)
                u_ = par_trans(p, p_, u + eps * w)
                v_ = par_trans(p, p_, v - eps * base_connection.curvature_tensor(p, u, w, v))
                w_ = par_trans(p, p_, w)
                return (p_, u_, v_, w_), None

            (p, u, *_), _ = jax.lax.scan(body, (*pu, *vw), jnp.empty(Ns))

            return jnp.stack([p, u])

        def log(self, pu: jnp.array, qr: jnp.array) -> jnp.array:
            """Compute the logarithm of the Levi-Civita connection of the Sasaki metric.

            Logarithmic map at base_point p of pu computed by
            iteratively relaxing a discretized geodesic between pu and qw.

            For a derivation of the algorithm see https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/8717.

            :param pu: element of TM
            :param qr: element of TM
            :return: tangent vector at pu (inverse of exp)
            """

            base_connection = self._TM.base_manifold.connec
            par_trans = base_connection.transp
            Ns = self.Ns

            def do_log(bs_pt, pt):
                pu = self.geodesic_discrete(bs_pt, pt)
                p1, u1 = pu[1][0], pu[1][1]
                p0, u0 = bs_pt[0], bs_pt[1]
                w = par_trans(p1, p0, u1) - u0
                v = base_connection.log(p0, p1)
                return Ns * jnp.array([v, w])

            return do_log(pu, qr)

        def curvature_tensor(self, pu, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at pu on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            # TODO: Implement from "Curvature of the Induced Riemannian Metric on the Tangent Bundle of a Riemannian
            #  Manifold" (1971) by Kowalski
            raise NotImplementedError('This function has not been implemented yet.')

        def geopoint(self, pu: jnp.array, qr: jnp.array, t: float) -> jnp.array:
            """Evaluate geodesic in TM

            :param pu: element of TM
            :param qr: element of TM
            :param t: scalar between 0 and 1
            :return: element of the geodesic between pv and pu evaluated at time t
            """
            return self.exp(pu, t * self.log(pu, qr))

        def retr(self, pu: jnp.array, vw: jnp.array) -> jnp.array:
            return self.exp(pu, vw)

        def transp(self, pu, qr, vw):
            raise NotImplementedError('This function has not been implemented yet.')

        def pairmean(self, pu: jnp.array, qr: jnp.array) -> jnp.array:
            """Fréchet mean of 2 point in TM

            :param pu: element of TM
            :param qr: element of TM
            :return: Fréchet mean of pv and qw
            """
            return self.geopoint(pu, qr, .5)

        def dist(self, pu: jnp.array, qr: jnp.array) -> float:
            """Distance function that is induced on TM by the Sasaki metric

            :param pu: element of TM
            :param qr: element of TM
            :return: distance between pu and qr in TM
            """
            vw = self.log(pu, qr)
            return jnp.sqrt(self.inner(pu, vw, vw))

        def jacobiField(self, p_A: jnp.array, p_B: jnp.array, t: float, X: jnp.array) -> jnp.array:
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, p_A: jnp.array, p_B: jnp.array, t: float, X: jnp.array) -> jnp.array:
            raise NotImplementedError('This function has not been implemented yet.')


def _gradient_descent(TM: Manifold, x_ini: jnp.array, grad: Callable, step_size: float = 0.1, max_iter: int = 100,
                      tol: float = 1e-6) -> jnp.array:
    """Apply a gradient descent to compute the discrete geodesic in TM with the Sasaki structure.

    :param TM: Tangent bundle
    :param x_ini: initial discrete curve as list of points in TM
    :param grad: gradient function
    :param step_size: step size
    :param max_iter: maximum number of iterations
    :param tol: tolerance for convergence
    :return: discrete geodesic as list of points in _TM
    """

    def body(args):
        x, grad_norm, tol, i = args
        grad_x = grad(x)

        x.at[1:-1].set(jax.vmap(TM.connec.exp)(x[1:-1], -step_size * grad_x))
        grad_norm = jnp.linalg.norm(grad_x)

        return x, grad_norm, tol, i + 1

    def cond(args):
        _, g_norm, tol, i = args

        c = jnp.array([g_norm > tol, i < max_iter])
        return jnp.all(c)

    x, g_norm, _, i = jax.lax.while_loop(cond, body, (x_ini, jnp.array(1.), tol, jnp.array(0)))

    return x
