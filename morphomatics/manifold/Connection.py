################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2023 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import abc

import jax
import jax.numpy as jnp


class Connection(metaclass=abc.ABCMeta):
    """
    Interface setting out a template for a connection on the tangent bundle of a manifold.
    """

    @abc.abstractmethod
    def __str__(self):
        """Returns a string representation of the particular connection."""

    @abc.abstractmethod
    def exp(self, p, X):
        """Exponential map of the connection at p applied to the tangent vector X.
        """

    @abc.abstractmethod
    def retr(self, p, X):
        """Computes a retraction mapping a vector X in the tangent space at
        p to the manifold.
        """

    @abc.abstractmethod
    def log(self, p, q):
        """Logarithmic map of the connection at p applied to q.
        """

    def geopoint(self, p, q, t):
        """Evaluates the geodesic between p and q at time t.
        """
        return self.exp(p, t * self.log(p, q))

    @abc.abstractmethod
    def transp(self, p, q, X):
        """Computes a vector transport which transports a vector X in the
        tangent space at p to the tangent space at q.
        """

    @abc.abstractmethod
    def curvature_tensor(self, p, X, Y, Z):
        """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
        covariant derivative of Y in direction X and [] being the Lie bracket, the convention
            R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
        is used.
        """

    def jacobiField(self, p, q, t, X):
        """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: tangent vector at gam(t)
        """
        return jax.lax.cond(t == 1,
                            lambda args: jnp.zeros_like(args[3]),
                            lambda args: jax.lax.cond(t == 0,
                                                      lambda args2: args2[3],
                                                      lambda args2: self.eval_jacobiField(*args2)[1],
                                                      args),
                            (p, q, t, X))

    @abc.abstractmethod
    def eval_jacobiField(self, p, q, t, X):
        """
        Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: [b, J] with J and b being the Jacobi field at t and the corresponding basepoint
        """

    def dxgeo(self, p, q, t, X):
        """Evaluates the differential of the geodesic gam from p to q w.r.t. the starting point p at X,
        i.e, d_p gamma(t; ., q) applied to X; the result is en element of the tangent space at gam(t).
        """

        return self.jacobiField(p, q, t, X)

    def dygeo(self, p, q, t, X):
        """Evaluates the differential of the geodesic gam from p to q w.r.t. the end point q at X,
        i.e, d_q gamma(t; p, .) applied to X; the result is en element of the tangent space at gam(t).
        """

        return self.jacobiField(q, p, 1 - t, X)


def _eval_jacobi_embed(C: Connection, p, q, t, X):
    """ Implementation of eval_jacobi for isometrically embedded manifolds using (forward-mode) automatic
    differentiation of geopoint(..).

    ATTENTION: the result must be projected to the tangent space!
    """
    f = lambda O: C.geopoint(O, q, t)

    return jax.jvp(f, (p,), (X,))
