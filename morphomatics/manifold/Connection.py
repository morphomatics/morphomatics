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

import abc


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
    def jacobiField(self, p, q, t, X):
        """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: tangent vector at gam(t)
        """

    @abc.abstractmethod
    def adjJacobi(self, p, q, t, X):
        """Evaluates an adjoint Jacobi field for the geodesic gam from p to q at p.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at gam(t)
        :return: tangent vector at p
        """

    def dxgeo(self, p, q, t, X):
        """Evaluates the differential of the geodesic gam from p to q w.r.t the starting point p at X,
        i.e, d_p gamma(t; ., q) applied to X; the result is en element of the tangent space at gam(t).
        """

        return self.jacobiField(p, q, t, X)

    def dygeo(self, p, q, t, X):
        """Evaluates the differential of the geodesic gam from p to q w.r.t the end point q at X,
        i.e, d_q gamma(t; p, .) applied to X; the result is en element of the tangent space at gam(t).
        """

        return self.jacobiField(q, p, 1 - t, X)

    def adjDxgeo(self, p, q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t the starting point p at X,
        i.e, the adjoint  of d_p gamma(t; ., q) applied to X, which is en element of the tangent space at p.
        """

        return self.adjJacobi(p, q, t, X)

    def adjDygeo(self, p, q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t the endpoint q at X,
        i.e, the adjoint  of d_q gamma(t; p, .) applied to X, which is en element of the tangent space at q.
        """

        return self.adjJacobi(q, p, 1 - t, X)
