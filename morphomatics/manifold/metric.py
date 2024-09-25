################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import abc

import jax
import jax.numpy as jnp

from morphomatics.manifold.connection import Connection


class Metric(Connection):
    """
    Interface setting out a template for a metric on the tangent bundle of a manifold. It is modelled as a subclass of
    its Levi-Civita connection.
    """

    @property
    @abc.abstractmethod
    def typicaldist(self):
        """Returns the "scale" of the manifold. This is used by the
        trust-regions solver to determine default initial and maximal
        trust-region radii.
        """

    @abc.abstractmethod
    def dist(self, p, q):
        """Returns the geodesic distance between two points p and q on the
        manifold."""

    def squared_dist(self, p, q):
        return self.dist(p, q)**2

    @abc.abstractmethod
    def inner(self, p, X, Y):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors X and Y from the tangent space at p.
        """

    @abc.abstractmethod
    def egrad2rgrad(self, p, X):
        """Maps the Euclidean gradient X in the ambient space on the tangent
        space of the manifold at p.
        """

    @abc.abstractmethod
    def ehess2rhess(self, p, G, H, X):
        """Converts the Euclidean gradient P_G and Hessian H of a function at
        a point p along a tangent vector X to the Riemannian Hessian
        along X on the manifold.
        """

    def norm(self, p, X):
        """Computes the norm of a tangent vector X in the tangent space at
        p.
        """
        return jnp.sqrt(self.inner(p, X, X))

    @abc.abstractmethod
    def flat(self, p, X):
        """Lower vector X at p with the metric"""

    @abc.abstractmethod
    def sharp(self, p, dX):
        """Raise covector dX at p with the metric"""

    @abc.abstractmethod
    def adjJacobi(self, p, q, t, X):
        """Evaluates an adjoint Jacobi field for the geodesic gam from p to q at p.
        :param p: element of the Riemannian manifold
        :param q: element of the Riemannian manifold
        :param t: scalar in [0,1]
        :param X: tangent vector at gam(t)
        :return: tangent vector at p
        """

    def adjDxgeo(self, p, q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t. the starting point p at X,
        i.e, the adjoint  of d_p gamma(t; ., q) applied to X, which is an element of the tangent space at p.
        """

        return self.adjJacobi(p, q, t, X)

    def adjDygeo(self, p, q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from p to q w.r.t. the endpoint q at X,
        i.e, the adjoint  of d_q gamma(t; p, .) applied to X, which is en element of the tangent space at q.
        """

        return self.adjJacobi(q, p, 1 - t, X)


def _eval_adjJacobi_embed(g: Metric, p, q, t, X):
    """ Implementation of eval_adjJacobi for isometrically embedded manifolds using (forward-mode) automatic
    differentiation of geopoint(..).

    ATTENTION: the result must be projected to the tangent space!
    """
    f = lambda O: g.geopoint(O, q, t)
    gam, Jt = jax.vjp(f, p)
    co_X = g.flat(gam, X)
    return g.sharp(p, Jt(co_X)[0])
