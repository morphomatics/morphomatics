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

from pymanopt.manifolds.manifold import Manifold



class ManoptWrap(Manifold):
    """
    Wraper for pymanopt to make manifolds from morphomatics compatible.
    """

    def __init__(self, M):
        self._M = M

    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._M.__str__

    # Manifold properties that subclasses can define

    @property
    def dim(self):
        """
        Dimension of the manifold
        """
        return self._M.dim

    @property
    def typicaldist(self):
        """Returns the "scale" of the manifold. This is used by the
        trust-regions solver to determine default initial and maximal
        trust-region radii.
        """
        return self._M.metric.typicaldist

    def inner(self, X, G, H):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `G` and `H` in the tangent space at `X`.
        """
        return self._M.metric.inner(X, G, H)

    def dist(self, X, Y):
        """
        Geodesic distance on the manifold
        """
        return self._M.metric.dist(X, Y)

    def proj(self, X, G):
        """Projects a vector `G` in the ambient space on the tangent space at
        `X`.
        """
        return self._M.metric.proj(X, G)

    def norm(self, X, G):
        """Computes the norm of a tangent vector `G` in the tangent space at
        `X`.
        """
        return self._M.metric.norm(X, G)

    def exp(self, X, U):
        """
        The exponential (in the sense of Lie group theory) of a tangent
        vector U at X.
        """
        return self._M.connec.exp(X, U)

    def retr(self, X, G):
        """
        A retraction mapping from the tangent space at X to the manifold.
        See Absil for definition of retraction.
        """
        return self.exp(X, G)

    def log(self, X, Y):
        """
        The logarithm (in the sense of Lie group theory) of Y. This is the
        inverse of exp.
        """
        return self._M.connec.log(X, Y)

    def transp(self, x1, x2, d):
        """
        Transports d, which is a tangent vector at x1, into the tangent
        space at x2.
        """
        return self._M.connec.transp(x1, x2, d)

    def rand(self):
        """Returns a random point on the manifold."""
        return self._M.rand()

    def randvec(self, X):
        """Returns a random vector in the tangent space at `X`. This does not
        follow a specific distribution.
        """
        return self._M.randvec()

    def zerovec(self, X):
        """Returns the zero vector in the tangent space at X."""
        return self._M.zerovec()

    def ehess2rhess(self, X, Hess):
        """
        Convert Euclidean into Riemannian Hessian.
        """
        return

    def pairmean(self, X, Y):
        """
        Computes the intrinsic mean of X and Y, that is, a point that lies
        mid-way between X and Y on the geodesic arc joining them.
        """
        return self.exp(X, 0.5 * self.log(X, Y))
