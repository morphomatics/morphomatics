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

import numpy as np

class Metric(metaclass=abc.ABCMeta):
    """
    Interface setting out a template for a metric on the tangent bundle of a manifold.
    """

    @abc.abstractmethod
    def __str__(self):
        """Returns a string representation of the particular metric."""

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

    @abc.abstractmethod
    def inner(self, p, X, Y):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors X and Y from the tangent space at p.
        """

    @abc.abstractmethod
    def proj(self, p, X):
        """Projects a vector X in the ambient space on the tangent space at
        p.
        """

    @abc.abstractmethod
    def egrad2rgrad(self, p, X):
        """Maps the Euclidean gradient X in the ambient space on the tangent
        space of the manifold at p.
        """

    @abc.abstractmethod
    def ehess2rhess(self, p, G, H, X):
        """Converts the Euclidean gradient G and Hessian H of a function at
        a point p along a tangent vector X to the Riemannian Hessian
        along X on the manifold.
        """

    @abc.abstractmethod
    def retr(self, p, X):
        """Computes a retraction mapping a vector X in the tangent space at
        p to the manifold.
        """

    def norm(self, p, X):
        """Computes the norm of a tangent vector X in the tangent space at
        p.
        """
        return np.sqrt(self.inner(p, X, X))
