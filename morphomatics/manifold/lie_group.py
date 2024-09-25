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

# postponed evaluation of annotations to circumvent cyclic dependencies (will be default behavior in Python 4.0)
from __future__ import annotations

import abc

class LieGroup(metaclass=abc.ABCMeta):
    """
    Interface setting out a template for Lie group classes.
    """

    def __init__(self, M: Manifold):
        """ Construct connection.
        :param M: underlying manifold
        """
        self._M = M

    @abc.abstractmethod
    def __str__(self):
        """Returns a string representation of the particular group."""

#    @abc.abstractmethod
#    def compose(self, g, f):
#        """Group operation"""

    @property
    @abc.abstractmethod
    def identity(self):
        """Returns the identity element e of the Lie group."""

    @abc.abstractmethod
    def coords(self, X):
        """Coordinate map for the tangent space at the identity."""

    @abc.abstractmethod
    def coords_inverse(self, X):
        """Inverse coordinate map for the tangent space at the identity."""

    @abc.abstractmethod
    def bracket(self, X, Y):
        """Lie bracket in Lie algebra."""

    @abc.abstractmethod
    def lefttrans(self, g, f):
        """Left translation of g by f.
        """

    @abc.abstractmethod
    def righttrans(self, g, f):
        """Right translation of g by f.
        """

    @abc.abstractmethod
    def inverse(self, g):
        """Inverse map of the Lie group.
        """

    @abc.abstractmethod
    def exp(self, X):
        """Computes the Lie-theoretic exponential map of a tangent vector X at e.
        """

    @abc.abstractmethod
    def log(self, g):
        """Computes the Lie-theoretic logarithm of g. This is the inverse of `exp`.
        """

    @abc.abstractmethod
    def dleft(self, f, X):
        """Derivative of the left translation by f at e applied to the tangent vector X.
        """

    @abc.abstractmethod
    def dright(self, f, X):
        """Derivative of the right translation by f at e applied to the tangent vector X.
        """

    @abc.abstractmethod
    def dleft_inv(self, f, X):
        """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
        """

    @abc.abstractmethod
    def dright_inv(self, f, X):
        """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
        """

    @abc.abstractmethod
    def adjrep(self, g, X):
        """Adjoint representation of g applied to the tangent vector X at the identity.
        """
