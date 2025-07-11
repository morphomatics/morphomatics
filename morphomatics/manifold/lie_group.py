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

import abc
import jax
import jax.numpy as jnp

from morphomatics.manifold.connection import Connection

class LieGroup(Connection):
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
    def coords_inv(self, X):
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
    def exp(self, *argv):
        """Computes the Lie-theoretic and canonical Cartan Shouten (CCS) connection exponential map
        (depending on signature, i.e. whether a footpoint is given as well)
        """

    @abc.abstractmethod
    def log(self, *argv):
        """Computes the Lie-theoretic and CCS connection exponential map
        (depending on signature, i.e. whether a footpoint is given as well)
        """

    def curvature_tensor(self, f, X, Y, Z):
        """Evaluates the curvature tensor R of the CCS connection at f on the vectors X, Y, Z. With nabla_X Y denoting
        the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
            R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
        is used.
        """
        return - 1 / 4 * self.bracket(self.bracket(X, Y), Z)

    def transp(self, f, g, X):
        """
        Parallel transport of the CCS connection along one-parameter subgroups; see Sec. 5.3.3 of
        X. Pennec and M. Lorenzi,
        "Beyond Riemannian geometry: The affine connection setting for transformation groups."

        """
        f_invg = self.lefttrans(g, self.inverse(f))
        h = self.geopoint(self.identity, f_invg, .5)

        # return self.dleft_inv(f_invg, self.dleft(h, self.dright(h, X)))
        return self.lefttrans(self.inverse(f_invg), self.lefttrans(h, self.righttrans(h, X)))

    @abc.abstractmethod
    def adjrep(self, g, X):
        """Adjoint representation of g applied to the tangent vector X at the identity.
        """
