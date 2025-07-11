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

import abc

import jax.random

from morphomatics.manifold import Manifold


class ShapeSpace(Manifold):
    """ Abstract base class for shape spaces. """

    @abc.abstractmethod
    def update_ref_geom(self, v):
        '''
        :arg v: #n-by-3 array of vertex coordinates
        '''

    @abc.abstractmethod
    def to_coords(self, v):
        '''
        :arg v: #n-by-3 array of vertex coordinates
        :return: manifold coordinates
        '''

    @abc.abstractmethod
    def from_coords(self, c):
        '''
        :arg c: manifold coords.
        :returns: #n-by-3 array of vertex coordinates
        '''

    @property
    @abc.abstractmethod
    def ref_coords(self):
        """ :returns: Coordinates of reference shape """

    @property
    def M(self) -> Manifold:
        """
        :returns: Manifold of shape coordinates
        (might be other than #self and more efficient for JIT due to fewer dependencies).
        """
        return self

    def randvec(self, X, key: jax.Array):
        return self.connec.log(X, self.rand(key))
