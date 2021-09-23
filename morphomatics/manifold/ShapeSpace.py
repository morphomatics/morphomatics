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

    def randvec(self, X):
        Y = self.rand()
        y = self.log(X, Y)
        return y / self.norm(X, y)

    def pairmean(self, X, Y):
        y = self.log(X,Y)
        return self.exp(X, y / 2)
