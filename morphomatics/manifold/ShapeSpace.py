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
from pymanopt.manifolds.manifold import Manifold


class ShapeSpace(Manifold):
    """ Abstract base class for manifold spaces. """

    @abc.abstractmethod
    def update_ref_geom(self, v):
        '''
        :arg v: #v-by-3 array of vertex coordinates
        '''

    @abc.abstractmethod
    def to_coords(self, v):
        '''
        :arg v: #v-by-3 array of vertex coordinates
        :return: manifold coordinates
        '''

    @abc.abstractmethod
    def from_coords(self, c):
        '''
        :arg c: manifold coords.
        :returns: #v-by-3 array of vertex coordinates
        '''

    @property
    @abc.abstractmethod
    def identity(self):
        """ :returns: The identity element (in the sense of Lie group theory)  """

    @abc.abstractmethod
    def projToGeodesic(self, X, Y, P, max_iter=10):
        '''
        Project P onto geodesic from X to Y.

        See:
        Felix Ambellan, Stefan Zachow, Christoph von Tycowicz.
        Geodesic B-Score for Improved Assessment of Knee Osteoarthritis.
        Proc. Information Processing in Medical Imaging (IPMI), LNCS, 2021.

        :arg X, Y: manifold coords defining geodesic X->Y.
        :arg P: manifold coords to be projected to X->Y.
        :returns: manifold coords of projection of P to X->Y
        '''
        assert X.shape == Y.shape
        assert Y.shape == P.shape

        # initial guess
        Pi = X

        # solver loop
        for _ in range(max_iter):
            v = self.log(Pi, Y)
            v /= self.norm(Pi, v)
            w = self.log(Pi, P)
            d = self.inner(Pi, v, w)

            # print(f'|<v, w>|={d}')
            if abs(d) < 1e-6: break

            Pi = self.exp(Pi, d * v)

        return Pi


    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        return self.norm(X, self.log(X, Y))

    def randvec(self, X):
        Y = self.rand()
        y = self.log(X, Y)
        return y / self.norm(X, y)

    def pairmean(self, X, Y):
        y = self.log(X,Y)
        return self.exp(X, y / 2)
