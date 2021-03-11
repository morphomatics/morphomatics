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

import numpy as np

from ..geom import Surface
from .ShapeSpace import ShapeSpace
from .util import align

class PointDistributionModel(ShapeSpace):
    """ Linear manifold space model. """

    def __init__(self, reference: Surface):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        """
        assert reference is not None
        self.ref = reference

    def __str__(self):
        return 'Point Distribution Model'

    @property
    def dim(self):
        return self.ref.v.size

    @property
    def typicaldist(self):
        return self.ref.v.std()

    def update_ref_geom(self, v):
        self.ref.v=v

    def to_coords(self, v):
        # align
        v = align(v, self.ref.v)
        return v.reshape(-1)

    def from_coords(self, c):
        return c.reshape(self.ref.v.shape)

    @property
    def identity(self):
        return self.ref.v.copy().reshape(-1)

    def inner(self, X, G, H):
        """
        :arg G: (list of) tangent vector(s) at X
        :arg H: (list of) tangent vector(s) at X
        :returns: inner product at X between G and H, i.e. <G,H>_X
        """
        return G @ np.asanyarray(H).T

    def proj(self, X, U):
        return U

    def ehess2rhess(self, X, egrad, ehess, H):
        return ehess

    def exp(self, X, G):
        return X+G

    retr = exp

    def log(self, X, Y):
        return Y-X

    def rand(self):
        v = np.random.randn(self.ref.v.shape)
        return self.to_coords(v)

    def transp(self, X1, X2, G):
        return G

    def projToGeodesic(self, X, Y, P):
        return super().projToGeodesic(X, Y, P, max_iter=1)
