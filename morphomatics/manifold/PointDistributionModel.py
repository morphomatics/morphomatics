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

import jax.numpy as jnp

from ..geom import Surface
from . import ShapeSpace, Metric, Connection
from .util import align, randn_with_key


class PointDistributionModel(ShapeSpace, Metric, Connection):
    """ Linear manifold space model. """

    def __init__(self, reference: Surface):
        """
        :arg reference: Reference surface (shapes will be encoded as deformations thereof)
        """
        assert reference is not None
        self.ref = reference

        name = 'Point Distribution Model'
        dimension = reference.v.size
        point_shape = reference.v.shape
        super().__init__(name, dimension, point_shape, self, self, None)

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
    def ref_coords(self):
        return self.ref.v.copy().reshape(-1)

    def dist(self, p, q):
        return jnp.sqrt(self.squared_dist(p, q))

    def squared_dist(self, p, q):
        return jnp.sum((p-q)**2)

    def inner(self, p, X, Y):
        """
        :arg X: (list of) tangent vector(s) at p
        :arg Y: (list of) tangent vector(s) at p
        :returns: inner product at p between P_G and H, i.e. <X,Y>_p
        """
        return X @ Y.T

    def proj(self, p, X):
        return X

    def egrad2rgrad(self, p, X):
        return X

    def ehess2rhess(self, p, egrad, ehess, X):
        return ehess

    def exp(self, p, X):
        return p + X

    retr = exp

    def log(self, p, q):
        return q - p

    def curvature_tensor(self, p, X, Y, Z):
        return self.zerovec()

    def rand(self):
        v = randn_with_key(self.ref.v.shape)
        return self.to_coords(v)

    def zerovec(self):
        jnp.zeros(self.ref.v.shape)

    def transp(self, p, q, X):
        return X

    def jacobiField(self, p, q, t, X):
        """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
        :param p: point
        :param q: point
        :param t: scalar in [0,1]
        :param X: tangent vector at p
        :return: tangent vector at gam(t)
        """
        return (1-t) * X

    def eval_adjJacobi(self, p, q, t, X):
        """
        :param p: point
        :param q: point
        :param t: scalar in [0, 1]
        :param X: vector at gam(p,q,t)
        :return: vector at p
        """
        return X / (1.0 - t)

    def projToGeodesic(self, p, q, m):
        '''
        :arg X, Y: manifold coords defining geodesic X->Y.
        :arg P: manifold coords to be projected to X->Y.
        :returns: manifold coords of projection of P to X->Y
        '''
        return super().projToGeodesic(p, q, m, max_iter=1)

    def coords(self, X):
        """Coordinate map of the tangent space at the identity"""
        return X.reshape(-1)
