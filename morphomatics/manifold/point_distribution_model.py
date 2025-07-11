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

import jax
import jax.numpy as jnp
import numpy as np

from ..geom import Surface
from . import ShapeSpace, Metric
from .util import align, projToGeodesic_flat


class PointDistributionModel(ShapeSpace, Metric):
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

    def tree_flatten(self):
        return tuple(), (self.ref.v.tolist(), self.ref.f.tolist())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        return cls(Surface(*aux_data))

    @property
    def typicaldist(self):
        return self.ref.v.std()

    def update_ref_geom(self, v):
        self.ref.v=v

    def to_coords(self, v):
        # align
        return align(v, self.ref.v)

    def from_coords(self, c):
        return np.asarray(c)

    @property
    def ref_coords(self):
        return self.ref.v

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
        return X.reshape(-1) @ Y.reshape(-1)

    def flat(self, p, X):
        return X

    def sharp(self, p, dX):
        return dX

    def proj(self, p, X):
        return X

    def egrad2rgrad(self, p, X):
        return X

    def exp(self, p, X):
        return p + X

    retr = exp

    def log(self, p, q):
        return q - p

    def curvature_tensor(self, p, X, Y, Z):
        return self.zerovec()

    def rand(self, key: jax.Array):
        v = jax.random.normal(key, self.ref.v.shape)
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

    def adjJacobi(self, p, q, t, X):
        """
        :param p: point
        :param q: point
        :param t: scalar in [0, 1]
        :param X: vector at gam(p,q,t)
        :return: vector at p
        """
        return X / (1.0 - t)

    projToGeodesic = projToGeodesic_flat

    def coords(self, X):
        """Coordinate map of the tangent space at the identity"""
        return X.reshape(-1)
