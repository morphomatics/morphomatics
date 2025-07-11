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

import numpy as np
from typing import Sequence

import jax
import jax.numpy as jnp

from morphomatics.manifold import ShapeSpace, Metric, Kendall
from morphomatics.manifold.discrete_ops import pole_ladder
from morphomatics.manifold.connection import _eval_jacobi_embed


class SizeAndShape(ShapeSpace):
    """
    Size-and-shape space: (SO_m)-equivalence classes of landmark configurations in R^m.
    """

    def __init__(self, shape: Sequence[int], structure='Canonical'):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")

        m = shape[-1]
        dimension = int(np.prod(shape) - m - m * (m - 1) / 2)

        self.ref = None

        name = 'Size-and-shape space of ' + 'x'.join(map(str, shape[:-1])) + ' Landmarks in R^' + str(shape[-1])
        super().__init__(name, dimension, shape)
        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children, aux + (self.point_shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *aux_data, shape = aux_data
        obj = cls(shape, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def update_ref_geom(self, v):
        self.ref = self.to_coords(v)

    def to_coords(self, v):
        '''
        :arg v: array of landmark coordinates
        :return: manifold coordinates
        '''
        return SizeAndShape.project(v)

    def from_coords(self, c):
        '''
        :arg c: manifold coords.
        :returns: array of landmark coordinates
        '''
        return c

    @property
    def ref_coords(self):
        """ :returns: Coordinates of reference shape """
        return self.ref

    def rand(self, key: jax.Array):
        p = jax.random.normal(key, self.point_shape)
        return SizeAndShape.project(p)

    def randvec(self, p, key: jax.Array):
        v = jax.random.normal(key, self.point_shape)
        return SizeAndShape.horizontal(p, v)

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    @staticmethod
    def project(x):
        """
        Project to pre-shape space.
        : param x: Point to project.
        :returns: Projected x.
        """
        return Kendall.center(x)

    def proj(self, p, X):
        """ Project a vector X from the ambient Euclidean space onto the tangent space at p. """
        return SizeAndShape.horizontal(p, X)

    @staticmethod
    def vertical(p, X):
        """
        Compute vertical component of X at base point p by solving the sylvester equation
        App^T+pp^TA = Xp^T-pX^T for A. If p has full rank (det(pp^T) > 0), then there exists a unique solution
        A, it is skew-symmetric and Ap is the vertical component of X
        """
        return Kendall.vertical(p, X)

    @staticmethod
    def horizontal(p, X):
        """
        compute horizontal component of X.
        """
        return Kendall.horizontal(p, X)

    def initCanonicalStructure(self):
        """
        Instantiate the preshape sphere with canonical structure.
        """
        structure = SizeAndShape.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    class CanonicalStructure(Metric):
        """
        The Riemannian metric used is the induced metric from the ambient space (R^m)^k.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            super().__init__(M)
            self._M = M

        def __str__(self):
            return "canonical structure"

        @property
        def typicaldist(self):
            return np.pi/2

        def inner(self, p, X, Y):
            return (X * Y).sum()

        def norm(self, p, X):
            return jnp.sqrt(self.inner(p, X, X))

        def flat(self, p, X):
            """Lower vector X at p with the metric"""
            return X

        def sharp(self, p, dX):
            """Raise covector dX at p with the metric"""
            return dX

        def egrad2rgrad(self, p, X):
            return self._M.proj(p, X)

        def exp(self, p, X):
            return p + X

        retr = exp

        def log(self, p, q):
            q = Kendall.wellpos(p, q)
            return q - p

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def transp(self, p, q, X):
            # pole ladder transports along horizontal geodesic, thus, map (p, X) to the well-positioned representative
            R = Kendall.opt_rot(q, p)
            return pole_ladder(self._M, p @ R, q, X @ R, 10)

        def pairmean(self, p, q):
            return self.geopoint(p, q, .5)

        def dist(self, p, q):
            q = Kendall.wellpos(p, q)
            return self.norm(p, self.log(p, q))

        def squared_dist(self, p, q):
            v = self.log(p, q)
            return self.inner(p, v, v)

        def jacobiField(self, p, q, t, X):
            b, J = _eval_jacobi_embed(self, p, q, t, X)
            return b, SizeAndShape.horizontal(b, J)

        def adjJacobi(self, p, q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')
