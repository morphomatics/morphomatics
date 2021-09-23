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


class Manifold(metaclass=abc.ABCMeta):
    """
    Abstract base class setting out a template for manifold classes.
    Morphomatics's Lie group and Riemannian manifold classes inherit from Manifold.
    """

    def __init__(self, name, dimension, point_shape, connec=None, metric=None, group=None):
        assert isinstance(dimension, (int, np.integer)), \
            "dimension must be an integer"
        # assert ((isinstance(point_shape, int) and point_shape > 0) or
        #         (isinstance(point_shape, (list, tuple)) and
        #          all(np.array(point_shape) > 0))), \
        #     ("'point_shape' must be a positive integer or a sequence of "
        #      "positive integers")

        self._name = name
        self._dimension = dimension
        self._point_shape = point_shape
        # (possibly) define a connection on the tangent bundle
        self._connec = connec
        # (possibly) define a metric on the tangent bundle
        self._metric = metric
        # (possibly) define a group operation turning the manifold into a Lie group
        self._group = group

    @property
    def __str__(self):
        """Returns a string representation of the particular manifold."""
        conf = 'metric='+self._metric.__str__ if self._metric else ''
        conf += ' connection='+self._connec.__str__ if self._connec else ''
        conf += ' group='+self._group.__str__ if self._group else ''
        if not conf:
            return self._name
        return f'{self._name} ({conf.strip()})'

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension

    @property
    def point_shape(self):
        """Dimensions of elements of the manifold.

        Tuple of dimension, e.g., if an element is given by a 3-by-3 matrix, then its point shape is [3, 3].
        """
        return self._point_shape

    @property
    def metric(self):
        return self._metric

    @property
    def connec(self):
        return self._connec

    @property
    def group(self):
        return self._group

    @abc.abstractmethod
    def rand(self):
        """Returns a random point of the manifold."""

    @abc.abstractmethod
    def randvec(self, p):
        """Returns a random vector in the tangent space at p."""

    @abc.abstractmethod
    def zerovec(self):
        """Returns the zero vector in any tangent space."""

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
        assert self.connec
        assert self.metric

        # initial guess
        Pi = X

        # solver loop
        for _ in range(max_iter):
            v = self.connec.log(Pi, Y)
            v /= self.metric.norm(Pi, v)
            w = self.connec.log(Pi, P)
            d = self.metric.inner(Pi, v, w)

            # print(f'|<v, w>|={d}')
            if abs(d) < 1e-6: break

            Pi = self.connec.exp(Pi, d * v)

        return Pi
