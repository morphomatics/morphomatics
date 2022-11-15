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
import jax

from morphomatics.manifold import Metric, Connection, LieGroup

class Manifold(metaclass=abc.ABCMeta):
    """
    Abstract base class setting out a template for manifold classes.
    Morphomatics's Lie group and Riemannian manifold classes inherit from Manifold.
    """

    def __init__(self, name, dimension: int, point_shape,
                 connec: Connection = None, metric: Metric = None, group: LieGroup = None):
        self._name = name
        self._dimension = dimension
        self._point_shape = point_shape
        # (possibly) define a connection on the tangent bundle
        self._connec = connec
        # (possibly) define a metric on the tangent bundle
        self._metric = metric
        # (possibly) define a group operation turning the manifold into a Lie group
        self._group = group

    def __str__(self):
        """Returns a string representation of the particular manifold."""
        conf = 'metric='+str(self._metric) if self._metric else ''
        conf += ' connection='+str(self._connec) if self._connec else ''
        conf += ' group='+str(self._group) if self._group else ''
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
    def metric(self) -> Metric:
        return self._metric

    @property
    def connec(self) -> Connection:
        return self._connec

    @property
    def group(self) -> LieGroup:
        return self._group

    @abc.abstractmethod
    def rand(self, key: jax.random.KeyArray):
        """Returns a random point of the manifold."""

    @abc.abstractmethod
    def randvec(self, p, key: jax.random.KeyArray):
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
        Pi = X.copy()

        # solver loop
        for _ in range(max_iter):
            v = self.connec.log(Pi, Y)
            v = v / self.metric.norm(Pi, v)
            w = self.connec.log(Pi, P)
            d = self.metric.inner(Pi, v, w)

            # print(f'|<v, w>|={d}')
            if abs(d) < 1e-6: break

            Pi = self.connec.exp(Pi, d * v)

        return Pi
