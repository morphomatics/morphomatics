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
import jax
from jax.tree_util import register_pytree_node

import operator as op

from morphomatics.manifold import Metric, Connection, LieGroup

class ManifoldMeta(abc.ABCMeta):
    """Metaclass for abstract base class for which subclasses are to be registered as jax PyTree"""
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        register_pytree_node(cls, op.methodcaller('tree_flatten'), cls.tree_unflatten)
        return cls

class Manifold(metaclass=ManifoldMeta):
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

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""

    def tree_unflatten_instance(self, aux_data, children):
        """Specifies an unflattening recipe given an instance (possibly of a subclass)."""
        C, M, G = aux_data
        Cdict, Mdict, Gdict = children

        def setup(cls, dict):
            if cls is None.__class__:
                return None
            obj = cls(self)
            obj.__dict__.update(dict)
            return obj

        self._connec = setup(C, Cdict)
        self._metric = setup(M, Mdict)
        self._group = setup(G, Gdict)

    def tree_flatten(self):
        """Specifies a flattening recipe for PyTree registration."""
        aux_data = (self._connec.__class__, self._metric.__class__, self._group.__class__)
        wo_ = lambda o: {} if o is None else {k: v for k, v in o.__dict__.items() if k[0] != '_'}
        children = (wo_(self._connec), wo_(self._metric), wo_(self._group))
        return (children, aux_data)

    def __str__(self):
        return self._name

    def __repr__(self):
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
    def rand(self, key: jax.Array):
        """Returns a random point of the manifold."""

    @abc.abstractmethod
    def randvec(self, p, key: jax.Array):
        """Returns a random vector in the tangent space at p."""

    @abc.abstractmethod
    def zerovec(self):
        """Returns the zero vector in any tangent space."""

    @abc.abstractmethod
    def proj(self, p, X):
        """Projects a vector X in the ambient space on the tangent space at
        p.
        """
