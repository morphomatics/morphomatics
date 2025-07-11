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

import functools

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as tree

from .manifold import Manifold

try:
    import pymanopt

    from pymanopt.manifolds.manifold import Manifold as ManoptManifold
    from pymanopt.autodiff.backends._backend import Backend
    from pymanopt.autodiff import backend_decorator_factory


    def conjugate_result(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return tree.tree_map(jnp.conj, function(*args, **kwargs))

        return wrapper

    class JaxBackend(Backend):
        def __init__(self):
            super().__init__("Jax")

        @staticmethod
        def is_available():
            return jax is not None

        @Backend._assert_backend_available
        def prepare_function(self, function):
            return function

        @Backend._assert_backend_available
        def generate_gradient_operator(self, function, num_arguments):
            gradient = conjugate_result(
                jax.grad(function) if num_arguments == 1 else
                jax.grad(function, argnums=list(range(num_arguments)))
            )
            return gradient

        @staticmethod
        def _hessian_vector_product(function, argnum):
            raise NotImplementedError()

        @Backend._assert_backend_available
        def generate_hessian_operator(self, function, num_arguments):
            raise NotImplementedError()


    if "jax" not in pymanopt.autodiff.backends.__all__:
        factory = backend_decorator_factory(JaxBackend)
        pymanopt.autodiff.backends.jax = factory
        pymanopt.function.jax = factory
        pymanopt.autodiff.backends.__all__.append("jax")
        pymanopt.function.__all__.append("jax")

except ImportError:
    _has_manopt = False
    ManoptManifold = object
else:
    _has_manopt = True


class ManoptWrap(ManoptManifold):
    """
    Wraper for pymanopt to make manifolds from morphomatics compatible.
    """

    def __init__(self, M: Manifold):
        self._M = M
        super().__init__(str(M), M.dim) # as of v0.2.6rc1

    # Manifold properties that subclasses can define

    @property
    def typicaldist(self):
        """Returns the "scale" of the manifold. This is used by the
        trust-regions solver to determine default initial and maximal
        trust-region radii.
        """
        return self._M.metric.typicaldist

    def inner_product(self, X, G, H):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `P_G` and `H` in the tangent space at `X`.
        """
        return self._M.metric.inner(X, G, H)

    def dist(self, X, Y):
        """
        Geodesic distance on the manifold
        """
        return self._M.metric.dist(X, Y)

    def projection(self, X, G):
        """Projects a vector `P_G` in the ambient space on the tangent space at
        `X`.
        """
        return self._M.metric.proj(X, G)

    def euclidean_to_riemannian_gradient(self, X, G):
        """Maps the Euclidean gradient P_G in the ambient space on the tangent
        space of the manifold at X.
        """
        return self._M.metric.egrad2rgrad(X, G)

    def norm(self, X, G):
        """Computes the norm of a tangent vector `P_G` in the tangent space at
        `X`.
        """
        return self._M.metric.norm(X, G)

    def exp(self, X, U):
        """
        The exponential (in the sense of Lie group theory) of a tangent
        vector U at X.
        """
        return self._M.connec.exp(X, U)

    def retraction(self, X, G):
        """
        A retraction mapping from the tangent space at X to the manifold.
        See Absil for definition of retraction.
        """
        return self.exp(X, G)

    def log(self, X, Y):
        """
        The logarithm (in the sense of Lie group theory) of Y. This is the
        inverse of exp.
        """
        return self._M.connec.log(X, Y)

    def transport(self, x1, x2, d):
        """
        Transports d, which is a tangent vector at x1, into the tangent
        space at x2.
        """
        return self._M.connec.transp(x1, x2, d)

    def random_point(self):
        """Returns a random point on the manifold."""
        key = jax.random.PRNGKey(np.random.randint(1 << 32))
        return self._M.rand(key)

    def random_tangent_vector(self, X):
        """Returns a random vector in the tangent space at `X`. This does not
        follow a specific distribution.
        """
        key = jax.random.PRNGKey(np.random.randint(1 << 32))
        return self._M.randvec(X, key)

    def zero_vector(self, X):
        """Returns the zero vector in the tangent space at X."""
        return self._M.zerovec(X)

    def euclidean_to_riemannian_hessian(self, p, grad, Hess, X):
        """
        Convert Euclidean into Riemannian Hessian.
        """
        return

    def pair_mean(self, X, Y):
        """
        Computes the intrinsic mean of X and Y, that is, a point that lies
        mid-way between X and Y on the geodesic arc joining them.
        """
        return self.exp(X, 0.5 * self.log(X, Y))
