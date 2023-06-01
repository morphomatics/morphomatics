################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2023 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm, funm

from morphomatics.manifold import Manifold, LieGroup, Connection


class GLpn(Manifold):
    """Returns the product Lie group GL^+(n)^k, i.e., a product of k n-by-n matrices each with positive determinant.

     manifold = GLpn(n, k)

     Elements of GL^+(n)^k are represented as arrays of size kxnxn.

     # NOTE: Tangent vectors are represented as left translations in the Lie algebra, i.e., a tangent vector X at g is
     represented as as d_gL_{g^(-1)}(X)
     """

    def __init__(self, n=3, k=1, structure='AffineGroup'):
        self._n = n
        self._k = k

        if k == 1:
            name = 'Orientation preserving maps of R^n'
        elif k > 1:
            name = '{k}-tuple of orientation preserving maps of R^{n}'.format(k=k, n=n)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        super().__init__(name, k * n**2, point_shape=(k, n, n))

        if structure:
            getattr(self, f'init{structure}Structure')()

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    def rand(self, key: jax.random.KeyArray):
        """Returns a random point in the Lie group. This does not
        follow a specific distribution."""
        A = jax.random.normal(key, self.point_shape)
        return jax.vmap(expm)(A)

    def randvec(self, A, key: jax.random.KeyArray):
        """Returns a random vector in the tangent space at A.
        """
        return jax.random.normal(key, self.point_shape)

    def zerovec(self):
        """Returns the zero vector in the tangent space at g."""
        return jnp.zeros((self.k, self.n, self.n))

    def proj(self, p, X):
        return X

    def initAffineGroupStructure(self):
        """
        Standard group structure with canonical Cartan Shouten Connction.
        """
        structure = GLpn.AffineGroupStructure(self)
        self._connec = structure
        self._group = structure

    class AffineGroupStructure(Connection, LieGroup):
        """
        Standard group structure on GL+n(k) where the composition of two elements is given by component-wise matrix
        multiplication. The connection is the corresponding canonical Cartan Shouten connection. No Riemannian metric is
        used.
        """

        def __init__(self, G):
            """
            Constructor.
            """
            self._G = G

        def __str__(self):
            return 'standard group structure on GL+(n)^k with CCS connection'

        # Group

        @property
        def identity(self):
            """Returns the identity element e of the Lie group."""
            return jnp.tile(jnp.eye(self._G.n), (self._G.k, 1, 1))

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.einsum('kij,kjl->kil', X, Y) - jnp.einsum('kij,kjl->kil', Y, X)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return jnp.reshape(X, (self._G.k * self._G.n**2, 1))

        def lefttrans(self, g, f):
            """Left translation of g by f.
            """
            return jnp.einsum('kij,kjl->kil', f, g)

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return jnp.einsum('kij,kjl->kil', g, f)

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return jnp.linalg.inv(g)

        def exp(self, *argv):
            """Computes the Lie-theoretic and connection exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            return jax.lax.cond(len(argv) == 1,
                                lambda A: A[-1],  # group exp
                                lambda A: jnp.einsum('...ij,...jk', A[-1], A[0]),  # exp of CCS connection
                                (argv[0], jax.vmap(expm)(argv[-1])))

        retr = exp

        def log(self, *argv):
            """Computes the Lie-theoretic and connection logarithm map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # NOTE: as logm() is not available in jax we apply log via funm() (so far this is CPU only; not as stable as
            # logm in numpy)
            logm = lambda m: jnp.real(funm(m, jnp.log))
            return jax.vmap(logm)(jax.lax.cond(len(argv) == 1,
                                               lambda A: A[-1],
                                               lambda A: jnp.einsum('...ij,...kj', A[-1], A[0]),
                                               argv))

        def curvature_tensor(self, f, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at f on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def dleft(self, f, X):
            """Derivative of the left translation by f applied to the tangent vector X at the identity.
            """
            return jnp.einsum('kij,kjl->kil', f, X)

        def dright(self, f, X):
            """Derivative of the right translation by f at g applied to the tangent vector X.
            """
            return jnp.einsum('kij,kjl->kil', X, f)

        def dleft_inv(self, f, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('kij,kjl->kil', self.inverse(f), X)

        def dright_inv(self, f, X):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('kij,kjl->kil', X, self.inverse(f))

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            return jnp.einsum('kij,kjl,klm->kim', g, X, self.inverse(g))

        def transp(self, p, q, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def jacobiField(self, R, Q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')
