################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm, funm

from morphomatics.manifold import Manifold, LieGroup, Connection


class GLpn(Manifold):
    """Returns the Lie group GL^+(n), i.e., the group of n-by-n matrices each with positive determinant.

     manifold = GLpn(n)

     Elements of GL^+(n) are represented as arrays of size nxn.

     # NOTE: Tangent vectors are represented as left translations in the Lie algebra, i.e., a tangent vector X at g is
     represented as d_gL_{g^(-1)}(X)
     """

    def __init__(self, n=3, structure='AffineGroup'):
        self._n = n

        name = 'Orientation preserving maps of R^n'

        super().__init__(name, n**2, point_shape=(n, n))

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children, aux+(self._n,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *aux_data, n = aux_data
        obj = cls(n, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    @property
    def n(self):
        return self._n

    def rand(self, key: jax.Array):
        """Returns a random point in the Lie group. This does not
        follow a specific distribution."""
        A = jax.random.normal(key, self.point_shape)
        return expm(A)

    def randvec(self, A, key: jax.Array):
        """Returns a random vector in the tangent space at A.
        """
        return jax.random.normal(key, self.point_shape)

    def zerovec(self):
        """Returns the zero vector in the tangent space at g."""
        return jnp.zeros((self.n, self.n))

    def proj(self, p, X):
        return X

    def initAffineGroupStructure(self):
        """
        Standard group structure with canonical Cartan Shouten connection.
        """
        structure = GLpn.AffineGroupStructure(self)
        self._connec = structure
        self._group = structure

    class AffineGroupStructure(Connection, LieGroup):
        """
        Standard group structure on GL+(n) where the composition of two elements is given by component-wise matrix
        multiplication. The connection is the corresponding canonical Cartan Shouten (CCS) connection. No Riemannian
        metric is used.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return 'standard group structure on GL+(n) with CCS connection'

        # Group

        @property
        def identity(self):
            """Returns the identity element e of the Lie group."""
            return jnp.eye(self._M.n)

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.einsum('ij,jl->il', X, Y) - jnp.einsum('ij,jl->il', Y, X)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return jnp.reshape(X, (self._M.n ** 2, 1))

        def coords_inverse(self, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def lefttrans(self, g, f):
            """Left translation of g by f.
            """
            return jnp.einsum('ij,jl->il', f, g)

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return jnp.einsum('ij,jl->il', g, f)

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
                                lambda A: jnp.einsum('ij,jk', A[-1], A[0]),  # exp of CCS connection
                                (argv[0], expm(argv[-1])))

        retr = exp

        def log(self, *argv):
            """Computes the Lie-theoretic and connection logarithm map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # NOTE: as logm() is not available in jax we apply log via funm() (so far this is CPU only; not as stable as
            # logm in numpy)
            logm = lambda m: jnp.real(funm(m, jnp.log))
            return logm(jax.lax.cond(len(argv) == 1,
                                     lambda A: A[-1],
                                     lambda A: jnp.einsum('ij,kj', A[-1], A[0]),
                                     argv))

        def curvature_tensor(self, f, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at f on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return - 1 / 4 * self.bracket(self.bracket(X, Y), Z)

        def dleft(self, f, X):
            """Derivative of the left translation by f applied to the tangent vector X at the identity.
            """
            return jnp.einsum('ij,jl->il', f, X)

        def dright(self, f, X):
            """Derivative of the right translation by f at g applied to the tangent vector X.
            """
            return jnp.einsum('ij,jl->il', X, f)

        def dleft_inv(self, f, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('ij,jl->il', self.inverse(f), X)

        def dright_inv(self, f, X):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('ij,jl->il', X, self.inverse(f))

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            return jnp.einsum('ij,jl,lm->im', g, X, self.inverse(g))

        def transp(self, f, g, X):
            """
            Parallel transport of the CCS connection along one-parameter subgroups; see Sec. 5.3.3 of
            X. Pennec and M. Lorenzi,
            "Beyond Riemannian geometry: The affine connection setting for transformation groups."

            """
            f_invg = self.lefttrans(g, self.inverse(f))
            h = self.geopoint(self.identity, f_invg, .5)

            return self.dleft_inv(f_invg, self.dleft(h, self.dright(h, X)))

        def jacobiField(self, R, Q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')
