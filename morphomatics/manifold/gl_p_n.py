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
from jax.scipy.linalg import expm, funm
from scipy.linalg import logm as scipy_logm

from morphomatics.manifold import Manifold, LieGroup


class GLpn(Manifold):
    """Returns the Lie group GL^+(n), i.e., the group of n-by-n matrices each with positive determinant.

     manifold = GLpn(n)

     Elements of GL^+(n) are represented as arrays of size nxn.

     # NOTE: Tangent vectors are represented as left translations in the Lie algebra, i.e., a tangent vector X at g is
     represented as d_gL_{g^(-1)}(X)
     """

    def __init__(self, n=3, structure='GLGroup'):
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

    def initGLGroupStructure(self):
        """
        Standard group structure with canonical Cartan Shouten connection.
        """
        structure = GLGroupStructure(self)
        self._connec = structure
        self._group = structure

class GLGroupStructure(LieGroup):
    """
    Standard group structure on GL+(n) where the composition of two elements is given by component-wise matrix
    multiplication. The connection is the corresponding canonical Cartan Shouten (CCS) connection. No Riemannian
    metric is used.
    """

    def __init__(self, M: Manifold):
        """ Construct group.
        :param M: underlying manifold
        """
        self._M = M

    def __str__(self):
        return 'standard group structure on GL+(n) with CCS connection'

    # Group

    @property
    def identity(self):
        """Returns the identity element e of the Lie group."""
        return jnp.eye(self._M.point_shape[-1])

    def bracket(self, X, Y):
        """Lie bracket in the Lie algebra."""
        return jnp.einsum('ij,jl->il', X, Y) - jnp.einsum('ij,jl->il', Y, X)

    def coords(self, X):
        """Coordinate map for the tangent space at the identity."""
        return jnp.reshape(X, (self._M.point_shape[-1] ** 2, 1))

    def coords_inv(self, x):
        n = self._M.point_shape[-1]
        return jnp.reshape(x, (n, n))

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
        """Computes the Lie-theoretic and CCS connection exponential map
        (depending on signature, i.e. whether a footpoint is given as well)
        """
        return jax.lax.cond(len(argv) == 1,
                            lambda A: A[-1],  # group exp
                            lambda A: jnp.einsum('ij,jk', A[-1], A[0]),  # exp of CCS connection
                            (argv[0], expm(argv[-1])))

    retr = exp

    def log(self, *argv):
        """Computes the Lie-theoretic and CCS connection logarithm map
        (depending on signature, i.e. whether a footpoint is given as well)
        """
        # NOTE: as logm() is not available in jax; funm() is CPU only (and rather unstable)
        # logm = lambda m: jnp.real(funm(m, jnp.log))
        return logm(jax.lax.cond(len(argv) == 1,
                                 lambda A: A[-1],
                                 lambda A: jnp.einsum('ij,jk', A[-1], self.inverse(A[0])),
                                 argv))

    def adjrep(self, g, X):
        """Adjoint representation of g applied to the tangent vector X at the identity.
        """
        return jnp.einsum('ij,jl,lm->im', g, X, self.inverse(g))

    def jacobiField(self, R, Q, t, X):
        raise NotImplementedError('This function has not been implemented yet.')

@jax.custom_jvp
def logm(m):
    """Computes the matrix logarithm of m."""
    m = jnp.asarray(m)

    # Promote the input to inexact (float/complex).
    # Note that jnp.result_type() accounts for the enable_x64 flag.
    m = m.astype(jnp.result_type(float, m.dtype))

    # Wrap scipy function to return the expected dtype.
    _scipy_logm = lambda a: scipy_logm(a).astype(m.dtype)

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(m.shape),
        dtype=m.dtype)

    # Use vmap_method="sequential" because scipy's logm does not seam to handle broadcasted inputs.
    return jax.pure_callback(_scipy_logm, result_shape_dtype, m, vmap_method="sequential")

@logm.defjvp
def logm_jvp(primals, tangents):
    """Evaluate the derivative of the matrix logarithm at
    X in direction G.
    """
    m, = primals
    x, = tangents

    n = m.shape[1]
    # set up [[m, x], [0, m]]
    W = jnp.vstack((jnp.hstack((m, x)), jnp.hstack((jnp.zeros_like(m), m))))
    logW = logm(W)
    return logW[:n, :n], logW[:n, n:]
