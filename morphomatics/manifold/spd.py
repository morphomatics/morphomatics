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

from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm_frechet, sqrtm, inv

from morphomatics.manifold import Manifold, Metric, LieGroup
from morphomatics.manifold.discrete_ops import pole_ladder
from morphomatics.manifold.util import projToGeodesic_flat, multisym as sym


class SPD(Manifold):
    """Returns the Sym+(d) manifold, i.e., the manifold of dxd symmetric positive matrices (SPD).

     manifold = SPD(d)

     Elements of Sym+(d) are represented as dxd matrices with positive eigenvalues.

     """

    def __init__(self, d=3, structure='LogEuclidean'):
        if d <= 0:
            raise RuntimeError("d must be an integer no less than 1.")

        name = 'Manifold of symmetric positive definite {d} x {d} matrices'.format(d=d)

        self._d = d

        dimension = int(self._d*(self._d+1)/2)
        point_shape = (self._d, self._d)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children, aux+(self._d,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *aux_data, d = aux_data
        obj = cls(d, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def randsym(self, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return sym(S)

    def rand(self, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return jnp.einsum('ji,jk->ik', S, S)

    def randvec(self, X, key: jax.Array):
        U = self.randsym(key)
        nrmU = jnp.sqrt(jnp.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, S, H):
        """Orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector ((k,d,d) array) onto the tangent space at S"""
        # dright_inv(S,multisym(H)) reduces to dlog(S, ...)
        return dlog(S, sym(H))

    def initLogEuclideanStructure(self):
        """
        Instantiate SPD(d)^k with log-Euclidean structure.
        """
        structure = SPD.LogEuclideanStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    def initAffineInvariantStructure(self):
        """
        Instantiate SPD(d)^k with affine invariant structure.
        """
        structure = SPD.AffineInvariantStructure(self)
        self._metric = structure
        self._connec = structure

    class LogEuclideanStructure(Metric, LieGroup):
        """
            The Riemannian metric used is the log-Euclidean metric that is induced by the standard Euclidean
            trace metric; see
                    Arsigny, V., Fillard, P., Pennec, X., and Ayache., N.
                    Fast and simple computations on tensors with Log-Euclidean metrics.

            This structure also provides a Lie group structure of Sym+(d)^k.
            Tangent vectors are treated as elements in the Lie algebra to improve efficiency.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return f"SPD({self._M._d})-log-Euclidean structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return jnp.sqrt(self._M.dim * 6)

        def inner(self, S, X, Y):
            """Product log-Euclidean metric"""
            return frobenius_inner(X, Y)

        def egrad2rgrad(self, S, D):
            # adjoint of right-translation by S * inverse metric at S * proj of D to tangent space at S
            # first two terms simplify to transpose of Dexp at log(S)
            return dexp(log_mat(S), sym(D))  # Dexp^T = Dexp for sym. matrices

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # cond: group or Riemannian exp
            X = argv[0] if len(argv) == 1 else (argv[-1] + log_mat(argv[0]))
            return exp_mat(X)

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            T = log_mat(argv[-1])
            return sym(T if len(argv) == 1 else T - log_mat(argv[0]))

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the log-Euclidean connection at p on the vectors X, Y, Z. With
            nabla_X Y denoting the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return jnp.zeros(self._M.point_shape)

        def transp(self, S, T, X):
            """Log-Euclidean parallel transport for Sym+(d)^k.
            :param S: element of Symp+(d)
            :param T: element of Symp+(d)
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            # if X were not in algebra but at tangent space at S
            # return dexp(log_mat(T), dlog(S, X))

            return X

        def pairmean(self, S, T):
            return self.exp(S, 0.5 * self.log(S, T))

        def dist(self, S, T):
            """Log-Euclidean distance function in Sym+(d)"""
            return self.norm(S, self.log(S, T))

        def squared_dist(self, S, T):
            """Squared log-Euclidean distance function in Sym+(d)"""
            d = self.log(S, T)
            return self.inner(S, d, d)

        def flat(self, S, X):
            """Lower vector X at S with the log-Euclidean metric"""
            return X

        def sharp(self, S, dX):
            """Raise covector dX at S with the log-Euclidean metric"""
            return dX

        def jacobiField(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return U, (1 - t) * self.transp(S, U, X)

        def adjJacobi(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return (1 - t) * self.transp(U, S, X)

        projectToGeodesic = projToGeodesic_flat

        @property
        def identity(self):
            return jnp.eye(self._M._d)

        def lefttrans(self, S, T):
            """(Commutative) Translation of S by T"""
            return self.exp(log_mat(S) + log_mat(T))

        righttrans = lefttrans

        def inverse(self, S):
            """Inverse map of the Lie group.
            """
            return jnp.linalg.inv(S)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x = X[0, 0]
            y = X[0, 1]
            z = X[0, 2]
            a = X[1, 1]
            b = X[1, 2]
            c = X[2, 2]
            return jnp.array([x, y, z, a, b, c])

        def coords_inv(self, c):
            """Inverse of coords"""
            x, y, z, a, b, c = c[0], c[1], c[2], c[3], c[4], c[5]

            X = np.zeros(self._M.point_shape)
            X[0, 0] = x
            X[0, 1], X[1, 0] = y, y
            X[0, 2], X[2, 0] = z, z
            X[1, 1] = a
            X[1, 2], X[2, 1] = b, b
            X[2, 2] = c
            return X

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.zeros(self._M.point_shape)

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')

    class AffineInvariantStructure(Metric):
        """
            The Riemannian metric used is the product affine-invariant metric; see
                     X. Pennec, P. Fillard, and N. Ayache,
                     A Riemannian framework for tensor computing.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return f"SPD({self._M._d})-affine-invariant structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return jnp.sqrt(self._M.dim * 6)

        def inner(self, S, X, Y):
            """Product affine-invariant Riemannian metric"""
            S_inv = inv(S)
            return jnp.trace(S_inv @ X @ S_inv @ Y)

        def egrad2rgrad(self, S, D):
            """Taken from the Rieoptax implementation of SPD with affine-invariant metric"""
            return S @ D @ S.T

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, S, X):
            """Affine-invariant exponential map
            """
            S_sqrt, S_invSqrt = invSqrt_sqrt_mat(S)

            return S_sqrt @ exp_mat(S_invSqrt @ X @ S_invSqrt) @ S_sqrt

        def log(self, S, P):
            """Affine-invariant logarithm map
            """
            S_sqrt, S_invSqrt = invSqrt_sqrt_mat(S)

            return S_sqrt @ log_mat(S_invSqrt @ P @ S_invSqrt) @ S_sqrt

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the affine-invariant connection at p on the vectors X, Y, Z. With
            nabla_X Y denoting the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def geopoint(self, S, T, t):
            """Evaluate the affine-invariant geodesic from S to T at time t in [0, 1]"""
            return self.exp(S, t * self.log(S, T))

        def transp(self, S, T, X):
            """Affine-invariant parallel transport for Sym+(d).
            :param S: element of Symp+(d)
            :param T: element of Symp+(d)
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            # S_sqrt, S_invSqrt = invSqrt_sqrt_mat(S)
            # E = exp_mat(S_invSqrt @ self.log(S, T) @ S_invSqrt / 2)
            #
            # return S_sqrt @ E @ S_invSqrt @ X @ S_invSqrt @ E @ S_sqrt

            return pole_ladder(self._M, S, T, X)

        def pairmean(self, S, T):
            return self.exp(S, 0.5 * self.log(S, T))

        def squared_dist(self, S, T):
            """Squared affine-invariant distance function in Sym+(d)"""

            # eigval = jnp.linalg.eigvals(jnp.linalg.inv(T) @ S) # CPU only
            # eigval = jax.scipy.linalg.eigh(S, T, eigvals_only=True) # only T=None supported
            L, _ = jax.scipy.linalg.cho_factor(T, lower=True)
            U = jax.scipy.linalg.solve_triangular(L, S, lower=True)
            U = jax.scipy.linalg.solve_triangular(L, U.T, lower=True)
            eigval = jax.scipy.linalg.eigh(U, eigvals_only=True)
            return jnp.sum(jnp.log(eigval)**2)

        def dist(self, S, T):
            """Affine-invariant distance function in Sym+(d)^k"""
            return jnp.sqrt(self.squared_dist(S, T))

        def flat(self, S, X):
            """Lower vector X at S with the metric"""
            S_inv = inv(S)  # g^{1/2}
            return jnp.einsum('ij,jk,kl', S_inv, S_inv, X)

        def sharp(self, S, dX):
            """Raise covector dX at S with the metric"""
            return jnp.einsum('ij,jk,kl', S, S, dX)

        def jacobiField(self, S, T, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, S, T, t, X):
            raise NotImplementedError('This function has not been implemented yet.')


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def funm(S: jnp.array, f: Callable):
    """Matrix function based on scalar function"""
    vals, vecs = jnp.linalg.eigh(S)
    return jnp.einsum('...ij,...j,...kj', vecs, f(vals), vecs)


@funm.defjvp
def funm_jvp(f, primals, tangents):
    """
    Custom JVP rule for funm. A derivation can be found in Equation (2.7) of
    Shapiro, A. (2002). On differentiability of symmetric matrix valued functions.
    School of Industrial and Systems Engineering, Georgia Institute of Technology.
    """
    S, = primals
    X, = tangents

    vals, vecs = jnp.linalg.eigh(S)
    fvals = f(vals)
    primal_out = jnp.einsum('...ij,...j,...kj', vecs, fvals, vecs)

    # frechet derivative of f(S)
    deno = vals[..., None] - vals[..., None, :]
    nume = fvals[..., None] - fvals[..., None, :]
    same_sub = jax.vmap(jax.grad(f))(vals.reshape(-1)).reshape(vals.shape + (1,))
    diff_pow_diag = jnp.where(deno != 0, nume / deno, same_sub)
    diag = jnp.einsum('...ji,...jk,...kl', vecs, X, vecs) * diff_pow_diag
    tangent_out = jnp.einsum('...ij,...jk,...lk', vecs, diag, vecs)

    return primal_out, tangent_out


def sqrt_mat(U):
    """Matrix square root"""
    return funm(U, lambda a: jnp.sqrt(jnp.clip(a, 1e-10, None)))


def invSqrt_mat(U):
    """Inverse of matrix square root (with regularization)"""
    return funm(U, lambda a: 1/jnp.sqrt(jnp.clip(a, 1e-10, None)))


@jax.custom_jvp
def invSqrt_sqrt_mat(U):
    """Matrix square root and its inverse (with regularization).
            Only one eigendecomposition is computed."""
    vals, vecs = jnp.linalg.eigh(U)
    U_sqrt = jnp.einsum('...ij,...j,...kj', vecs, jnp.sqrt(jnp.clip(vals, 1e-10, None)), vecs)
    U_invSqrt = jnp.einsum('...ij,...j,...kj', vecs, 1 / jnp.sqrt(jnp.clip(vals, 1e-10, None)), vecs)

    return jnp.stack([U_sqrt, U_invSqrt])


@invSqrt_sqrt_mat.defjvp
def invSqrt_sqrt_mat_jvp(primals, tangents):
    U, = primals
    X, = tangents

    vals, vecs = jnp.linalg.eigh(U)
    vals = jnp.clip(vals, 1e-10, None)
    sqrt_vals = jnp.sqrt(vals)
    invSqrt_vals = 1 / sqrt_vals
    U_sqrt = jnp.einsum('...ij,...j,...kj', vecs, sqrt_vals, vecs)
    U_invSqrt = jnp.einsum('...ij,...j,...kj', vecs, invSqrt_vals, vecs)

    primal_out = jnp.stack([U_sqrt, U_invSqrt])

    # frechet derivative of f(S); adapted from rieoptax
    deno = vals[..., None] - vals[..., None, :]
    nume_sqrt = sqrt_vals[..., None] - sqrt_vals[..., None, :]
    nume_invSqrt = invSqrt_vals[..., None] - invSqrt_vals[..., None, :]

    # same_sub_sqrt = .5 * invSqrt_vals[..., None]
    # same_sub_invSqrt = -.5 * (invSqrt_vals / vals)[..., None]
    # auto-diff. appears to more stable than the above
    same_sub_sqrt = jax.vmap(jax.grad(jnp.sqrt))(vals.reshape(-1)).reshape(vals.shape + (1,))
    same_sub_invSqrt = jax.vmap(jax.grad(lambda x: 1/jnp.sqrt(x)))(vals.reshape(-1)).reshape(vals.shape + (1,))

    diff_pow_diag_sqrt = jnp.where(deno != 0, nume_sqrt / deno, same_sub_sqrt)
    diff_pow_diag_invSqrt = jnp.where(deno != 0, nume_invSqrt / deno, same_sub_invSqrt)

    VtXV = jnp.einsum('...ji,...jk,...kl', vecs, X, vecs)
    diag_sqrt = VtXV * diff_pow_diag_sqrt
    diag_invSqrt = VtXV * diff_pow_diag_invSqrt

    tangent_out_sqrt = jnp.einsum('...ij,...jk,...lk', vecs, diag_sqrt, vecs)
    tangent_out_invSqrt = jnp.einsum('...ij,...jk,...lk', vecs, diag_invSqrt, vecs)
    tangent_out = jnp.stack([tangent_out_sqrt, tangent_out_invSqrt])

    return primal_out, tangent_out


def log_mat(U):
    """Matrix logarithm (w/ regularization)"""
    return funm(U, lambda a: jnp.log(jnp.clip(a, 1e-10, None)))


def exp_mat(U):
    """Matrix exponential"""
    return funm(U, jnp.exp)


def dexp(X, G):
    """Evaluate the derivative of the matrix exponential at
    X in direction P_G.
    """
    dexpm = lambda X_, G_: expm_frechet(X_, G_, compute_expm=False)
    return dexpm(X, G)


def dlog(X, G):
    """Evaluate the derivative of the matrix logarithm at
    X in direction G.
    """
    ### using logm for [[X, G], [0, X]]
    # n = X.shape[1]
    # # set up [[X, G], [0, X]]
    # W = jnp.hstack((jnp.dstack((X, G)), jnp.dstack((jnp.zeros_like(X), X))))
    # return jnp.array([matrix_log(W[i])[:n, n:] for i in range(X.shape[0])])

    ### using (forward-mode) automatic differentiation of log_mat(X)
    return jax.jvp(log_mat, (X,), (G,))[1]


def frobenius_inner(A: jnp.array, B: jnp.array) -> jnp.array:
    """Trace of A^T @ B"""
    return jnp.einsum("ij,ij", A, B)
