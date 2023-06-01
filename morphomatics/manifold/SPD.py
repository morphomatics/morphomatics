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

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm_frechet

from morphomatics.manifold import Manifold, Metric, LieGroup
from morphomatics.manifold.util import multisym


class SPD(Manifold):
    """Returns the product manifold Sym+(d)^k, i.e., a product of k dxd symmetric positive matrices (SPD).

     manifold = SPD(k, d)

     Elements of Sym+(d)^k are represented as arrays of size kxdxd where every dxd slice is an SPD matrix, i.e., a
     symmetric matrix S with positive eigenvalues.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.

     """

    def __init__(self, k=1, d=3, structure='LogEuclidean'):
        if d <= 0:
            raise RuntimeError("d must be an integer no less than 1.")

        if k == 1:
            name = 'Manifold of symmetric positive definite {d} x {d} matrices'.format(d=d, k=k)
        elif k > 1:
            name = 'Manifold of {k} symmetric positive definite {d} x {d} matrices (Sym^+({d}))^{k}'.format(d=d, k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k
        self._d = d

        dimension = int((self._d*(self._d+1)/2) * self._k)
        point_shape = (self._k, self._d, self._d)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def randsym(self, key: jax.random.KeyArray):
        S = jax.random.normal(key, self.point_shape)
        return multisym(S)

    def rand(self, key: jax.random.KeyArray):
        return self.group.exp(self.randsym(key))

    def randvec(self, X, key: jax.random.KeyArray):
        U = self.randsym(key)
        nrmU = jnp.sqrt(jnp.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, S, H):
        """orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector ((k,3,3) array) onto the tangent space at S"""
        # dright_inv(S,multisym(H)) reduces to dlog(S, ...)
        return dlog(S, multisym(H))

    def initLogEuclideanStructure(self):
        """
        Instantiate SPD(d)^k with log-Euclidean structure.
        """
        structure = SPD.LogEuclideanStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    class LogEuclideanStructure(Metric, LieGroup):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product but featuring the log-Euclidean
        multiplication ensuring a group structure s.t. the metric is bi-invariant.

            The Riemannian metric used is the product Log-Euclidean metric that is induced by the standard Euclidean
            trace metric; see
                    Arsigny, V., Fillard, P., Pennec, X., and Ayache., N.
                    Fast and simple computations on tensors with Log-Euclidean metrics.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "SPD(k, d)-canonical structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return jnp.sqrt(self._M.dim * 6)

        def inner(self, S, X, Y):
            """product metric"""
            return jnp.sum(jnp.einsum('...ij,...ij', X, Y))

        def eleminner(self, S, X, Y):
            """element-wise inner product"""
            return jnp.einsum('...ij,...ij', X, Y)

        def norm(self, S, X):
            """norm from product metric"""
            return jnp.sqrt(self.inner(S, X, X))

        def elemnorm(self, S, X):
            """element-wise norm"""
            return jnp.sqrt(self.eleminner(S, X, X))

        def egrad2rgrad(self, S, D):
            # adjoint of right-translation by S * inverse metric at S * proj of D to tangent space at S
            # first two terms simplify to transpose of Dexp at log(S)
            return dexp(log_mat(S), multisym(D))  # Dexp^T = Dexp for sym. matrices

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # cond: group or Riemannian exp
            X = jax.lax.cond(len(argv) == 1, lambda a: a[0], lambda a: a[-1] + log_mat(a[0]), argv)
            return exp_mat(X)

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
            (depending on signature, i.e. whether footpoint is given as well)
            """

            T = log_mat(argv[-1])
            # if len(argv) == 2: # Riemannian log
            #     T = T - log_mat(argv[0])
            T = jax.lax.cond(len(argv) == 1, lambda ST: ST[1], lambda ST: ST[1] - log_mat(ST[0]), (argv[0], T))

            return multisym(T)

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return jnp.zeros(self._M.point_shape)

        def geopoint(self, S, T, t):
            """ Evaluate the geodesic from S to T at time t in [0, 1]"""
            return self.exp(S, t * self.log(S, T))

        @property
        def identity(self):
            return jnp.tile(jnp.eye(3), (self._M.k, 1, 1))

        def transp(self, S, T, X):
            """Parallel transport for Sym+(d)^k.
            :param S: element of Symp+(d)^k
            :param T: element of Symp+(d)^k
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            # if X were not in algebra but at tangent space at S
            # return dexp(log_mat(T), dlog(S, X))

            return X

        def pairmean(self, S, T):
            return self.exp(S, 0.5 * self.log(S, T))

        def dist(self, S, T):
            """Distance function in Sym+(d)^k"""
            return self.norm(S, self.log(S, T))

        def squared_dist(self, S, T):
            """Squared distance function in Sym+(d)^k"""
            d = self.log(S, T)
            return self.inner(S, d, d)

        def flat(self, S, X):
            """Lower vector X at S with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, S, dX):
            """Raise covector dX at S with the metric"""
            raise NotImplementedError('This function has not been implemented yet.')

        def jacobiField(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return U, (1 - t) * self.transp(S, U, X)

        def adjJacobi(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return (1 - t) * self.transp(U, S, X)

        def lefttrans(self, S, X):
            """Left-translation of X by R"""
            return self.exp(log_mat(S) + log_mat(X))

        righttrans = lefttrans

        def dleft(self, S, X):
            """Derivative of the left translation by f at e applied to the tangent vector X.
            """
            return dexp(log_mat(S), X)

        dright = dleft

        def dleft_inv(self, S, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return dlog(S, X)

        dright_inv = dleft_inv

        def inverse(self, S):
            """Inverse map of the Lie group.
            """
            return jnp.linalg.inv(S)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x = X[:, 0, 0]
            y = X[:, 0, 1]
            z = X[:, 0, 2]
            a = X[:, 1, 1]
            b = X[:, 1, 2]
            c = X[:, 2, 2]
            return jnp.hstack((x, y, z, a, b, c))
            # i, j = np.triu_indices(X.shape[-1])
            # return X[:, i, j].T.reshape(-1)

        def coords_inverse(self, c):
            """Inverse of coords"""
            k = self._M._k
            x, y, z, a, b, c = c[:k], c[k:2 * k], c[2 * k:3 * k], c[3 * k:4 * k], c[4 * k:5 * k], c[5 * k:]

            X = np.zeros(self._M.point_shape)
            X[:, 0, 0] = x
            X[:, 0, 1], X[:, 1, 0] = y, y
            X[:, 0, 2], X[:, 2, 0] = z, z
            X[:, 1, 1] = a
            X[:, 1, 2], X[:, 2, 1] = b, b
            X[:, 2, 2] = c
            return X

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.zeros(self._M.point_shape)

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')

    def projToGeodesic(self, X, Y, P, max_iter=10):
        '''
        :arg X, Y: elements of Symp+(d)^k defining geodesic X->Y.
        :arg P: element of Symp+(d)^k to be projected to X->Y.
        :returns: projection of P to X->Y
        '''

        # all tagent vectors in common space i.e. algebra
        v = self.connec.log(X, Y)
        v = v / self.metric.norm(X, v)

        w = self.connec.log(X, P)
        d = self.metric.inner(X, v, w)

        return self.connec.exp(X, d * v)


def logm(S):
    # Matrix logarithm (w/ projection to SPD cone)
    vals, vecs = jnp.linalg.eigh(S)
    vals = jnp.log(jnp.clip(vals, 1e-10, None))
    return jnp.einsum('...ij,...j,...kj', vecs, vals, vecs)


def expm(X):
    vals, vecs = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...kj', vecs, jnp.exp(vals), vecs)


@jax.custom_jvp
def log_mat(U):
    return logm(U)


@jax.custom_jvp
def exp_mat(U):
    return expm(U)


@log_mat.defjvp
def log_mat_jvp(U, X):
    d = U[0].shape[-1]
    V = U[0] - jnp.eye(d)
    sqn = jnp.sum(V**2, axis=(-1,-2))[..., None, None]

    # double-where trick to catch NaNs
    # 1st where
    U_ = jnp.where(sqn < 1e-6, U[0] + jnp.diag(jnp.arange(d)), U[0])

    # auto-diff. of logm()
    primal_out, tangent_out = jax.jvp(logm, (U_,), X)

    # 2nd where
    trunc_log = lambda D: D - .5 * D @ D # truncated power series (in terms of U-I)
    trunc_jvp = lambda A, W: 2 * W - .5 * (W @ A + A @ W) # diff. of truncated power series expr.
    tangent_out = jnp.where(sqn < 1e-6, trunc_jvp(U[0], X[0]), tangent_out)
    primal_out = jnp.where(sqn < 1e-6, trunc_log(V), primal_out)

    return primal_out, tangent_out


@exp_mat.defjvp
def exp_mat_jvp(U, X):
    d = U[0].shape[-1]
    sqn = jnp.sum(U[0]**2, axis=(-1,-2))[..., None, None]

    # double-where trick to catch NaNs
    # 1st where
    U_ = jnp.where(sqn < 1e-6, U[0] - jnp.diag(jnp.arange(d)), U[0])

    # auto-diff. of logm()
    primal_out, tangent_out = jax.jvp(expm, (U_,), X)

    # 2nd where
    trunc_exp = lambda A: A + jnp.eye(d) + .5 * A @ A # truncated power series
    trunc_jvp = lambda A, W: W + .5 * (W @ A + A @ W) # diff. of truncated power series expr.
    tangent_out = jnp.where(sqn < 1e-6, trunc_jvp(U[0], X[0]), tangent_out)
    primal_out = jnp.where(sqn < 1e-6, trunc_exp(U[0]), primal_out)

    return primal_out, tangent_out


def dexp(X, G):
    """Evaluate the derivative of the matrix exponential at
    X in direction P_G.
    """
    dexpm = lambda X_, G_: expm_frechet(X_, G_, compute_expm=False)
    return jax.vmap(dexpm)(X, G)


def dlog(X, G):
    """Evaluate the derivative of the matrix logarithm at
    X in direction P_G.
    """
    ### using logm for [[X, P_G], [0, X]]
    # n = X.shape[1]
    # # set up [[X, P_G], [0, X]]
    # W = jnp.hstack((jnp.dstack((X, P_G)), jnp.dstack((jnp.zeros_like(X), X))))
    # return jnp.array([matrix_log(W[i])[:n, n:] for i in range(X.shape[0])])

    ### using (forward-mode) automatic differentiation of log_mat(X)
    return jax.jvp(log_mat, (X,), (G,))[1]
