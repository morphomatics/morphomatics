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

from morphomatics.manifold import Manifold, Metric, LieGroup
from morphomatics.manifold.util import multiskew, vectime3d

I = np.eye(3)
O = np.zeros(3)
versor2skew = jnp.array([[O, -I[2], I[1]], [I[2], O, -I[0]], [-I[1], I[0], O]])


class SO3(Manifold):
    """Returns the product manifold SO(3)^k, i.e., a product of k rotations in 3 dimensional space.

     manifold = SO3(k)

     Elements of SO(3)^k are represented as arrays of size kx3x3 where every 3x3 slice is an orthogonal R, i.e., a
     R R with R * R^T = eye(3) and det(R) = 1.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.
     """

    def __init__(self, k=1, structure='Canonical'):
        if k == 1:
            name = 'Rotations manifold SO(3)'
        elif k > 1:
            name = 'Rotations manifold SO(3)^{k}'.format(k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k

        dimension = 3 * self._k
        point_shape = (self._k, 3, 3)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def initCanonicalStructure(self):
        """
        Instantiate SO(3)^k with canonical structure.
        """
        structure = SO3.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    @property
    def k(self):
        return self._k

    def randskew(self, key: jax.random.KeyArray):
        S = jax.random.normal(key, self.point_shape)
        return multiskew(S)

    def rand(self, key: jax.random.KeyArray):
        return self.group.exp(self.randskew(key))

    def randvec(self, X, key: jax.random.KeyArray):
        U = self.randskew(key)
        nrmU = jnp.sqrt(jnp.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, X, H):
        """orthogonal (with respect to the euclidean inner product) projection of ambient
        vector ((k,3,3) array) onto the tangentspace at X"""
        return multiskew(jnp.einsum('...ij,...kj', H, X))

    class CanonicalStructure(Metric, LieGroup):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "SO3(k)-canonical structure"

        @property
        def typicaldist(self):
            return jnp.pi * jnp.sqrt(3 * self._M.k)

        def inner(self, R, X, Y):
            """product metric"""
            return jnp.sum(jnp.einsum('...ij,...ij', X, Y))

        def norm(self, R, X):
            """norm from product metric"""
            return jnp.sqrt(self.inner(R, X, X))

        def flat(self, R, X):
            """Lower vector X at R with the metric"""
            return X

        def sharp(self, R, dX):
            """Raise covector dX at R with the metric"""
            return dX

        def egrad2rgrad(self, R, X):
            return self._M.proj(R, X)

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, R, X):
            return self.exp(R, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            return jax.lax.cond(len(argv) == 1,
                                lambda A: A[-1],
                                lambda A: jnp.einsum('...ij,...jk', A[-1], A[0]),
                                (argv[0], expm(argv[-1])))

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            return logm(jax.lax.cond(len(argv) == 1,
                                     lambda A: A[-1],
                                     lambda A: jnp.einsum('...ij,...kj', A[-1], A[0]),
                                     argv))

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return 1/4 * self.bracket(self.bracket(X, Y), Z)

        def geopoint(self, R, Q, t):
            """Evaluate the geodesic from R to Q at time t in [0, 1]"""
            return self.exp(R, t * self.log(R, Q))

        @property
        def identity(self):
            return jnp.tile(jnp.eye(3), (self._M.k, 1, 1))

        def transp(self, R, Q, X):
            """Parallel transport for SO(3)^k.
            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :param X: tangent vector at R
            :return: parallel transport of X at Q
            """
            O = expm(.5 * logm(jnp.einsum('...ij,...kj', Q, R)))
            return jnp.einsum('...ji,...jk,...kl', O, X, O)

        def pairmean(self, R, Q):
            return self.exp(R, 0.5 * self.log(R, Q))

        def dist(self, R, Q):
            """product distance function"""
            return jnp.sqrt(self.squared_dist(R, Q))

        def squared_dist(self, R, Q):
            """product distance function"""
            X = logm(jnp.einsum('...ij,...kj', Q, R))
            return jnp.sum(X ** 2)

        def projToGeodesic(self, R, Q, P, max_iter=10):
            '''
            :arg X, Y: elements of SO(3)^k defining geodesic X->Y.
            :arg P: element of SO(3)^k to be projected to X->Y.
            :returns: projection of P to X->Y
            '''

            # all tagent vectors in common space i.e. algebra
            v = self.log(R, Q)
            v = v / self.norm(R, v)

            # initial guess
            Pi = R.copy()

            # solver loop
            for _ in range(max_iter):
                w = self.log(Pi, P)
                d = self.inner(Pi, v, w)

                # print(f'|<v, w>|={d}')
                if abs(d) < 1e-6: break

                Pi = self.exp(Pi, d * v)

            return Pi

        def jacobiField(self, R, Q, t, X):
            # ### using (forward-mode) automatic differentiation of geopoint(..)
            f = lambda O: self.geopoint(O, Q, t)
            geo, J = jax.jvp(f, (R,), (self.dright(R, X),))
            return geo, multiskew(self.dright_inv(geo, J))

        def jacONB(self, R, Q):
            """Let J be the Jacobi operator along the geodesic from R to Q. This code diagonalizes J.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :returns lam, P_G: eigenvalues and orthonormal eigenbasis of Jac at R
            """
            k = len(R)  # self._M.k

            V = self.log(R, Q)
            nor_v = jnp.linalg.norm(V, ord='fro', axis=(1, 2))
            # normalize V
            V = vectime3d(nor_v, V)

            # (number of faces) x (number of basis elements) x 3 x 3
            F = jnp.zeros((k, 3, 3, 3))
            lam = jnp.zeros((k, 3))

            x = jnp.atleast_2d(V[..., 0, 1]).T
            y = jnp.atleast_2d(V[..., 0, 2]).T
            z = jnp.atleast_2d(V[..., 1, 2]).T

            # eigenvalues
            lam = lam.at[:, 1:].set(1 / 4 * jnp.tile(x ** 2 + y ** 2 + z ** 2, (1, 2)))

            # change sign for convenience
            e1 = jnp.tile(-versor2skew[2], (k, 1, 1))
            e2 = jnp.tile(versor2skew[1], (k, 1, 1))
            e3 = jnp.tile(-versor2skew[0], (k, 1, 1))

            elemnorm = lambda A: jnp.sqrt(jnp.einsum('...ij,...ij', A, A))

            # normalized eigenbasis unless x = 0 and z = 0
            F = F.at[:, 0, ...].set(vectime3d(1 / elemnorm(V), V))
            F = F.at[:, 1, ...].set(vectime3d(-z, e1) + vectime3d(x, e3))
            F = F.at[:, 2, ...].set(vectime3d(-x * y, e1) + vectime3d(x ** 2 + z ** 2, e2) + vectime3d(- y * z, e3))

            # take care of division by 0 later
            # with jnp.errstate(all='ignore'):
            F = F.at[:, 1, ...].set(vectime3d(1 / elemnorm(F[:, 1, ...]), F[:, 1, ...]))
            F = F.at[:, 2, ...].set(vectime3d(1 / elemnorm(F[:, 2, ...]), F[:, 2, ...]))

            # take care of special cases
            zero_xz = jnp.abs(x + z).reshape(k, 1, 1, 1) < 1e-12
            F = jnp.where(zero_xz, 1 / jnp.sqrt(2) * jnp.stack((e1, e2, e3), axis=1), F)

            return lam, F

        def jacop(self, R, Q, X):
            """ Evaluate the Jacobi operator along the geodesic from R to Q at r.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :param X: tangent vector at R
            :returns: tangent vector at R
            """
            V = self.log(R, Q)

            # normalize tangent vectors
            V = V / jnp.linalg.norm(V, axis=(-2, -1))[..., None, None]

            return 1 / 4 * (-jnp.einsum('...ij,...jk,...kl', V, V, X) + 2 * jnp.einsum('...ij,...jk,...kl', V, X, V)
                            - jnp.einsum('...ij,...jk,...kl', X, V, V))

        def adjJacobi(self, R, Q, t, X):
            ### using (reverse-mode) automatic differentiation of geopoint(..)
            f = lambda O: self.geopoint(O, Q, t)
            geo, vjp = jax.vjp(f, R)
            return multiskew(self.dright_inv(R, vjp(self.dright(geo, X))[0]))

            # # gam(t)
            # P = self.geopoint(R, Q, t)
            #
            # # orthonormal eigenvectors of the Jacobi operator that generate the parallel frame field along gam and the
            # # eigenvalues
            # lam, F = self.jacONB(R, Q)
            #
            # # expand X w.r.t. F at gam(t)
            # Fp = jnp.zeros_like(F)
            # alpha = jnp.zeros_like(lam)
            #
            # eleminner = lambda A, B: jnp.einsum('...ij,...ij', A, B)
            # elemdist = lambda A, B: jnp.linalg.norm(logm(jnp.einsum('...ij,...kj', A, B)), axis=(1, 2))
            #
            # for i in range(3):
            #     Fp = Fp.at[:, i, ...].set(self.transp(R, P, F[:, i, ...]))
            #     alpha = alpha.at[:, i].set(eleminner(Fp[:, i, ...], X))
            #
            # # weights for the linear combination of the three basis elements
            #
            # weights = weightfun(lam, t, elemdist(R, Q))
            #
            # # evaluate the adjoint Jacobi field at R
            # a = weights * alpha
            # for i in range(3):
            #     F = F.at[:, i, ...].set(vectime3d(a[:, i], F[:, i, ...]))
            #
            # return multiskew(np.sum(F, axis=1))

        def dleft(self, f, X):
            """Derivative of the left translation by f at e applied to the tangent vector X.
            """
            return jnp.einsum('...ij,...jk', f, X)

        def dright(self, f, X):
            """Derivative of the right translation by f at e applied to the tangent vector X.
            """
            return jnp.einsum('...ij,...jk', X, f)

        def dleft_inv(self, f, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('...ji,...jk', f, X)

        def dright_inv(self, f, X):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
            """
            return jnp.einsum('...ij,...kj', X, f)

        def lefttrans(self, R, X):
            """Left-translation of X to the tangent space at R"""
            return jnp.einsum('...ij,...jk', R, X)

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return jnp.einsum('...ij,...jk', g, f)

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return jnp.einsum('...ij->...ji', g)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x = X[:, 0, 1]
            y = X[:, 0, 2]
            z = X[:, 1, 2]
            return jnp.hstack((x, y, z))

        def coords_inverse(self, c):
            """Inverse of coords"""
            k = self._M._k
            x, y, z = c[:k], c[k:2 * k], c[2 * k:]

            X = jnp.zeros(self._M.point_shape)
            X = X.at[:, 0, 1].set(x)
            X = X.at[:, 1, 0].set(-x)
            X = X.at[:, 0, 2].set(y)
            X = X.at[:, 2, 0].set(-y)
            X = X.at[:, 1, 2].set(z)
            X = X.at[:, 2, 1].set(-z)
            return X

        def bracket(self, X, Y):
            return jnp.einsum('kij,kjl->kil', X, Y) - jnp.einsum('kij,kjl->kil', Y, X)

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')


def weightfun(k, t, d):
    """Weights in the solution of the Jacobi equation. The type determines the boundary condition; see "A variational
    model for data fitting on manifolds by minimizing the acceleration of a Bézier curve" (Bergmann, Gousenbourger).
    Here, the weights give the solution to the boundary value problem with fixed endpoint.
    :param k: (sets of) eigenvalues of the Jacobi operator
    :param t: value in [0,1] where the Jacobi field(s) is(are) evaluated at
    :param d: length(s) of the geodesic(s)
    :returns weight(s)
    """
    # eigenvalues are non-negative
    w = jnp.sin((1 - t) * jnp.multiply(jnp.sqrt(k).T, d).T) / jnp.where(k == 0, 1,
                                                                        jnp.sin(jnp.multiply(jnp.sqrt(k).T, d).T))

    return jnp.where(w == 0, 1 - t, w)


def logm(R):
    decision_matrix = R.diagonal(axis1=1, axis2=2)
    decision_matrix = jnp.hstack([decision_matrix, decision_matrix.sum(axis=1)[:, None]])

    i = decision_matrix.argmax(axis=1)
    j = (i + 1) % 3
    k = (j + 1) % 3

    q1 = jnp.empty_like(decision_matrix)
    ind = jnp.arange(len(i))
    q1 = q1.at[ind, i].set(1 - decision_matrix[:, -1] + 2 * R[ind, i, i])
    q1 = q1.at[ind, j].set(R[ind, j, i] + R[ind, i, j])
    q1 = q1.at[ind, k].set(R[ind, k, i] + R[ind, i, k])
    q1 = q1.at[ind, 3].set(R[ind, k, j] - R[ind, j, k])

    q2 = jnp.empty_like(decision_matrix)
    q2 = q2.at[:, 0].set(R[:, 2, 1] - R[:, 1, 2])
    q2 = q2.at[:, 1].set(R[:, 0, 2] - R[:, 2, 0])
    q2 = q2.at[:, 2].set(R[:, 1, 0] - R[:, 0, 1])
    q2 = q2.at[:, 3].set(1 + decision_matrix[:, -1])

    quat = jnp.where((i == 3)[:, None], q2, q1)

    quat = quat / jnp.linalg.norm(quat, axis=1)[:, None]
    # w > 0 to ensure 0 <= angle <= pi
    quat = quat * jnp.where(quat[:, 3] < 0, -1., 1.)[:, None]

    def get_scale(s2):
        s = jnp.sqrt(s2 + jnp.finfo(jnp.float64).eps)
        angle = 2 * jnp.arctan2(s, quat[:, 3])
        return angle / jnp.sin(angle / 2)

    sin2 = jnp.sum(quat[:, :3] ** 2, axis=1)
    scale = jnp.where(sin2 < 1e-6, 2.0, get_scale(sin2))
    versors = scale[:, None] * quat[:, :3]
    return jnp.einsum('ijk,...k', versor2skew, versors)


def expm(X):
    # X -> quaternion

    def quaternion(sqn):
        norms = jnp.sqrt(sqn + jnp.finfo(jnp.float64).eps)
        scale = .5 * jnp.sin(norms / 2) / norms
        x = scale * (X[..., 2, 1] - X[..., 1, 2])
        y = scale * (X[..., 0, 2] - X[..., 2, 0])
        z = scale * (X[..., 1, 0] - X[..., 0, 1])
        w = jnp.cos(norms / 2)
        return jnp.stack([x, y, z, w])

    def quaternion_truncated(sqn):
        scale = 0.25 - sqn / 96 + sqn ** 2 / 7680
        x = scale * (X[..., 2, 1] - X[..., 1, 2])
        y = scale * (X[..., 0, 2] - X[..., 2, 0])
        z = scale * (X[..., 1, 0] - X[..., 0, 1])
        w = 1 - sqn / 8 + sqn ** 2 / 384
        return jnp.stack([x, y, z, w])

    sq_norms = .5 * jnp.einsum('...ij,...ij', X, X)
    x, y, z, w = jnp.where(sq_norms <= 1e-3 ** 2, quaternion_truncated(sq_norms), quaternion(sq_norms))

    # to rotation matrix
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = jnp.zeros_like(X)

    matrix = matrix.at[:, 0, 0].set(x2 - y2 - z2 + w2)
    matrix = matrix.at[:, 1, 0].set(2 * (xy + zw))
    matrix = matrix.at[:, 2, 0].set(2 * (xz - yw))

    matrix = matrix.at[:, 0, 1].set(2 * (xy - zw))
    matrix = matrix.at[:, 1, 1].set(w2 - x2 + y2 - z2)
    matrix = matrix.at[:, 2, 1].set(2 * (yz + xw))

    matrix = matrix.at[:, 0, 2].set(2 * (xz + yw))
    matrix = matrix.at[:, 1, 2].set(2 * (yz - xw))
    matrix = matrix.at[:, 2, 2].set(w2 - x2 - y2 + z2)

    return matrix
