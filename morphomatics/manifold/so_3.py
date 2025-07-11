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

import numpy as np

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Metric
from morphomatics.manifold.util import projToGeodesic_group, multiskew as skew
from morphomatics.manifold.gl_p_n import GLGroupStructure

I = np.eye(3)
O = np.zeros(3)
versor2skew = jnp.array([[O, -I[2], I[1]], [I[2], O, -I[0]], [-I[1], I[0], O]])


class SO3(Manifold):
    """Returns the manifold SO(3), i.e., the set of rotations of 3-dimensional space.

     manifold = SO3()

     Elements of SO(3) are represented as orthogonal 3x3 matrices, i.e., matrices R with R * R^T = eye(3) and det(R) = 1.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.
     """

    def __init__(self, structure='Canonical'):
        name = 'Rotations manifold SO(3)'
        dimension = 3
        point_shape = (3, 3)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        obj = cls(structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initCanonicalStructure(self):
        """
        Instantiate SO(3) with canonical structure.
        """
        structure = SO3.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    def randskew(self, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return skew(S)

    def rand(self, key: jax.Array):
        return expm(self.randvec(I, key))

    def randvec(self, X, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return skew(S)

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, X, H):
        """Orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector ((3,3) array) onto the tangent space at X"""
        return skew(jnp.einsum('ij, kj', H, X))

    @staticmethod
    def project(R):
        """maps 3x3-matrix R to closest SO(3)-matrix by factoring out singular values"""
        U, _, Vt = jnp.linalg.svd(R)
        O = U @ Vt
        det = jnp.sign(jnp.linalg.det(O))
        return U @ Vt.at[:, -1].set(det*Vt[:, -1])

    class CanonicalStructure(Metric, GLGroupStructure):
        """
        The Riemannian metric used is the induced from the embedding space R^(3x3), i.e., this manifold is a
        Riemannian submanifold of R^(3x3) endowed with the usual trace inner product.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "SO3-canonical structure"

        @property
        def typicaldist(self):
            return jnp.pi * jnp.sqrt(3)

        def inner(self, R, X, Y):
            """Frobenius metric"""
            return jnp.einsum('ij, ij', X, Y)

        def norm(self, R, X):
            """Norm from product metric"""
            return jnp.sqrt(self.inner(R, X, X))

        def flat(self, R, X):
            """Lower vector X at R with the metric"""
            return X

        def sharp(self, R, dX):
            """Raise covector dX at R with the metric"""
            return dX

        def egrad2rgrad(self, R, X):
            return self._M.proj(R, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            e = expm(argv[-1])
            return e if len(argv) == 1 else jnp.einsum('ij, jk', e, argv[0])

        retr = exp


        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithm map
                (depending on signature, i.e. whether footpoint is given as well)

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            return logm(argv[-1] if len(argv)==1 else jnp.einsum('ij,kj', argv[-1], argv[0]))

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return 1/4 * self.bracket(self.bracket(X, Y), Z)

        def geopoint(self, R, Q, t):
            """Evaluate the geodesic from R to Q at time t"""
            return self.exp(R, t * self.log(R, Q))

        def transp(self, R, Q, X):
            """Parallel transport for SO(3).
            :param R: element of SO(3)
            :param Q: element of SO(3)
            :param X: tangent vector at R
            :return: parallel transport of X at Q
            """
            O = expm(.5 * logm(jnp.einsum('ij,kj', Q, R)))
            return jnp.einsum('ji,jk,kl', O, X, O)

        def dist(self, R, Q):
            """Product distance function"""
            return jnp.sqrt(self.squared_dist(R, Q))

        def squared_dist(self, R, Q):
            """Product squared distance function"""
            X = logm(jnp.einsum('ij,kj', Q, R))
            return jnp.sum(X**2)

        projToGeodesic = projToGeodesic_group

        def jacobiField(self, R, Q, t, X):
            # ### using (forward-mode) automatic differentiation of geopoint(..)
            f = lambda O: self.geopoint(O, Q, t)
            X_at_R = self.righttrans(R, X) # dRight_R|_e(X)
            geo, J = jax.jvp(f, (R,), (X_at_R,))
            J_at_e = self.righttrans(self.inverse(geo), J) # dRight_geo^-1|_geo(J)
            return geo, skew(J_at_e)

        def jacONB(self, R, Q):
            """Let J be the Jacobi operator along the geodesic from R to Q. This code diagonalizes J.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param R: element of SO(3)
            :param Q: element of SO(3)
            :returns lam, P_G: eigenvalues and orthonormal eigenbasis of Jac at R
            """

            V = self.log(R, Q)

            lam = jnp.zeros(3)

            x = V[0, 1]
            y = V[0, 2]
            z = V[1, 2]

            # eigenvalues (the first one is 0)
            lam = lam.at[1:].set(1 / 4 * (x ** 2 + y ** 2 + z ** 2))

            def normal_case():
                # normalized eigenbasis unless x = 0 and z = 0
                A = jnp.array([[0, -z, 0],
                               [z, 0, x],
                               [0, -x, 0]])

                c = x*y
                d = x**2 + z**2
                e = y*z
                B = jnp.array([[0, -c, d],
                               [c, 0, -e],
                               [-d, e, 0]])

                return A, B

            def special_case():
                # normalized eigenbasis when x = 0 and z = 0

                X1 = jnp.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 0]])
                X3 = jnp.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
                A = jnp.einsum('ij,jk', R, X1)
                B = jnp.einsum('ij,jk', R, X3)

                return A, B

            A, B = jax.lax.cond(jnp.abs(x) + jnp.abs(z) > 1e-6, normal_case, special_case)

            # (number of basis elements) x 3 x 3
            F = jnp.zeros((3, 3, 3))
            F = F.at[0].set(V / jnp.linalg.norm(V, ord='fro'))
            F = F.at[1].set(A / jnp.linalg.norm(A, ord='fro'))
            F = F.at[2].set(B / jnp.linalg.norm(B, ord='fro'))

            return lam, F

        def jacop(self, R, Q, X):
            """ Evaluate the Jacobi operator along the geodesic from R to Q at r.

            :param R: element of SO(3)
            :param Q: element of SO(3)
            :param X: tangent vector at R
            :returns: tangent vector at R
            """
            V = self.log(R, Q)

            # normalize tangent vectors
            V = V / jnp.linalg.norm(V)

            return 1 / 4 * (-jnp.einsum('ij,jk,kl', V, V, X) + 2 * jnp.einsum('ij,jk,kl', V, X, V)
                            - jnp.einsum('ij,jk,kl', X, V, V))

        def adjJacobi(self, R, Q, t, X):
            ### using (reverse-mode) automatic differentiation of geopoint(..)
            f = lambda O: self.geopoint(O, Q, t)
            geo, vjp = jax.vjp(f, R)
            X_at_geo = self.righttrans(geo, X)  # dRight_geo|_e(X)
            aJ = vjp(X_at_geo)[0]
            aJ_at_e = self.righttrans(self.inverse(R), aJ)  # dRight_R^-1|_R(J)
            return skew(aJ_at_e)

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
            # for i in range(3):
            #     Fp = Fp.at[i].set(self.transp(R, P, F[i]))
            #     alpha = alpha.at[i].set(jnp.einsum('...ij,...ij', Fp[i], X))
            #
            # # weights for the linear combination of the three basis elements
            #
            # weights = weightfun(lam, t, self.dist(R, Q))
            #
            # # evaluate the adjoint Jacobi field at R
            # a = weights * alpha
            # for i in range(3):
            #     F = F.at[i].set(a[i] * F[i])
            #
            # return multiskew(jnp.sum(F, axis=0))

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return g.transpose()

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x = X[0, 1]
            y = X[0, 2]
            z = X[1, 2]
            return jnp.array([x, y, z])

        def coords_inv(self, c):
            """Inverse of coords"""
            x, y, z = c[0], c[1], c[2]

            X = jnp.zeros(self._M.point_shape)
            X = X.at[0, 1].set(x)
            X = X.at[1, 0].set(-x)
            X = X.at[0, 2].set(y)
            X = X.at[2, 0].set(-y)
            X = X.at[1, 2].set(z)
            X = X.at[2, 1].set(-z)
            return X


def weightfun(lam, t, d):
    """Weights in the solution of the Jacobi equation. The type determines the boundary condition; see "A variational
    model for data fitting on manifolds by minimizing the acceleration of a Bézier curve" (Bergmann, Gousenbourger).
    Here, the weights give the solution to the boundary value problem with fixed endpoint.
    :param lam: (sets of) eigenvalues of the Jacobi operator
    :param t: value in [0,1] where the Jacobi field(s) is(are) evaluated at
    :param d: length(s) of the geodesic(s)
    :returns weight(s)
    """
    # eigenvalues are non-negative
    w = jnp.sin((1 - t) * jnp.multiply(jnp.sqrt(lam).T, d).T) / jnp.where(lam == 0, 1,
                                                                        jnp.sin(jnp.multiply(jnp.sqrt(lam).T, d).T))

    return jnp.where(w == 0, 1 - t, w)


def logm(R):
    decision_vector = R.diagonal()
    decision_vector = jnp.hstack([decision_vector, decision_vector.sum()])

    i = decision_vector.argmax()
    j = (i + 1) % 3
    k = (j + 1) % 3

    q1 = jnp.empty_like(decision_vector)
    q1 = q1.at[i].set(1 - decision_vector[-1] + 2 * R[i, i])
    q1 = q1.at[j].set(R[j, i] + R[i, j])
    q1 = q1.at[k].set(R[k, i] + R[i, k])
    q1 = q1.at[3].set(R[k, j] - R[j, k])

    q2 = jnp.empty_like(decision_vector)
    q2 = q2.at[0].set(R[2, 1] - R[1, 2])
    q2 = q2.at[1].set(R[0, 2] - R[2, 0])
    q2 = q2.at[2].set(R[1, 0] - R[0, 1])
    q2 = q2.at[3].set(1 + decision_vector[-1])

    quat = jax.lax.cond(i == 3, lambda _: q2, lambda _: q1, i)

    quat = quat / jnp.linalg.norm(quat)
    # w > 0 to ensure 0 <= angle <= pi
    quat = quat * jax.lax.cond(quat[3] < 0, lambda _: -1., lambda _: 1., i)

    def get_scale(s2):
        s = jnp.sqrt(s2 + jnp.finfo(jnp.float64).eps)
        angle = 2 * jnp.arctan2(s, quat[3])
        return angle / jnp.sin(angle / 2)

    sin2 = jnp.sum(quat[:3] ** 2)
    scale = jax.lax.cond(sin2 < 1e-6, lambda _: 2.0, lambda s: get_scale(s), sin2)
    versors = scale * quat[:3]
    return jnp.einsum('ijk,k', versor2skew, versors)


def expm(X):
    # X -> quaternion

    def quaternion(sqn):
        norms = jnp.sqrt(sqn + jnp.finfo(jnp.float64).eps)
        scale = .5 * jnp.sin(norms / 2) / norms
        x = scale * (X[2, 1] - X[1, 2])
        y = scale * (X[0, 2] - X[2, 0])
        z = scale * (X[1, 0] - X[0, 1])
        w = jnp.cos(norms / 2)
        return jnp.stack([x, y, z, w])

    def quaternion_truncated(sqn):
        scale = 0.25 - sqn / 96 + sqn ** 2 / 7680
        x = scale * (X[2, 1] - X[1, 2])
        y = scale * (X[0, 2] - X[2, 0])
        z = scale * (X[1, 0] - X[0, 1])
        w = 1 - sqn / 8 + sqn ** 2 / 384
        return jnp.stack([x, y, z, w])

    sq_norms = .5 * jnp.einsum('ij,ij', X, X)
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

    matrix = matrix.at[0, 0].set(x2 - y2 - z2 + w2)
    matrix = matrix.at[1, 0].set(2 * (xy + zw))
    matrix = matrix.at[2, 0].set(2 * (xz - yw))

    matrix = matrix.at[0, 1].set(2 * (xy - zw))
    matrix = matrix.at[1, 1].set(w2 - x2 + y2 - z2)
    matrix = matrix.at[2, 1].set(2 * (yz + xw))

    matrix = matrix.at[0, 2].set(2 * (xz + yw))
    matrix = matrix.at[1, 2].set(2 * (yz - xw))
    matrix = matrix.at[2, 2].set(w2 - x2 - y2 + z2)

    return matrix
