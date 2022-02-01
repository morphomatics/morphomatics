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

import numpy as np

from scipy.spatial.transform import Rotation
import numpy.linalg as la

from morphomatics.manifold import Manifold, Metric, Connection, LieGroup
from pymanopt.manifolds.rotations import randskew, randrot
from pymanopt.tools.multi import multiskew

from morphomatics.manifold.util import vectime3d


class SO3(Manifold):
    """Returns the product manifold SO(3)^k, i.e., a product of k rotations in 3 dimensional space.

     manifold = SO3(k)

     Elements of SO(3)^k are represented as arrays of size kx3x3 where every 3x3 slice is an orthogonal matrix, i.e., a
     matrix R with R * R^T = eye(3) and det(R) = 1.

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
        point_shape = [self._k, 3, 3]
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

    def rand(self):
        return randrot(3, self._k)

    def randvec(self, X):
        U = randskew(3, self._k)
        nrmU = np.sqrt(np.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self):
        return np.zeros((self._k, 3, 3))

    class CanonicalStructure(Metric, Connection, LieGroup):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
        Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

            # tensor mapping versor to skew-sym. matrix
            I = np.eye(3)
            O = np.zeros(3)
            self.versor2skew = np.array([[O, -I[2], I[1]], [I[2], O, -I[0]], [-I[1], I[0], O]])
            # ... and the opposite direction
            self.skew2versor = .5 * self.versor2skew

        @property
        def __str__(self):
            return "SO3(k)-canonical structure"

        @property
        def typicaldist(self):
            return np.pi * np.sqrt(3 * self._M.k)

        def inner(self, R, X, Y):
            """product metric"""
            return np.sum(np.einsum('...ij,...ij', X, Y))

        def eleminner(self, R, X, Y):
            """element-wise inner product"""
            return np.einsum('...ij,...ij', X, Y)

        def norm(self, R, X):
            """norm from product metric"""
            return np.sqrt(self.inner(R, X, X))

        def elemnorm(self, R, X):
            """element-wise norm"""
            return np.sqrt(self.eleminner(R, X, X))

        def proj(self, X, H):
            """orthogonal (with respect to the euclidean inner product) projection of ambient
            vector ((k,3,3) array) onto the tangentspace at X"""
            # skew-sym. part of: H * X.T
            return multiskew(np.einsum('...ij,...kj', X, H))

        egrad2rgrad = proj

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            return

        def retr(self, R, X):
            # TODO
            return self.exp(R, X)

        def exp(self, R, X):
            """Riemannian exponential with base point R evaluated at X

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            assert R.shape == X.shape

            a = R.ndim - 2
            versors = Rotation.from_rotvec(np.tensordot(X, self.skew2versor, axes=([a, a+1], [a, a+1])))
            return np.einsum('...ij,...jk', versors.as_matrix(), R)

        def log(self, R, Q):
            """Riemannian logarithm with base point R evaluated at Q

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            assert R.shape == Q.shape

            versors = Rotation.from_matrix(np.einsum('...ij,...kj', Q, R)).as_rotvec()
            X = np.einsum('ijk,...k', self.versor2skew, versors)
            return multiskew(X)

        def geopoint(self, R, Q, t):
            """Evaluate the geodesic from R to Q at time t in [0, 1]"""
            assert R.shape == Q.shape and np.isscalar(t)

            return self.exp(R, t * self.log(R, Q))

        def identity(self):
            return np.tile(np.eye(3), (self._M.k, 1, 1))

        def transp(self, R, Q, X):
            """Parallel transport for SO(3)^k.
            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :param X: tangent vector at R
            :return: parallel transport of X at Q
            """
            assert R.shape == Q.shape == X.shape

            # o <- log(R,Q)/2
            o = .5 * Rotation.from_matrix(np.einsum('...ij,...kj', Q, R)).as_rotvec()
            O = Rotation.from_rotvec(o).as_matrix()
            return np.einsum('...ji,...jk,...kl', O, X, O)

        def pairmean(self, R, Q):
            assert R.shape == Q.shape

            return self.exp(R, 0.5 * self.log(R, Q))

        def elemdist(self, R, Q):
            """element-wise distance function"""
            assert R.shape == Q.shape

            versors = Rotation.from_matrix(np.einsum('...ij,...kj', Q, R)).as_rotvec()
            return np.sqrt(2)*la.norm(versors, axis=1)

        def dist(self, R, Q):
            """product distance function"""
            versors = Rotation.from_matrix(np.einsum('...ij,...kj', Q, R)).as_rotvec()
            return np.sqrt(2 * np.sum(versors**2))

        def projToGeodesic(self, R, Q, P, max_iter = 10):
            '''
            :arg X, Y: elements of SO(3)^k defining geodesic X->Y.
            :arg P: element of SO(3)^k to be projected to X->Y.
            :returns: projection of P to X->Y
            '''

            assert R.shape == Q.shape
            assert Q.shape == P.shape

            # all tagent vectors in common space i.e. algebra
            v = self.log(R, Q)
            v /= self.norm(R, v)

            # initial guess
            Pi = R

            # solver loop
            for _ in range(max_iter):
                w = self.log(Pi, P)
                d = self.inner(Pi, v, w)

                # print(f'|<v, w>|={d}')
                if abs(d) < 1e-6: break

                Pi = self.exp(Pi, d * v)

            return Pi

        def jacONB(self, R, Q):
            """Let J be the Jacobi operator along the geodesic from R to Q. This code diagonalizes J.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :returns lam, G: eigenvalues and orthonormal eigenbasis of Jac at R
            """
            assert R.shape == Q.shape
            k = self._M.k

            V = self.log(R, Q)
            nor_v = np.linalg.norm(V, ord='fro', axis=(1, 2))
            # normalize V
            V = vectime3d(nor_v, V)

            # (number of faces) x (number of basis elements) x 3 x 3
            F = np.zeros((k, 3, 3, 3))
            lam = np.zeros((k, 3))

            x = np.atleast_2d(V[..., 0, 1]).T
            y = np.atleast_2d(V[..., 0, 2]).T
            z = np.atleast_2d(V[..., 1, 2]).T

            # eigenvalues
            lam[:, 1:] = 1 / 4 * np.tile(x**2 + y**2 + z**2, (1, 2))

            # change sign for convenience
            e1 = np.tile(-self.versor2skew[2], (k, 1, 1))
            e2 = np.tile(self.versor2skew[1], (k, 1, 1))
            e3 = np.tile(-self.versor2skew[0], (k, 1, 1))

            # normalized eigenbasis unless x = 0 and z = 0
            F[:, 0, ...] = vectime3d(1 / self.elemnorm(R, V), V)
            F[:, 1, ...] = vectime3d(-z, e1) + vectime3d(x, e3)
            F[:, 2, ...] = vectime3d(-x * y, e1) + vectime3d(x**2 + z**2, e2) + vectime3d(- y * z, e3)

            # take care of division by 0 later
            with np.errstate(all='ignore'):
                F[:, 1, ...] = vectime3d(1 / self.elemnorm(R, F[:, 1, ...]), F[:, 1, ...])
                F[:, 2, ...] = vectime3d(1 / self.elemnorm(R, F[:, 2, ...]), F[:, 2, ...])

            # take care of special cases
            ind = np.nonzero(np.abs(x + z) < 1e-12)
            m = np.size(ind[0])
            if m > 0:
                for i in ind[0]:
                    F[i] = 1 / np.sqrt(2) * np.stack((e1[0], e2[0], e3[0]))

            return lam, F

        def jacop(self, R, Q, X):
            """ Evaluate the Jacobi operator along the geodesic from R to Q at r.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :param X: tangent vector at R
            :returns: tangent vector at R
            """
            assert R.shape == Q.shape == X.shape

            V = self.log(R, Q)
            # normalize tangent vectors
            for v in V:
                v = v / np.linalg.norm(v)

            return 1 / 4 * (-np.einsum('...ij,...jk,...kl', V, V, X) + 2 * np.einsum('...ij,...jk,...kl', V, X, V)
                            - np.einsum('...ij,...jk,...kl', X, V, V))

        def jacobiField(self, R, Q, t, X):
            """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from R to Q.
            :param R: element of the space of SO(3)^k
            :param Q: element of the space of SO(3)^k
            :param t: scalar in [0,1]
            :param X: tangent vector at R
            :return: tangent vector at gam(t)
            """
            assert R.shape == Q.shape == X.shape and np.isscalar(t)

            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                # gam(t)
                P = self.geopoint(R, Q, t)

                # orthonormal eigenvectors of the Jacobi operator that generate the parallel frame field along gam and the
                # eigenvalues
                lam, F = self.jacONB(R, Q)

                Fp = np.zeros_like(F)
                alpha = np.zeros_like(lam)

                for i in range(3):
                    # transport bases elements to gam(t)
                    Fp[:, i, ...] = self.transp(R, P, F[:, i, ...])
                    # expand X w.r.t. F at R
                    alpha[:, i] = self.eleminner(R, F[:, i, ...], X)

                # weights for the linear combination of the three basis elements
                weights = weightfun(lam, t, self.elemdist(R, Q))

                # evaluate the adjoint Jacobi field at R
                a = weights * alpha
                for i in range(3):
                    Fp[:, i, ...] = vectime3d(a[:, i], Fp[:, i, ...])

                return multiskew(np.sum(Fp, axis=1))

        def adjJacobi(self, R, Q, t, X):
            """Evaluates an adjoint Jacobi field for the geodesic gam from R to Q.
            :param R: element of the space of SO(3)^k
            :param Q: element of the space of SO(3)^k
            :param t: scalar in [0,1]
            :param X: tangent vector at gam(t)
            :return: tangent vector at R
            """
            assert R.shape == Q.shape == X.shape and np.isscalar(t)

            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                # gam(t)
                P = self.geopoint(R, Q, t)

                # orthonormal eigenvectors of the Jacobi operator that generate the parallel frame field along gam and the
                # eigenvalues
                lam, F = self.jacONB(R, Q)

                # expand X w.r.t. F at gam(t)
                Fp = np.zeros_like(F)
                alpha = np.zeros_like(lam)

                for i in range(3):
                    Fp[:, i, ...] = self.transp(R, P, F[:, i, ...])
                    alpha[:, i] = self.eleminner(P, Fp[:, i, ...], X)

                # weights for the linear combination of the three basis elements
                weights = weightfun(lam, t, self.elemdist(R, Q))

                # evaluate the adjoint Jacobi field at R
                a = weights * alpha
                for i in range(3):
                    F[:, i, ...] = vectime3d(a[:, i], F[:, i, ...])

                return multiskew(np.sum(F, axis=1))

        def lefttrans(self, R, Q):
            """Left-translation of R by Q"""
            return np.einsum('...ij,...jk', Q, R)

        def righttrans(self, R, Q):
            """Right translation of R by Q.
            """
            return np.einsum('...ij,...jk', R, Q)

        def dleft(self, f, X):
            """Derivative of the left translation by f at e applied to the tangent vector X.
            """
            return None

        def dright(self, f, X):
            """Derivative of the right translation by f at e applied to the tangent vector X.
            """
            return None

        def dleft_inv(self, f, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return None

        def dright_inv(self, f, X):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
            """
            return None

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return None

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return None

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return None

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            return None


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
    w = np.sin((1 - t) * np.multiply(np.sqrt(k).T, d).T) / np.where(k == 0, 1, np.sin(np.multiply(np.sqrt(k).T, d).T))

    return np.where(w == 0, 1 - t, w)


def exp_mat(U):
    """Matrix exponential, only use for normal matrices, i.e., square matrices U with U * U^T = U^T * UU"""
    vals, vecs = la.eig(U)
    vals = np.exp(vals)
    return np.real(np.einsum('...ij,...j,...kj', vecs, vals, vecs))


def log_mat(U):
    """Matrix logarithm, only use for normal matrices, i.e., square matrices U with U * U^T = U^T * UU"""
    vals, vecs = la.eig(U)
    vals = np.log(vals)

    return np.real(np.einsum('...ij,...j,...kj', vecs, vals, vecs))
