################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2021 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see /LICENSE                                              #
#                                                                              #
################################################################################

import numpy as np

from scipy.spatial.transform import Rotation
from scipy.linalg import logm, expm
import numpy.linalg as la

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.rotations import randskew, randrot
from pymanopt.tools.multi import multiskew

class SO3(Manifold):
    """Returns the product manifold SO(3)^k, i.e., a product of k rotations in 3 dimensional space.

     manifold = SO3(k)

     Elements of SO(3)^k are represented as arrays of size kx3x3 where every 3x3 slice is an orthogonal matrix, i.e., a
     matrix R with R * R^T = eye(3) and det(R) = 1.

     The Riemannian metric used is the induced metric from the embedding space (R^nxn)^k, i.e., this manifold is a
     Riemannian submanifold of (R^3x3)^k endowed with the usual trace inner product.

     NOTE: Tangent vectors are represented in the Lie algebra, i.e., as skew symmetric matrices.
     """

    def __init__(self, k=1):
        if k == 1:
            self._name = 'Rotations manifold SO(3)'
        elif k > 1:
            self._name = 'Rotations manifold SO(3)^{k}'.format(k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k

        # tensor mapping versor to skew-sym. matrix
        I = np.eye(3)
        O = np.zeros(3)
        self.versor2skew = np.array([[O, -I[2], I[1]], [I[2], O, -I[0]], [-I[1], I[0], O]])
        # ... and the opposite direction
        self.skew2versor = .5 * self.versor2skew

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return 3 * self._k

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(3 * self._k)

    @property
    def k(self):
        return self._k

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

    def lefttrans(self, R, X):
        """Left-translation of X to the tangent space at R"""
        return np.einsum('...ij,...jk', R, X)

    def ehess2rhess(self, R, H):
        # TODO
        return

    def retr(self, R, X):
        # TODO
        return self.exp(R, X)

    def exp(self, R, X):
        """Riemannian exponential with base point R evaluated at X"""
        assert R.shape == X.shape

        a = R.ndim - 2
        versors = Rotation.from_rotvec(np.tensordot(X, self.skew2versor, axes=([a, a+1], [a, a+1])))
        return np.einsum('...ij,...jk', versors.as_matrix(), R)

    def log(self, R, Q):
        """Riemannian logarithm with base point R evaluated at Q"""
        assert R.shape == Q.shape

        versors = Rotation.from_matrix(np.einsum('...ij,...kj', Q, R)).as_rotvec()
        return np.einsum('ijk,...k', self.versor2skew, versors)

    def geopoint(self, R, Q, t):
        """Evaluate the geodesic from R to Q at time t in [0, 1]"""
        assert R.shape == Q.shape and np.isscalar(t)

        return self.exp(R, t * self.log(R, Q))

    def rand(self):
        return randrot(3, self._k)

    def randvec(self, X):
        U = randskew(3, self._k)
        nrmU = np.sqrt(np.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self, X):
        return np.zeros((self._k, 3, 3))

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
        """Let Jac be the Jacobi operator along the geodesic from X to Y. This code diagonalizes J.

        For the definition of the Jacobi operator see:
            Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

        :param R: element of SO(3)^k
        :param Q: element of SO(3)^k
        :returns lam, G: eigenvalues and orthonormal eigenbasis of  Jac at R
        """
        assert R.shape == Q.shape

        V = self.log(R, Q)

        # (number of faces) x (number of basis elements) x 3 x 3
        F = np.zeros((self._k, 3, 3, 3))
        lam = np.zeros((self._k, 3))

        x = np.atleast_2d(V[..., 0, 1]).T
        y = np.atleast_2d(V[..., 0, 2]).T
        z = np.atleast_2d(V[..., 1, 2]).T

        # eigenvalues
        lam[:, 1:] = 1 / 4 * np.tile(x**2 + y**2 + z**2, (1, 2))

        # change sign for convenience
        e1 = np.tile(-self.versor2skew[2], (self._k, 1, 1))
        e2 = np.tile(self.versor2skew[1], (self._k, 1, 1))
        e3 = np.tile(-self.versor2skew[0], (self._k, 1, 1))

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

        For the definition of the Jacobi operator see:
            Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

        :param R: element of SO(3)^k
        :param Q: element of SO(3)^k
        :param X: tangent vector at R
        :returns: tangent vector at R
        """
        assert R.shape == Q.shape == X.shape

        V = self.log(R, Q)
        return 1 / 4 * (-np.einsum('...ij,...jk,...kl', V, V, X) + 2 * np.einsum('...ij,...jk,...kl', V, X, V)
                        - np.einsum('...ij,...jk,...kl', X, V, V))

    def adjJacobi(self, R, Q, t, X):
        """Evaluates an adjoint Jacobi field along the geodesic gam from R to Q.
        :param R: element of the space of differential coordinates
        :param Q: element of the space of differential coordinates
        :param t: scalar in [0,1]
        :param X: tangent vector at gam(t)
        :return: tangent vector at X
        """
        assert R.shape == Q.shape == X.shape and np.isscalar(t)

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

        # weights for the linear combination of the three basis fields
        weights = weightfun(lam, t, self.elemdist(R, Q))

        # evaluate the adjoint Jacobi field at R
        a = weights * alpha
        for i in range(3):
            F[:, i, ...] = vectime3d(a[:, i], F[:, i, ...])

        return np.sum(F, axis=1)

    def adjDxgeo(self, R, Q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from R to Q w.r.t the starting point R at X,
        i.e, the adjoint  of d_R gamma(t; ., Q) applied to X, which is en element of the tangent space at gamma(t).
        """
        assert R.shape == Q.shape == X.shape and np.isscalar(t)

        return self.adjJacobi(R, Q, t, X)

    def adjDygeo(self, R, Q, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from R to Q w.r.t the endpoint Q at X,
        i.e, the adjoint  of d_Q gamma(t; R, .) applied to X, which is en element of the tangent space at gamma(t).
        """
        assert R.shape == Q.shape == X.shape and np.isscalar(t)

        return self.adjJacobi(Q, R, 1 - t, X)


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


def vectime3d(x, A):
    """
    :param x: vector of length k 
    :param A: array of size k x n x m
    :return: k x n x m array such that the j-th n x m slice of A is multiplied with the j-th element of x

    In case of k=1, x * A is returned.
    """
    if np.isscalar(x) and A.ndim == 2:
        return x * A

    x = np.atleast_2d(x)
    assert x.ndim <= 2 and np.size(A.shape) == 3
    assert x.shape[0] == 1 or x.shape[1] == 1
    assert x.shape[0] == A.shape[0] or x.shape[1] == A.shape[0]

    if x.shape[1] == 1:
        x = x.T
    A = np.einsum('kij->ijk', A)
    return np.einsum('ijk->kij', x * A)
