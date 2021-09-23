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
import numpy.random as rnd
import numpy.linalg as la

from scipy.linalg import logm, expm_frechet

from morphomatics.manifold import Manifold, Metric, Connection, LieGroup
from pymanopt.tools.multi import multisym


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
        point_shape = [self._k, self._d, self._d]
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def initLogEuclideanStructure(self):
        """
        Instantiate SPD(d)^k with log-Euclidean structure.
        """
        structure = SPD.LogEuclideanStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    class LogEuclideanStructure(Metric, Connection, LieGroup):
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

        @property
        def __str__(self):
            return "SPD(k, d)-canonical structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return np.sqrt(self._M.dim * 6)

        def inner(self, S, X, Y):
            """product metric"""
            return np.sum(np.einsum('...ij,...ij', X, Y))

        def eleminner(self, R, X, Y):
            """element-wise inner product"""
            return np.einsum('...ij,...ij', X, Y)

        def norm(self, S, X):
            """norm from product metric"""
            return np.sqrt(self.inner(S, X, X))

        def elemnorm(self, R, X):
            """element-wise norm"""
            return np.sqrt(self.eleminner(R, X, X))

        def proj(self, X, H):
            """orthogonal (with respect to the Euclidean inner product) projection of ambient
            vector ((k,3,3) array) onto the tangent space at X"""
            # return dlog(X, multisym(H))
            return multisym(H)

        def egrad2rgrad(self, X, D):
            # should be adj_dexp instead of dexp (however, dexp appears to be self-adjoint for symmetric matrices)
            return dexp(log_mat(X), multisym(D))

        def lefttrans(self, R, X):
            """Left-translation of X by R"""
            return self.exp(self.identity(), log_mat(R) + log_mat(X))

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            return

        def retr(self, R, X):
            # TODO
            return self.exp(R, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """

            X = argv[-1]
            Y = X if len(argv) == 1 else X + log_mat(argv[0])

            vals, vecs = la.eigh(Y)
            return np.einsum('...ij,...j,...kj', vecs, np.exp(vals), vecs)

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
            (depending on signature, i.e. whether footpoint is given as well)
            """

            X = log_mat(argv[-1])
            if len(argv) == 2: # Riemannian log
                X -= log_mat(argv[0])
            return multisym(X)

        def geopoint(self, S, T, t):
            """ Evaluate the geodesic from S to T at time t in [0, 1]"""
            assert S.shape == T.shape and np.isscalar(t)

            return self.exp(S, t * self.log(S, T))

        def identity(self):
            return np.tile(np.eye(3), (self._M.k, 1, 1))

        def transp(self, S, T, X):
            """Parallel transport for Sym+(d)^k.
            :param S: element of Symp+(d)^k
            :param T: element of Symp+(d)^k
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            assert S.shape == T.shape == X.shape

            # if X were not in algebra but at tangent space at S
            # return dexp(log_mat(T), dlog(S, X))

            return X

        def pairmean(self, S, T):
            assert S.shape == T.shape

            return self.exp(S, 0.5 * self.log(S, T))

        def elemdist(self, R, Q):
            """element-wise distance function"""
            assert R.shape == Q.shape

            return

        def dist(self, S, T):
            """Distance function in Sym+(d)^k"""
            return self.norm(S, self.log(S, T))

        def jacONB(self, R, Q):
            """Let J be the Jacobi operator along the geodesic from R to Q. This code diagonalizes J.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :returns lam, G: eigenvalues and orthonormal eigenbasis of Jac at R
            """

            return None

        def jacop(self, R, Q, X):
            """ Evaluate the Jacobi operator along the geodesic from R to Q at r.

            For the definition of the Jacobi operator see:
                Rentmeesters, Algorithms for data fitting on some common homogeneous spaces, p. 74.

            :param R: element of SO(3)^k
            :param Q: element of SO(3)^k
            :param X: tangent vector at R
            :returns: tangent vector at R
            """
            return None

        def jacobiField(self, S, T, t, X):
            """Evaluates a Jacobi field (with boundary conditions gam(0) = X, gam(1) = 0) along the geodesic gam from p to q.
            :param S: element of the space of Symp+(d)^k
            :param T: element of the space of Symp+(d)^k
            :param t: scalar in [0,1]
            :param X: tangent vector at S
            :return: tangent vector at gam(t)
            """
            assert S.shape == T.shape == X.shape and np.isscalar(t)

            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                U = self.geopoint(S, T, t)
                return (1 - t) * self.transp(S, U, X)

        def adjJacobi(self, S, T, t, X):
            """Evaluates the adjoint Jacobi field for the geodesic gam from S to T at S.
            :param S: element of the space of Symp+(d)^k
            :param T: element of the space of Symp+(d)^k
            :param t: scalar in [0,1]
            :param X: tangent vector at gam(t)
            :return: tangent vector at S
            """
            assert S.shape == T.shape == X.shape and np.isscalar(t)

            if t == 1:
                return np.zeros_like(X)
            elif t == 0:
                return X
            else:
                U = self.geopoint(S, T, t)
                return (1 - t) * self.transp(U, S, X)

        def adjDxgeo(self, S, T, t, X):
            """Evaluates the adjoint of the differential of the geodesic gamma from S to T w.r.t the starting point S at X,
            i.e, the adjoint  of d_S gamma(t; ., T) applied to X, which is en element of the tangent space at gamma(t).
            """
            assert S.shape == T.shape == X.shape and np.isscalar(t)

            return self.adjJacobi(S, T, t, X)

        def adjDygeo(self, S, T, t, X):
            """Evaluates the adjoint of the differential of the geodesic gamma from S to T w.r.t the endpoint T at X,
            i.e, the adjoint  of d_T gamma(t; S, .) applied to X, which is en element of the tangent space at gamma(t).
            """
            assert S.shape == T.shape == X.shape and np.isscalar(t)

            return self.adjJacobi(T, S, 1 - t, X)

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

        def righttrans(self, g, f):
            """Right translation of g by f.
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

    def projToGeodesic(self, X, Y, P, max_iter=10):
        '''
        :arg X, Y: elements of Symp+(d)^k defining geodesic X->Y.
        :arg P: element of Symp+(d)^k to be projected to X->Y.
        :returns: projection of P to X->Y
        '''

        assert X.shape == Y.shape
        assert Y.shape == P.shape
        assert self.connec
        assert self.metric

        # all tagent vectors in common space i.e. algebra
        v = self.connec.log(X, Y)
        v /= self.metric.norm(X, v)

        w = self.connec.log(X, P)
        d = self.metric.inner(X, v, w)

        return self.connec.exp(X, d * v)

    def rand(self):
        S = np.random.random((self._k, self._d, self._d))
        return np.einsum('...ij,...kj', S, S)

    def randvec(self, X):
        Y = self.rand()
        y = self.log(X, Y)
        return multisym(y / self.norm(X, y))

    def zerovec(self):
        return np.zeros((self._k, self._d, self._d))

def log_mat(U):
    """Matrix logarithm, only use for normal matrices U, i.e., U * U^T = U^T * U"""
    vals, vecs = la.eigh(U)
    vals = np.log(np.where(vals > 1e-10, vals, 1))
    return np.real(np.einsum('...ij,...j,...kj', vecs, vals, vecs))

def dexp(X, G):
    """Evaluate the derivative of the matrix exponential at
    X in direction G.
    """
    return np.array([expm_frechet(X[i],G[i])[1] for i in range(X.shape[0])])

def dlog(X, G):
    """Evaluate the derivative of the matrix logarithm at
    X in direction G.
    """
    n = X.shape[1]
    # set up [[X, G], [0, X]]
    W = np.hstack((np.dstack((X, G)), np.dstack((np.zeros_like(X), X))))
    return np.array([logm(W[i])[:n, n:] for i in range(X.shape[0])])
