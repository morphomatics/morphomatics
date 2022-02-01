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

import copy

import numpy as np

from morphomatics.manifold import Manifold, Connection, LieGroup, SO3, GLpn


class SE3(Manifold):
    """Returns the product manifold SE(3)^k, i.e., a product of k rigid body motions.

     manifold = SE3(k)

     Elements of SE(3)^k are represented as arrays of size kx4x4 where every 4x4 slice are homogeneous coordinates of an
     element of SE(3), i.e., the upper-left 3x3 block is the rotational part, the upper-right 3x1 part is the
     translational part, and the lower row is [0 0 0 1]. Tangent vectors, consequently, follow the same ‘layout‘.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.
     """

    def __init__(self, k=1, structure='AffineGroup'):
        if k == 1:
            name = 'Rigid motions'
        elif k > 1:
            name = 'Special Euclidean group SE(3)^{k}'.format(k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k
        self._SO = SO3(k)

        dimension = 6 * self._k
        point_shape = [self._k, 4, 4]
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def initAffineGroupStructure(self):
        """
        Instantiate SE(3)^k with standard Lie group structure and canonical Cartan-Shouten connection.
        """
        structure = SE3.AffineGroupStructure(self)
        self._connec = structure
        self._group = structure

    @property
    def k(self):
        return self._k

    def rand(self):
        P = np.zero(self.point_shape)
        P[:, :3, :3] = self._SO.rand()
        P[:, :3, 3] = np.random.rand((self._k, 3))
        P[:, 3, 3] = 1
        return P

    def randvec(self, P):
        X = np.zero(self.point_shape)
        P[:, :3, :3] = self._SO.randvec(P[:, :3, :3])
        P[:, :3, 3] = np.random.rand((self._k, 3))
        return X

    def zerovec(self):
        return np.zeros(self.point_shape)

    class AffineGroupStructure(Connection, LieGroup):
        """
        Standard (product) Lie group structure on SE(3)^k. The connection used is the canonical Cartan-Shouten
        connection.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M
            self._GLp4 = GLpn(n=4, k=M.k)
            # SE(3) is subgroup of GL+(4) -> use methods of the ladder
            self._GLp4.initAffineGroupStructure()

        @property
        def __str__(self):
            return "SE3(k)-affine group structure"

        def lefttrans(self, P, S):
            """Left-translation of P by S"""
            return self._GLp4.group.lefttrans(P, S)

        def righttrans(self, P, S):
            """Right translation of P by S.
            """
            return self._GLp4.group.righttrans(P, S)

        def dleft(self, P, X):
            """Derivative of the left translation by P at the identity applied to the tangent vector X.
            """
            return self._GLp4.group.dleft(P, X)

        def dright(self, P, X):
            """Derivative of the right translation by P at the identity applied to the tangent vector X.
            """
            return self._GLp4.group.dright(P, X)

        def dleft_inv(self, P, X):
            """Derivative of the left translation by P^{-1} at f applied to the tangent vector X.
            """
            return self._GLp4.group.dleft_inv(P, X)

        def dright_inv(self, P, X):
            """Derivative of the right translation by P^{-1} at f applied to the tangent vector X.
            """
            return self._GLp4.group.dright_inv(P, X)

        def inverse(self, P):
            """Inverse map of the Lie group.
            """
            return self._GLp4.group.inverse(P)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x123 = np.stack((X[:, 0, 1], X[:, 0, 2], X[:, 1, 2]))
            x456 = X[:, :3, 3].transpose()
            x = np.concatenate((x123, x456), axis=0)
            return x.reshape((-1, 1), order='F')

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return self._GLp4.group.bracket(X, Y)

        def adjrep(self, P, X):
            """Adjoint representation of P applied to the tangent vector X at the identity.
            """
            return self._GLp4.group.adjrep(P, X)

        def retr(self, R, X):
            # TODO
            return self.exp(R, X)

        def exp(self,  *argv):
            """Computes the Lie-theoretic and connection exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            if len(argv) == 1:
                return self._GLp4.connec.exp(argv[0])
            elif len(argv) == 2:
                return self._GLp4.connec.exp(argv[0], argv[1])

        def log(self, *argv):
            """Computes the Lie-theoretic and connection logarithm map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            if len(argv) == 1:
                return self._GLp4.connec.log(argv[0])
            elif len(argv) == 2:
                return self._GLp4.connec.log(argv[0], argv[1])

        def geopoint(self, P, S, t):
            """Evaluate the geodesic from R to Q at time t in [0, 1]"""
            assert P.shape == S.shape and np.isscalar(t)

            return self.exp(P, t * self.log(P, S))

        def identity(self):
            """Identity element of SE(3)^k"""
            return np.tile(np.eye(4), (self._M.k, 1, 1))

        def transp(self, P, S, X):
            """Parallel transport for SE(3)^k.
            :param P: element of SE(3)^k
            :param S: element of SE(3)^k
            :param X: tangent vector at P
            :return: parallel transport of X at S
            """
            assert P.shape == S.shape == X.shape

            return

        def pairmean(self, P, S):
            assert P.shape == S.shape

            return self.exp(P, 0.5 * self.log(P, S))

        def jacONB(self, P, S):
            return

        def jacobiField(self, P, S, t, X):
            return

        def adjJacobi(self, P, S, t, X):
            return
