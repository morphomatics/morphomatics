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
from scipy.linalg import logm, expm
from morphomatics.manifold import Manifold, LieGroup, Connection, Metric

from morphomatics.manifold.util import vectime3d


class GLp3(Manifold):
    """Returns the product Lie group GL^+(3)^k, i.e., a product of k 3-by-3 matrices each with positive determinant.

     manifold = GLp3(k)

     Elements of GL^+(3)^k are represented as arrays of size kx3x3.

     # NOTE: Tangent vectors are represented as left translations in the Lie algebra, i.e., a tangent vector X at g is
     represented as as d_gL_{g^(-1)}(X)
     """

    def __init__(self, k=1, structure='AffineGroup'):
        self._k = k

        if k == 1:
            name = 'Orientation preserving maps of R^3'
        elif k > 1:
            name = '{k}-tuple of orientation preserving maps of R^3'.format(k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        super().__init__(name, k*9, point_shape=(k, 3, 3))

        if structure:
            getattr(self, f'init{structure}Structure')()

    @property
    def k(self):
        return self._k

    def rand(self):
        """Returns a random point in the Lie group. This does not
        follow a specific distribution."""
        A = np.random.rand(self.k, 3, 3)
        return A + np.tile(np.eye(3), (self._k, 1, 1))

    def randvec(self, A):
        """Returns a random vector in the tangent space at A.
        """
        return np.random.rand(self.k, 3, 3)

    def zerovec(self):
        """Returns the zero vector in the tangent space at g."""
        return np.zeros((self.k, 3, 3))

    def initAffineGroupStructure(self):
        """
        Standard group structure with canonical Cartan Shouten Connction.
        """
        structure = GLp3.AffineGroupStructure(self)
        self._connec = structure
        self._group = structure

    class AffineGroupStructure(Connection, LieGroup):
        """
        Standard group structure on GL+3(k) where the composition of two elements is given by component-wise matrix
        multiplication. The connection is the corresponding canonical Cartan Shouten connection. No Riemannian metric is
        used.
        """

        def __init__(self, G):
            """
            Constructor.
            """
            self._G = G

        @property
        def __str__(self):
            return 'standard group structure on GL+(3)^k with CCS connection'

        # Group

        def identity(self):
            """Returns the identity element e of the Lie group."""
            return np.tile(np.eye(3), (self._G.k, 1, 1))

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return np.einsum('kij,kjl->kil', X, Y) - np.einsum('kij,kjl->kil', Y, X)


        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return np.reshape(X, (self._G._k * 9, 1))

        def lefttrans(self, g, f):
            """Left translation of g by f.
            """
            return np.einsum('kij,kjl->kil', f, g)

        def righttrans(self, g, f):
            """Right translation of g by f.
            """
            return np.einsum('kij,kjl->kil', g, f)

        def inverse(self, g):
            """Inverse map of the Lie group.
            """
            return np.linalg.inv(g)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            X = argv[-1]
            g = np.array([expm(x) for x in X])
            if len(argv) == 1: # group exp
                return g
            elif len(argv) == 2: # exp of CCS connection
                return self.lefttrans(g, argv[0])

        def log(self, *argv):
            """Computes the Lie-theoretic logarithm (i.e., the element-wise matrix logarithm) of g. This is the inverse of
            `groupexp`.
            """
            g = argv[-1]
            if len(argv) == 1: # group log
                return np.array([logm(a) for a in g])
            elif len(argv) == 2: # log of CCS connection
                return np.array([logm(a) for a in self.lefttrans(g, self.inverse(argv[0]))])

        def dleft(self, f, X):
            """Derivative of the left translation by f applied to the tangent vector X at the identity.
            """
            return np.einsum('kij,kjl->kil', f, X)

        def dright(self, f, X):
            """Derivative of the right translation by f at g applied to the tangent vector X.
            """
            return np.einsum('kij,kjl->kil', X, f)

        def dleft_inv(self, f, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return np.einsum('kij,kjl->kil', self.inverse(f), X)

        def dright_inv(self, f, X):
            """Derivative of the right translation by f^{-1} at f applied to the tangent vector X.
            """
            return np.einsum('kij,kjl->kil', X, self.inverse(f))

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            return np.einsum('kij,kjl,klm->kim', g, X, self.inverse(g))

        def geopoint(self, g, h, t):
            """
            Evaluate the geodesic from g to h at time t.
            """
            return self.exp(g, t * self.log(g, h))

        def transp(self, p, q, X):
            #TODO
            return

        def jacobiField(self, p, q, t, X):
            #TODO
            return

        def adjJacobi(self, p, q, t, X):
            #TODO
            return
