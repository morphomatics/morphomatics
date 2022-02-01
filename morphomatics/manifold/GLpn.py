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
from morphomatics.manifold import Manifold, LieGroup, Connection


class GLpn(Manifold):
    """Returns the product Lie group GL^+(n)^k, i.e., a product of k n-by-n matrices each with positive determinant.

     manifold = GLpn(n, k)

     Elements of GL^+(n)^k are represented as arrays of size kxnxn.

     # NOTE: Tangent vectors are represented as left translations in the Lie algebra, i.e., a tangent vector X at g is
     represented as as d_gL_{g^(-1)}(X)
     """

    def __init__(self, n=3, k=1, structure='AffineGroup'):
        self._n = n
        self._k = k

        if k == 1:
            name = 'Orientation preserving maps of R^n'
        elif k > 1:
            name = '{k}-tuple of orientation preserving maps of R^{n}'.format(k=k, n=n)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        super().__init__(name, k * n**2, point_shape=(k, n, n))

        if structure:
            getattr(self, f'init{structure}Structure')()

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    def rand(self):
        """Returns a random point in the Lie group. This does not
        follow a specific distribution."""
        A = np.random.rand(self.k, self.n, self.n)
        return A + np.tile(np.eye(self.n), (self.k, 1, 1))

    def randvec(self, A):
        """Returns a random vector in the tangent space at A.
        """
        return np.random.rand(self.k, self.n, self.n)

    def zerovec(self):
        """Returns the zero vector in the tangent space at g."""
        return np.zeros((self.k, self.n, self.n))

    def initAffineGroupStructure(self):
        """
        Standard group structure with canonical Cartan Shouten Connction.
        """
        structure = GLpn.AffineGroupStructure(self)
        self._connec = structure
        self._group = structure

    class AffineGroupStructure(Connection, LieGroup):
        """
        Standard group structure on GL+n(k) where the composition of two elements is given by component-wise matrix
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
            return 'standard group structure on GL+(n)^k with CCS connection'

        # Group

        def identity(self):
            """Returns the identity element e of the Lie group."""
            return np.tile(np.eye(self._G.n), (self._G.k, 1, 1))

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return np.einsum('kij,kjl->kil', X, Y) - np.einsum('kij,kjl->kil', Y, X)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            return np.reshape(X, (self._G.k * self._G.n**2, 1))

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
            """Computes the Lie-theoretic and connection exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            X = argv[-1]
            g = np.array([expm(x) for x in X])
            if len(argv) == 1: # group exp
                return g
            elif len(argv) == 2: # exp of CCS connection
                return self.lefttrans(g, argv[0])

        def log(self, *argv):
            """Computes the Lie-theoretic and connection logarithm map
            (depending on signature, i.e. whether footpoint is given as well)
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
