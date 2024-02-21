################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Connection, LieGroup, Metric, SO3, GLpn, Euclidean
from morphomatics.manifold.SO3 import logm as SO3_logm, expm as SO3_expm


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
        self._connec = SE3.CartanShoutenConnection(self)
        self._group = SE3.GroupStructure(self)

    def initCanonicalRiemannianStructure(self):
        """
        Instantiate SE(3)^k with standard canonical left invariant Riemannian metric and the corresponding bi-invariant
        connection.
        """
        self._metric = SE3.CanonicalRiemannianStructure(self)
        self._connec = SE3.CanonicalRiemannianStructure(self)
        self._group = SE3.GroupStructure(self)

    @property
    def k(self):
        return self._k

    def rand(self, key: jax.random.KeyArray):
        k1, k2 = jax.random.split(key, 2)
        return jnp.zeros(self.point_shape)                   \
            .at[:, :3, :3].set(self._SO.rand(k1))             \
            .at[:, :3, 3].set(jax.random.normal(k2, (self._k, 3))) \
            .at[:, 3, 3].set(1)

    def randvec(self, P, key: jax.random.KeyArray):
        k1, k2 = jax.random.split(key, 2)
        return jnp.zeros(self.point_shape)                       \
            .at[:, :3, :3].set(self._SO.randvec(P[:, :3, :3], k1))  \
            .at[:, :3, 3].set(jax.random.normal(k2, (self._k, 3)))

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, P, X):
        X = X.at[:, :3, :3].set(self._SO.proj(P[:, :3, :3], X[:, :3, :3]))
        return X.at[:, 3, :].set(0)

    class GroupStructure(LieGroup):
        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M
            self._GLp4 = GLpn(n=4, k=M.k, structure='AffineGroup')

        def __str__(self):
            return "Semi-direct (product) group structure"

        @property
        def identity(self):
            """Identity element of SE(3)^k"""
            return jnp.tile(jnp.eye(4), (self._M.k, 1, 1))

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
            Rt = jnp.einsum('...ij->...ji', P[:, :3, :3])
            return P.at[:, :3, :3].set(Rt) \
                .at[:, :3, 3].set(jnp.einsum('...ij,...j', -Rt, P[:, :3, 3]))
            # return self._GLp4.group.inverse(P)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x123 = jnp.stack((X[:, 0, 1], X[:, 0, 2], X[:, 1, 2])) * 2**.5
            x456 = X[:, :3, 3].transpose()
            x = jnp.concatenate((x123, x456), axis=0)
            return x.reshape((-1, 1), order='F')

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return self._GLp4.group.bracket(X, Y)

        def adjrep(self, P, X):
            """Adjoint representation of P applied to the tangent vector X at the identity.
            """
            return self._GLp4.group.adjrep(P, X)

        def exp(self,  X):
            """Computes the Lie-theoretic exponential map at X
            """
            return expm(X)

        def log(self, S):
            """Computes the Lie-theoretic logarithm map at S
            """
            return logm(S)

    class CartanShoutenConnection(Connection):
        """
        Canonical Cartan-Shouten connection on SE(3) connection.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M
            # SE(3) is affine submanifold of GL+(4) -> use methods of the latter
            self._GLp4 = GLpn(n=4, k=M.k, structure='AffineGroup')

        def __str__(self):
            return "Canonical Cartan-Shouten connection"

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self,  S, X):
            """Computes connection exponential map
            """
            return jnp.einsum('...ij,...jk', expm(X), S)

        def log(self, P, S):
            """Computes the connection logarithm map
            """
            return logm(jnp.einsum('...ij,...jk', S, self._GLp4.group.inverse(P)))

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise self._GLp4.connec.curvature_tensor(p, X, Y, Z)

        def transp(self, P, S, X):
            """Parallel transport for SE(3)^k.
            :param P: element of SE(3)^k
            :param S: element of SE(3)^k
            :param X: tangent vector at P
            :return: parallel transport of X at S
            """
            return self._GLp4.connec.transp(P, S, X)

        def pairmean(self, P, S):
            return self.exp(P, 0.5 * self.log(P, S))

        def jacobiField(self, P, S, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

    class CanonicalRiemannianStructure(Metric):
        """
        Standard (product) Riemannian structure with the canonical right invariant metric that is the product of the
        canonical (bi-invariant) metrics on SO(3) and R^3. The resulting geodesics are products of their geodesics. For
        a reference, see, e.g.,

        Zefran et al., "Choice of Riemannian Metrics for Rigid Body Kinematics."

        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M
            # SE(3) is subgroup of GL+(4) -> use group methods of the latter
            self._SO3 = SO3(k=M.k)
            self._R3 = Euclidean()

        def __str__(self):
            return "Canonical right invariant metric"

        @property
        def typicaldist(self):
            return jnp.pi * jnp.sqrt(3 * self._M.k)

        def inner(self, R, X, Y):
            """Product of canonical bi-invariant metrics of SO(3) and R3"""
            return jnp.sum(jnp.einsum('...ij,...ij', X, Y))

        def flat(self, R, X):
            """Lower vector X at R with the metric"""
            # return X
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, R, dX):
            """Raise covector dX at R with the metric"""
            # return dX
            raise NotImplementedError('This function has not been implemented yet.')

        def egrad2rgrad(self, S, X):
            Y = X
            # translational part is already "Riemannian"
            return Y.at[:, :3, :3].set(self._SO3.metric.egrad2rgrad(S[:, :3, :3], X[:, :3, :3]))



        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, S, X):
            """Computes the Riemannian logarithmic map

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            P = jnp.zeros_like(S)
            P = P.at[:, :3, :3].set(self._SO3.connec.exp(S[:, :3, :3], X[:, :3, :3]))
            return P.at[:, :3, 3].set(S[:, :3, 3] + X[:, :3, 3])

        def log(self, S, P):
            """Computes the Riemannian exponential map

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            X = jnp.zeros_like(S)
            X = X.at[:, :3, :3].set(self._SO3.connec.log(S[:, :3, :3], P[:, :3, :3]))
            return X.at[:, :3, 3].set(P[:, :3, 3] - S[:, :3, 3])

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at S on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            V = jnp.zeros_like(X)
            # translational part it flat
            return V.at[:, :3, :3].set(self._SO3.connec.curvature_tensor(S[:, :3, :3], X[:, :3, :3], Y[:, :3, :3], Z[:, :3, :3]))

        def transp(self, S, P, X):
            """Parallel transport for the canonical Riemannian structure.
            :param R: element of SE(3)^k
            :param Q: element of SE(3)^k
            :param X: tangent vector at R
            :return: parallel transport of X at Q
            """
            Y = X
            # translational part has the identity as parallel transport
            return Y.at[:, :3, :3].set(self._SO3.connec.transp(S[:, :3, :3], P[:, :3, :3], X[:, :3, :3]))

        def pairmean(self, S, P):
            return self.exp(S, 0.5 * self.log(S, P))

        def dist(self, S, P):
            """product distance function"""
            return jnp.sqrt(self.squared_dist(S, P))

        def squared_dist(self, S, P):
            """product squared distance function"""
            return (self._SO3.metric.squared_dist(S[:, :3, :3], P[:, :3, :3])
                    + self._R3.metric.squared_dist(S[:, :3, 3], P[:, :3, 3]))

        def projToGeodesic(self, S, P, Q, max_iter=10):
            '''
            :arg S, P: elements of SE(3)^k defining geodesic S->P.
            :arg Q: element of SE(3)^k to be projected to S->P.
            :returns: projection of Q to S->P
            '''
            Pi = jnp.zeros_like(S)
            Pi = Pi.at[:, :3, :3].set(self._SO3.metric.projToGeodesic(S[:, :3, :3], P[:, :3, :3], Q[:, :3, :3], max_iter))
            return Pi.at[:, :3, 3].set(self._R3.metric.projToGeodesic(S[:, :3, 3], P[:, :3, 3], Q[:, :3, 3], max_iter))

        def jacobiField(self, S, P, t, X):
            J = jnp.zeros_like(X)
            J = J.at[:, :3, :3].set(self._SO3.connec.jacobiField(S[:, :3, :3], P[:, :3, :3], t, X[:, :3, :3]))
            return J.at[:, :3, 3].set(self._R3.connec.jacobiField(S[:, :3, 3], P[:, :3, 3], t, X[:, :3, 3]))

        def adjJacobi(self, S, P, t, X):
            J = jnp.zeros_like(X)
            J = J.at[:, :3, :3].set(self._SO3.metric.adjJacobi(S[:, :3, :3], P[:, :3, :3], t, X[:, :3, :3]))
            return J.at[:, :3, 3].set(self._R3.metric.adjJacobi(S[:, :3, 3], P[:, :3, 3], t, X[:, :3, 3]))

def logm(P):
    """
    Blanco, J. L. (2010). A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
    University of Malaga, Tech. Rep, 3, 6.
    """
    w = SO3_logm(P[:, :3, :3])

    theta2 = .5 * jnp.sum(w ** 2, axis=(-1, -2))
    theta = jnp.sqrt(theta2 + jnp.finfo(jnp.float64).eps)

    Vinv = jnp.eye(3) - .5 * w + jnp.where(theta < 1e-6, 1/12 + theta2/720 + theta2**2/30240,
                                           (1 - jnp.cos(.5 * theta) / jnp.sinc(.5 * theta / jnp.pi)) / theta2) * (w @ w)

    return P.at[:, :3, :3].set(w) \
               .at[:, :3, 3].set(jnp.einsum('...ij,...j', Vinv, P[:, :3, 3])) \
               .at[:, 3, 3].set(0)


def expm(X):
    """
    Blanco, J. L. (2010). "A tutorial on SE(3) transformation parameterizations and on-manifold optimization."
    University of Malaga, Tech. Rep, 3, 6.
    """
    R = SO3_expm(X[:, :3, :3])

    theta2 = .5 * jnp.sum(X[:, :3, :3] ** 2, axis=(-1,-2))
    theta = jnp.sqrt(theta2 + jnp.finfo(jnp.float64).eps)

    V = jnp.eye(3) \
        + jnp.where(theta < 1e-6, .5 - theta2/24 + theta2**2/720, (1.0 - jnp.cos(theta)) / theta2) * X[:, :3, :3] \
        + jnp.where(theta < 1e-6, 1/6 - theta2/120 + theta2**2/5040, (theta - jnp.sin(theta)) / (theta2 * theta)) \
        * (X[:, :3, :3] @ X[:, :3, :3])

    return X.at[:, :3, :3].set(R) \
               .at[:, :3, 3].set(jnp.einsum('...ij,...j', V, X[:, :3, 3])) \
               .at[:, 3, 3].set(1)
