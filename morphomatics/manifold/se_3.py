################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2024 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

import jax
import jax.numpy as jnp

from morphomatics.manifold import Manifold, Connection, LieGroup, Metric, SO3, GLpn, Euclidean
from morphomatics.manifold.so_3 import logm as SO3_logm, expm as SO3_expm
from morphomatics.manifold.util import multiskew

# SE(3) is affine submanifold of GL+(4) -> (re)use methods
GLp4 = GLpn(n=4, structure='AffineGroup')
# also reusing methods from rotations and R^3 for product structure
SO = SO3()
R3 = Euclidean()

class SE3(Manifold):
    """Returns the product manifold SE(3), i.e., rigid body motions.

     manifold = SE3()

     Elements of SE(3) are represented as matrices of size 4x4, where the upper-left 3x3 block is the rotational part,
     the upper-right 3x1 part is the translational part, and the lower row is [0 0 0 1]. Tangent vectors, consequently,
     follow the same ‘layout‘.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.
     """

    def __init__(self, structure='AffineGroup'):
        name = 'Rigid motions'
        dimension = 6
        point_shape = (4, 4)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        obj = cls(structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initAffineGroupStructure(self):
        """
        Instantiate SE(3) with standard Lie group structure and canonical Cartan-Shouten connection.
        """
        self._connec = SE3.CartanShoutenConnection(self)
        self._group = SE3.GroupStructure(self)

    def initCanonicalRiemannianStructure(self):
        """
        Instantiate SE(3) with standard canonical left invariant Riemannian metric and the corresponding bi-invariant
        connection.
        """
        self._metric = SE3.CanonicalRiemannianStructure(self)
        self._connec = SE3.CanonicalRiemannianStructure(self)
        self._group = SE3.GroupStructure(self)

    def rand(self, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        return jnp.zeros(self.point_shape) \
                   .at[:3, :3].set(SO.rand(k1)) \
                   .at[:3, 3].set(jax.random.normal(k2, (3,))) \
            .at[3, 3].set(1)

    def randvec(self, P, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        return jnp.zeros(self.point_shape) \
                   .at[:3, :3].set(SO.randvec(P[:3, :3], k1)) \
                   .at[:3, 3].set(jax.random.normal(k2, (3,)))

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, P, X):
        X = X.at[:3, :3].set(SO.proj(P[:3, :3], X[:3, :3]))
        return X.at[3, :].set(0)

    def get_so3(self, P):
        """get SO3-part of P in SE3 or se3"""
        return P[:3, :3]

    def get_r3(self, P):
        """get R3-part of P in SE3 or se3"""
        return P[:3, -1].squeeze()


    def homogeneous_coords(self, M, X):
       """create SE3-element from M in SO3 and X in R^3"""
       P = jnp.zeros(self.point_shape)
       P = P.at[:3, :3].set(M)
       P = P.at[:3, -1].set(X)
       P = P.at[3, 3].set(1.)
       return P

    class GroupStructure(LieGroup):
        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "Semi-direct (product) group structure"

        @property
        def identity(self):
            """Identity element of SE(3)"""
            return jnp.eye(4)

        def lefttrans(self, P, S):
            """Left-translation of P by S"""
            return GLp4.group.lefttrans(P, S)

        def righttrans(self, P, S):
            """Right translation of P by S.
            """
            return GLp4.group.righttrans(P, S)

        def dleft(self, P, X):
            """Derivative of the left translation by P at the identity applied to the tangent vector X.
            """
            return GLp4.group.dleft(P, X)

        def dright(self, P, X):
            """Derivative of the right translation by P at the identity applied to the tangent vector X.
            """
            return GLp4.group.dright(P, X)

        def dleft_inv(self, P, X):
            """Derivative of the left translation by P^{-1} at f applied to the tangent vector X.
            """
            return GLp4.group.dleft_inv(P, X)

        def dright_inv(self, P, X):
            """Derivative of the right translation by P^{-1} at f applied to the tangent vector X.
            """
            return GLp4.group.dright_inv(P, X)

        def inverse(self, P):
            """Inverse map of the Lie group.
            """
            Rt = jnp.einsum('ij->ji', P[:3, :3])
            return P.at[:3, :3].set(Rt) \
                       .at[:3, 3].set(jnp.einsum('ij,j', -Rt, P[:3, 3]))
            # return GLp4.group.inverse(P)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x123 = jnp.stack((X[0, 1], X[0, 2], X[1, 2])) * 2 ** .5
            x456 = X[:3, 3].transpose()
            x = jnp.concatenate((x123, x456), axis=0)
            return x.reshape((-1, 1), order='F')

        def coords_inverse(self, X):
            x123 = X[:3]
            x456 = X[3:]
            Y = self._M.group.identity
            Y = Y.at[:3, :3].set(self._SO3.group.coords_inverse(x123))
            Y = Y.at[:3, -1].set(x456)
            return Y

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return GLp4.group.bracket(X, Y)

        def adjrep(self, P, X):
            """Adjoint representation of P applied to the tangent vector X at the identity.
            """
            return GLp4.group.adjrep(P, X)

        def exp(self, X):
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

        def __str__(self):
            return "Canonical Cartan-Shouten connection"

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, S, X):
            """Computes connection exponential map
            """
            return jnp.einsum('...ij,...jk', expm(X), S)

        def log(self, P, S):
            """Computes the connection logarithm map
            """
            return logm(jnp.einsum('...ij,...jk', S, GLp4.group.inverse(P)))

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise GLp4.connec.curvature_tensor(p, X, Y, Z)

        def transp(self, P, S, X):
            """Parallel transport for SE(3)^k.
            :param P: element of SE(3)^k
            :param S: element of SE(3)^k
            :param X: tangent vector at P
            :return: parallel transport of X at S
            """
            return GLp4.connec.transp(P, S, X)

        def pairmean(self, P, S):
            return self.exp(P, 0.5 * self.log(P, S))

        def jacobiField(self, R, Q, t, X):
            # ### using (forward-mode) automatic differentiation of geopoint(..)
            f = lambda O: self.geopoint(O, Q, t)
            geo, J = jax.jvp(f, (R,), (self._M.group.dright(R, X),))
            J = self._M.group.dright_inv(geo, J)
            J = J.at[:-1, :-1].set(multiskew(J[:-1, :-1]))
            return geo, J

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

        def __str__(self):
            return "Canonical left invariant metric"

        @property
        def typicaldist(self):
            return jnp.pi * jnp.sqrt(3)

        def inner(self, R, X, Y):
            """Product of canonical bi-invariant metrics of SO(3) and R3"""
            return jnp.sum(jnp.einsum('ij,ij', X, Y))

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
            return Y.at[:3, :3].set(SO.metric.egrad2rgrad(S[:3, :3], X[:3, :3]))

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, S, X):
            """Computes the Riemannian exponential map
            """
            P = jnp.zeros_like(S)
            P = P.at[3, 3].set(1)
            P = P.at[:3, :3].set(SO.connec.exp(S[:3, :3], X[:3, :3]))
            return P.at[:3, 3].set(S[:3, 3] + X[:3, 3])

        def log(self, S, P):
            """Computes the Riemannian exponential map

                Note that tangent vectors are always represented in the Lie Algebra.Thus, the Riemannian and group
                operation coincide.
            """
            X = jnp.zeros_like(S)
            X = X.at[:3, :3].set(SO.connec.log(S[:3, :3], P[:3, :3]))
            return X.at[:3, 3].set(P[:3, 3] - S[:3, 3])

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at S on the vectors X, Y, Z. With nabla_X Y denoting
            the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            V = jnp.zeros_like(X)
            # translational part it flat
            return V.at[:3, :3].set(SO.connec.curvature_tensor(S[:3, :3], X[:3, :3], Y[:3, :3], Z[:3, :3]))

        def transp(self, S, P, X):
            """Parallel transport for the canonical Riemannian structure.
            :param R: element of SE(3)
            :param Q: element of SE(3)
            :param X: tangent vector at R
            :return: parallel transport of X at Q
            """
            Y = X
            # translational part has the identity as parallel transport
            return Y.at[:3, :3].set(SO.connec.transp(S[:3, :3], P[:3, :3], X[:3, :3]))

        def pairmean(self, S, P):
            return self.exp(S, 0.5 * self.log(S, P))

        def dist(self, S, P):
            """product distance function"""
            return jnp.sqrt(self.squared_dist(S, P))

        def squared_dist(self, S, P):
            """product squared distance function"""
            return (SO.metric.squared_dist(S[:3, :3], P[:3, :3])
                    + R3.metric.squared_dist(S[:3, 3], P[:3, 3]))

        def projToGeodesic(self, S, P, Q, max_iter=10):
            '''
            :arg S, P: elements of SE(3) defining geodesic S->P.
            :arg Q: element of SE(3) to be projected to S->P.
            :returns: projection of Q to S->P
            '''
            Pi = jnp.zeros_like(S)
            Pi = Pi.at[:3, :3].set(SO.metric.projToGeodesic(S[:3, :3], P[:3, :3], Q[:3, :3], max_iter))
            return Pi.at[:3, 3].set(R3.metric.projToGeodesic(S[:3, 3], P[:3, 3], Q[:3, 3], max_iter))

        def jacobiField(self, S, P, t, X):
            J = jnp.zeros_like(X)
            J = J.at[:3, :3].set(SO.connec.jacobiField(S[:3, :3], P[:3, :3], t, X[:3, :3]))
            return J.at[:3, 3].set(R3.connec.jacobiField(S[:3, 3], P[:3, 3], t, X[:3, 3]))

        def adjJacobi(self, S, P, t, X):
            J = jnp.zeros_like(X)
            J = J.at[:3, :3].set(SO.metric.adjJacobi(S[:3, :3], P[:3, :3], t, X[:3, :3]))
            return J.at[:3, 3].set(R3.metric.adjJacobi(S[:3, 3], P[:3, 3], t, X[:3, 3]))



def logm(P):
    """
    Blanco, J. L. (2010). A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
    University of Malaga, Tech. Rep, 3, 6.
    """
    w = SO3_logm(P[:3, :3])

    theta2 = .5 * jnp.sum(w ** 2)
    theta = jnp.sqrt(theta2 + jnp.finfo(jnp.float64).eps)

    Vinv = (jnp.eye(3) - .5 * w
            + jax.lax.cond(theta < 1e-6,
                           lambda _, _theta2: 1 / 12 + _theta2 / 720 + _theta2 ** 2 / 30240,
                           lambda _theta, _theta2: (1 - jnp.cos(.5 * _theta) / jnp.sinc(
                               .5 * _theta / jnp.pi)) / _theta2,
                           theta,
                           theta2)
            * (w @ w))

    return P.at[:3, :3].set(w) \
            .at[:3, 3].set(jnp.einsum('ij,j', Vinv, P[:3, 3])) \
            .at[3, 3].set(0)


def expm(X):
    """
    Blanco, J. L. (2010). "A tutorial on SE(3) transformation parameterizations and on-manifold optimization."
    University of Malaga, Tech. Rep, 3, 6.
    """
    R = SO3_expm(X[:3, :3])

    theta2 = .5 * jnp.sum(X[:3, :3] ** 2)
    theta = jnp.sqrt(theta2 + jnp.finfo(jnp.float64).eps)

    V = (jnp.eye(3)
         + jax.lax.cond(theta < 1e-6,
                        lambda _, _theta2: .5 - _theta2 / 24 + _theta2 ** 2 / 720,
                        lambda _theta, _theta2: (1.0 - jnp.cos(_theta)) / _theta2, theta, theta2)
         * X[:3, :3]
         + jax.lax.cond(theta < 1e-6,
                        lambda _, _theta2: 1 / 6 - _theta2 / 120 + _theta2 ** 2 / 5040,
                        lambda _theta, _theta2: (_theta - jnp.sin(_theta)) / (_theta2 * _theta), theta, theta2)
         * (X[:3, :3] @ X[:3, :3]))

    return X.at[:3, :3].set(R) \
            .at[:3, 3].set(jnp.einsum('ij,j', V, X[:3, 3])) \
            .at[3, 3].set(1)
