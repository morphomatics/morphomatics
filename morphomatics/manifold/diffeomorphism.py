import numpy as np
import jax
import jax.numpy as jnp

from jax.experimental.ode import odeint

from morphomatics.manifold import Manifold, LieGroup
from .util import LazyKernel


class Diffeomorphism(Manifold):
    """The diffeomorphism group, i.e. manifold of smooth invertible automorphisms of ambient space.

    Diffeomorphisms model plausible deformations of the ambiant space
    and, hence, any objects embedded therein. The parameter is a
    discrete vector field (referred to as the momentum). The diffeomorhism is
    given by integrating the smooth vector field obtained from the momentum by (kernel) smoothing.
    """

    def __init__(self, control_pts: jax.Array, scale: float = 0.1, structure="Group"):
        """Initialize the Diffeomorphism manifold.
        Args:
            control_pts (jax.Array): Control points defining the diffeomorphism.
            scale (float): Scale of the kernel.
        """

        if control_pts.ndim != 2:
            raise ValueError("Control points must be a 2D array (shape: [n_points, n_features]).")
        self._control_pts = control_pts
        self._scale = scale
        dim = np.prod(control_pts.shape)

        name = 'Diffeomorphism group'
        dimension = None
        super().__init__(name, dim, point_shape=control_pts.shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children + (self._control_pts, self._scale), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe for PyTree registration."""
        *children, pts, scale = children
        obj = cls(pts, scale, structure=None)
        obj.tree_unflatten_instance(aux_data, children)
        return obj

    def initGroupStructure(self):
        """Initialize the group structure of the diffeomorphism manifold."""
        self._group = Diffeomorphism.GroupStructure(self)

    def proj(self, p, X):
        return X

    def rand(self, key: jax.Array):
        return jax.random.normal(key, self.point_shape)

    def randvec(self, X, key: jax.Array):
        return jax.random.normal(key, self.point_shape)

    def zerovec(self, X):
        return jnp.zeros_like(X)

    class GroupStructure(LieGroup):
        """Group structure for the diffeomorphism manifold."""

        def __init__(self, M: 'Diffeomorphism'):
            super().__init__(M)

        def __str__(self):
            return "Group structure on diffeomorphism manifold"

        @property
        def identity(self):
            return jnp.zeros_like(self._M._control_pts)

        def coords(self, X):
            return X

        def coords_inv(self, X):
            return X

        def bracket(self, X, Y):
            raise NotImplementedError("Bracket operation is not implemented for diffeomorphisms.")

        def lefttrans(self, g, f):
            raise NotImplementedError("Left translation is not implemented for diffeomorphisms.")

        def righttrans(self, g, f):
            raise NotImplementedError("Right translation is not implemented for diffeomorphisms.")

        def inverse(self, g):
            raise NotImplementedError("Inverse is not implemented for diffeomorphisms.")

        def exp(self, *argv):
            if len(argv) == 2:
                raise NotImplementedError("CCS connection exponential map is not implemented for diffeomorphisms.")
            # we represent diffeomorphisms via momenta, so just return input momentum
            return argv[0]

        def log(self, *argv):
            if len(argv) == 2:
                raise NotImplementedError("CCS connection logarithmic map is not implemented for diffeomorphisms.")
            # we represent diffeomorphisms via momenta, so just return input diffeomorphism
            return argv[0]

        def retr(self, p, X):
            raise NotImplementedError("Retraction is not implemented for diffeomorphisms.")

        def curvature_tensor(self, f, X, Y, Z):
            raise NotImplementedError("Curvature tensor is not implemented for diffeomorphisms.")

        def transp(self, f, g, X):
            raise NotImplementedError("Parallel transport is not implemented for diffeomorphisms.")

        def adjrep(self, g, X):
            raise NotImplementedError("Adjoint representation is not implemented for diffeomorphisms.")

        def jacobiField(self, p, q, t, X):
            raise NotImplementedError("Jacobi field is not implemented for diffeomorphisms.")

        def action(self, g, x, t = jnp.linspace(0, 1, 2)):
            """Apply the diffeomorphism to a point set x in the ambient space.
            Args:
                g (jax.Array): Diffeomorphism (represented by a momentum).
                x (jax.Array): Points in the ambient space to be transformed.
                t (jax.Array): Time points for the integration (strictly increasing).
                NOTE: First entry in t will be ignored and assumed to be 0!
            Returns:
                jax.Array: Transformed points at times t."""

            # kernel matrix
            h = 1 / (2 * self._M._scale**2)  # inverse kernel bandwidth
            gaussian = lambda a, b: jnp.exp(-jnp.sum((a - b) ** 2) * h)

            def F(y, t, *args):
                """ODE function."""
                v, c, p = y

                p_dot = LazyKernel(p, c, gaussian) @ v

                # Hamiltonian function
                H = lambda v, c: .5 * jnp.sum(v * (LazyKernel(c, c, gaussian) @ v))
                Gv, Gc = jax.grad(H, argnums=(0,1))(v, c)

                return -Gc, Gv, p_dot

            # integrate F
            _, _, x_morphed = odeint(F, (g, self._M._control_pts, x), t)

            return x_morphed

