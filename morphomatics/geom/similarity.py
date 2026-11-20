################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2026 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from typing import NamedTuple

import jax.numpy as jnp

from .surface import Surface
from .misc import LazyKernel as K

def varifold(S: Surface, T: Surface, sigma: float) -> float:
    """
    Compute the varifold (squared) distance between two surfaces.

    Parameters
    ----------
    S : Surface
        sourse surface
    T : Surface
        target surfcae
    sigma : float
        The bandwidth of the Gaussian kernel used in the distance computation.

    Returns
    -------
    float
        The squared distance between the two surfaces represented as varifolds.
    """

    """Geometric features of a surface."""
    class Features(NamedTuple):
        x: jnp.ndarray
        n: jnp.ndarray
        a: jnp.ndarray
        @classmethod
        def extract(cls, shape: Surface):
            """Utility function to extract geometric features from surface."""

            # edges
            o = shape.v[shape.f[:, 0]]
            e1 = shape.v[shape.f[:, 1]] - o
            e2 = shape.v[shape.f[:, 2]] - o

            # centers, normals, and areas
            centers = o + (e1 + e2) / 3
            cross = jnp.cross(e1, e2)
            areas = jnp.linalg.norm(cross, axis=1)
            normals = cross / areas.reshape(-1, 1)

            return cls(centers, normals, areas)

    S_feat = Features.extract(S)
    T_feat = Features.extract(T)

    # kernels ...
    # ... Gaussian kernel
    k_radial = lambda a, b: jnp.exp(-jnp.sum((a - b) ** 2) / sigma**2)
    # ... Cauchy-Binet kernel
    k_zonal = lambda a, b: (a * b).sum(-1) ** 2
    # ... product kernel
    kernel = lambda a, b: k_radial(a.x, b.x) * k_zonal(a.n, b.n)

    S_sqnorm = jnp.inner(S_feat.a, K(S_feat, S_feat, kernel) @ S_feat.a)
    T_sqnorm = jnp.inner(T_feat.a, K(T_feat, T_feat, kernel) @ T_feat.a)
    ST_inner = jnp.inner(S_feat.a, K(S_feat, T_feat, kernel) @ T_feat.a)

    return S_sqnorm + T_sqnorm - 2 * ST_inner
