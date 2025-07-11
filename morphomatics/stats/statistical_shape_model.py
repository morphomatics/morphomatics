################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2025 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

# postponed evaluation of annotations to circumvent cyclic dependencies (will be default behavior in Python 4.0)
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from morphomatics.stats import ExponentialBarycenter as Mean, PrincipalGeodesicAnalysis as PGA

class StatisticalShapeModel(object):
    ''' Statistical manifold model. '''

    def __init__(self, shape_space_cstr):
        """
        Constructor.

        :arg shape_space_cstr: Constructor for particular type of manifold space taking a reference manifold.
        """
        self.shape_space_cstr = shape_space_cstr

    @property
    def variances(self):
        return self.pga.variances

    @property
    def modes(self):
        return self.pga.modes

    @property
    def coeffs(self):
        return self.pga.coeffs

    def construct(self, surfaces):
        '''
        Construct model.
        :arg surfaces: list of surfaces
        '''

        # compute mean / setup manifold space (s.t. mean agrees to reference)
        self.setupShapeSpace(surfaces)
        coords = jnp.stack([self.space.to_coords(S.v) for S in surfaces])
        self.mean_coords = Mean.compute(self.space.M, coords, self.space.ref_coords)

        # compute principal modes
        self.pga = PGA(self.space.M, coords, self.mean_coords)

    def setupShapeSpace(self, surfaces, max_outer = 10, max_inner = 10):
        '''
        Setup manifold space, i.e. determine reference manifold that agrees with mean manifold of \a surfaces.

        :arg surfaces: list of surfaces
        :arg max_outer: max. number of interations in outer loop
        :arg max_inner: max. number of iterations in inner loop
        '''

        # initial guess
        mean = surfaces[0].copy()
        space:ShapeSpace = self.shape_space_cstr(mean)

        # iterate until mean and reference agree
        max = np.finfo(float).max
        tol = mean.v.std(axis=0).mean() / 1000
        for _ in range(max_outer):
            # compute mean in manifold space
            data = jnp.stack([space.to_coords(S.v) for S in surfaces])
            x = Mean.compute(space.M, data, x=space.ref_coords if len(surfaces) > 2 else data[0], max_iter=max_inner)

            # compute vertex coordinates of mean
            v = space.from_coords(x)

            # check convergence
            max_ = np.linalg.norm(v - mean.v, axis=1).max()
            if max_ > max:
                print("{0} > {1} --> divergence".format(max_, max))
                break # divergence
            print(max_)

            # set mean / reference
            mean.v = v
            space.update_ref_geom(v)

            if max_ < tol:
                print("tol {0} reached".format(tol))
                break # convergence
            max = max_

        self.mean = mean
        self.space = space
