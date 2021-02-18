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


def align(src, ref):
    """ (Constrained) Procrustes alignment of src to ref using Kabsch algorithm
    :arg src: n-by-3 array of vertex coordinates of source object
    :arg ref: n-by-3 array of vertex coordinates of reference
    :returns: aligned coords.
    """
    n = len(ref)
    # cross-covariance matrix
    c_s = src.mean(axis=0)
    c_r = ref.mean(axis=0)
    xCov = (ref.T @ src) / n - np.outer(c_r, c_s)
    # optimal rotation
    U, S, Vt = np.linalg.svd(xCov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        R = Vt.T @ np.diag([1, 1, -1]) @ U.T
    # return aligned coords.
    return src @ R + (c_r - c_s @ R)
