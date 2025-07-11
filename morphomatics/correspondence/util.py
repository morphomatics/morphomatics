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

import numpy as np


def project_p2t(triangles, point, return_bary=False):
    """
    This functions projects a p-dimensional point on each of the given p-dimensional triangle.
    All operations are parallelized, which makes the code quite hard to read. For an easier take,
    follow the code in the function below (not written by me) for projection on a single triangle.

    The first estimates for each triangle in which of the following region the point lies, then
    solves for each region.


           ^t
     \     |
      \reg2|
       \   |
        \  |
         \ |
          \|
           *P2
           |\
           | \
     reg3  |  \ reg1
           |   \
           |reg0\
           |     \
           |      \ P1
    -------*-------*------->s
           |P0      \
     reg4  | reg5    \ reg6

     Most notations come from :
        [1] "David Eberly, 'Distance Between Point and Triangle in 3D',
    Geometric Tools, LLC, (1999)"

    parameters
    -------------------------------
    triangles   : (m,3,p) set of m p-dimensional triangles
    point       : (p,) coordinates of the point
    return_bary : Whether to return barycentric coordinates inside each triangle

    returns
    -------------------------------
    final_dists : (m,) distance from the point to each of the triangle
    projections : (m,p) coordinates of the projected point
    bary_coords : (m,3) barycentric coordinates of the projection within each triangle
    """

    if point.ndim == 2:
        point = point.squeeze()  # (1,p)

    # rewrite triangles in normal form base + axis
    bases = triangles[:, 0]  # (m,p)
    axis1 = triangles[:, 1] - bases  # (m,p)
    axis2 = triangles[:, 2] - bases  # (m,p)

    diff = bases - point[None, :]  # (m,p)

    #  Precompute quantities with notations from [1]

    a = np.einsum('ij,ij->i', axis1, axis1)  # (m,)
    b = np.einsum('ij,ij->i', axis1, axis2)  # (m,)
    c = np.einsum('ij,ij->i', axis2, axis2)  # (m,)
    d = np.einsum('ij,ij->i', axis1, diff)  # (m,)
    e = np.einsum('ij,ij->i', axis2, diff)  # (m,)
    f = np.einsum('ij,ij->i', diff, diff)  # (m,)

    det = a * c - b ** 2  # (m,)
    s = b * e - c * d  # (m,)
    t = b * d - a * e  # (m,)

    # Array of barycentric coordinates (s,t) and distances
    final_s = np.zeros(s.size)  # (m,)
    final_t = np.zeros(t.size)  # (m,)
    final_dists = np.zeros(t.size)  # (m,)

    # Find for which triangles which zone the point belongs to

    # s + t <= det
    test1 = (s + t <= det)  # (m,) with (m1) True values
    inds_0345 = np.where(test1)[0]  # (m1)
    inds_126 = np.where(~test1)[0]  # (m-m1)

    # s < 0 | s + t <= det
    test11 = s[inds_0345] < 0  # (m1,) with (m11) True values
    inds_34 = inds_0345[test11]  # (m11)
    inds_05 = inds_0345[~test11]  # (m1-m11)

    # t < 0 | (s + t <= det) and (s < 0)
    test111 = t[inds_34] < 0  # (m11) with (m111) True values
    inds_4 = inds_34[test111]  # (m111)
    inds_3 = inds_34[~test111]  # (m11 - m111)

    # t < 0 | s + t <= det and (s >= 0)
    test12 = t[inds_05] < 0  # (m-m11) with (m12) True values
    inds_5 = inds_05[test12]  # (m12;)
    inds_0 = inds_05[~test12]  # (m-m11-m12,)

    # s < 0 | s + t > det
    test21 = s[inds_126] < 0  # (m-m1) with (m21) True values
    inds_2 = inds_126[test21]  # (m21,)
    inds_16 = inds_126[~test21]  # (m-m1-m21)

    # t < 0 | (s + t > det) and (s > 0)
    test22 = t[inds_16] < 0  # (m-m1-m21) with (m22) True values
    inds_6 = inds_16[test22]  # (m22,)
    inds_1 = inds_16[~test22]  # (m-m1-m21-m22)

    # DEAL REGION BY REGION (in parallel within each)

    # REGION 4
    if len(inds_4) > 0:
        # print('Case 4',inds_4)
        test4_1 = d[inds_4] < 0
        inds4_1 = inds_4[test4_1]
        inds4_2 = inds_4[~test4_1]

        # FIRST PART - SUBDIVIDE IN 2
        final_t[inds4_1] = 0  # Useless already done

        test4_11 = (-d[inds4_1] >= a[inds4_1])
        inds4_11 = inds4_1[test4_11]
        inds4_12 = inds4_1[~test4_11]

        final_s[inds4_11] = 1.
        final_dists[inds4_11] = a[inds4_11] + 2.0 * d[inds4_11] + f[inds4_11]

        final_s[inds4_12] = -d[inds4_12] / a[inds4_12]
        final_dists[inds4_12] = d[inds4_12] * s[inds4_12] + f[inds4_12]

        # SECOND PART - SUBDIVIDE IN 2
        final_s[inds4_2] = 0  # Useless already done

        test4_21 = (e[inds4_2] >= 0)
        inds4_21 = inds4_2[test4_21]
        inds4_22 = inds4_2[~test4_21]

        final_t[inds4_21] = 0
        final_dists[inds4_21] = f[inds4_21]

        # SECOND PART OF SECOND PART - SUBDIVIDE IN 2
        test4_221 = (-e[inds4_22] >= c[inds4_22])
        inds4_221 = inds4_22[test4_221]
        inds4_222 = inds4_22[~test4_221]

        final_t[inds4_221] = 1
        final_dists[inds4_221] = c[inds4_221] + 2.0 * e[inds4_221] + f[inds4_221]

        final_t[inds4_222] = -e[inds4_222] / c[inds4_222]
        final_dists[inds4_222] = e[inds4_222] * t[inds4_222] + f[inds4_222]

    if len(inds_3) > 0:
        # print('Case 3', inds_3)
        final_s[inds_3] = 0

        test3_1 = e[inds_3] >= 0
        inds3_1 = inds_3[test3_1]
        inds3_2 = inds_3[~test3_1]

        final_t[inds3_1] = 0
        final_dists[inds3_1] = f[inds3_1]

        # SECOND PART - SUBDIVIDE IN 2

        test3_21 = (-e[inds3_2] >= c[inds3_2])
        inds3_21 = inds3_2[test3_21]
        inds3_22 = inds3_2[~test3_21]

        # print(inds3_21, inds3_22)

        final_t[inds3_21] = 1
        final_dists[inds3_21] = c[inds3_21] + 2.0 * e[inds3_21] + f[inds3_21]

        final_t[inds3_22] = -e[inds3_22] / c[inds3_22]
        final_dists[inds3_22] = e[inds3_22] * final_t[inds3_22] + f[inds3_22]  # -e*t ????

    if len(inds_5) > 0:
        # print('Case 5', inds_5)
        final_t[inds_5] = 0

        test5_1 = d[inds_5] >= 0
        inds5_1 = inds_5[test5_1]
        inds5_2 = inds_5[~test5_1]

        final_s[inds5_1] = 0
        final_dists[inds5_1] = f[inds5_1]

        test5_21 = (-d[inds5_2] >= a[inds5_2])
        inds5_21 = inds5_2[test5_21]
        inds5_22 = inds5_2[~test5_21]

        final_s[inds5_21] = 1
        final_dists[inds5_21] = a[inds5_21] + 2.0 * d[inds5_21] + f[inds5_21]

        final_s[inds5_22] = -d[inds5_22] / a[inds5_22]
        final_dists[inds5_22] = d[inds5_22] * final_s[inds5_22] + f[inds5_22]

    if len(inds_0) > 0:
        # print('Case 0', inds_0)
        invDet = 1.0 / det[inds_0]
        final_s[inds_0] = s[inds_0] * invDet
        final_t[inds_0] = t[inds_0] * invDet
        final_dists[inds_0] = final_s[inds_0] * (
                a[inds_0] * final_s[inds_0] + b[inds_0] * final_t[inds_0] + 2.0 * d[inds_0]) + \
                              final_t[inds_0] * (
                                      b[inds_0] * final_s[inds_0] + c[inds_0] * final_t[inds_0] + 2.0 * e[inds_0]) + \
                              f[inds_0]

    if len(inds_2) > 0:
        # print('Case 2', inds_2)

        tmp0 = b[inds_2] + d[inds_2]
        tmp1 = c[inds_2] + e[inds_2]

        test2_1 = tmp1 > tmp0
        inds2_1 = inds_2[test2_1]
        inds2_2 = inds_2[~test2_1]

        numer = tmp1[test2_1] - tmp0[test2_1]
        denom = a[inds2_1] - 2.0 * b[inds2_1] + c[inds2_1]

        test2_11 = (numer >= denom)
        inds2_11 = inds2_1[test2_11]
        inds2_12 = inds2_1[~test2_11]

        final_s[inds2_11] = 1
        final_t[inds2_11] = 0
        final_dists[inds2_11] = a[inds2_11] + 2.0 * d[inds2_11] + f[inds2_11]

        final_s[inds2_12] = numer[~test2_11] / denom[~test2_11]
        final_t[inds2_12] = 1 - final_s[inds2_12]
        final_dists[inds2_12] = final_s[inds2_12] * (
                a[inds2_12] * final_s[inds2_12] + b[inds2_12] * final_t[inds2_12] + 2 * d[inds2_12]) + \
                                final_t[inds2_12] * (
                                        b[inds2_12] * final_s[inds2_12] + c[inds2_12] * final_t[inds2_12] + 2 * e[
                                    inds2_12]) + f[inds2_12]

        final_s[inds2_2] = 0.

        test2_21 = (tmp1[~test2_1] <= 0.)
        inds2_21 = inds2_2[test2_21]
        inds2_22 = inds2_2[~test2_21]

        final_t[inds2_21] = 1
        final_dists[inds2_21] = c[inds2_21] + 2.0 * e[inds2_21] + f[inds2_21]

        test2_221 = (e[inds2_22] >= 0.)
        inds2_221 = inds2_22[test2_221]
        inds2_222 = inds2_22[~test2_221]

        final_t[inds2_221] = 0.
        final_dists[inds2_221] = f[inds2_221]

        final_t[inds2_222] = -e[inds2_222] / c[inds2_222]
        final_dists[inds2_222] = e[inds2_222] * final_t[inds2_222] + f[inds2_222]

    if len(inds_6) > 0:
        # print('Case 6', inds_6)
        tmp0 = b[inds_6] + e[inds_6]
        tmp1 = a[inds_6] + d[inds_6]

        test6_1 = tmp1 > tmp0
        inds6_1 = inds_6[test6_1]
        inds6_2 = inds_6[~test6_1]

        numer = tmp1[test6_1] - tmp0[test6_1]
        denom = a[inds6_1] - 2.0 * b[inds6_1] + c[inds6_1]

        test6_11 = (numer >= denom)
        inds6_11 = inds6_1[test6_11]
        inds6_12 = inds6_1[~test6_11]

        final_t[inds6_11] = 1
        final_s[inds6_11] = 0
        final_dists[inds6_11] = c[inds6_11] + 2.0 * e[inds6_11] + f[inds6_11]

        final_t[inds6_12] = numer[~test6_11] / denom[~test6_11]
        final_s[inds6_12] = 1 - final_t[inds6_12]
        final_dists[inds6_12] = final_s[inds6_12] * (a[inds6_12] * final_s[inds6_12] +
                                                     b[inds6_12] * final_t[inds6_12] + 2.0 * d[inds6_12]) + \
                                final_t[inds6_12] * (b[inds6_12] * final_s[inds6_12] +
                                                     c[inds6_12] * final_t[inds6_12] + 2.0 * e[inds6_12]) + f[inds6_12]

        final_t[inds6_2] = 0.

        test6_21 = (tmp1[~test6_1] <= 0.)
        inds6_21 = inds6_2[test6_21]
        inds6_22 = inds6_2[~test6_21]

        final_s[inds6_21] = 1
        final_dists[inds6_21] = a[inds6_21] + 2.0 * d[inds6_21] + f[inds6_21]

        test6_221 = (d[inds6_22] >= 0.)
        inds6_221 = inds6_22[test6_221]
        inds6_222 = inds6_22[~test6_221]

        final_s[inds6_221] = 0.
        final_dists[inds6_221] = f[inds6_221]

        final_s[inds6_222] = -d[inds6_222] / a[inds6_222]
        final_dists[inds6_222] = d[inds6_222] * final_s[inds6_222] + f[inds6_222]

    if len(inds_1) > 0:
        # print('Case 1', inds_1)
        numer = c[inds_1] + e[inds_1] - b[inds_1] - d[inds_1]

        test1_1 = numer <= 0
        inds1_1 = inds_1[test1_1]
        inds1_2 = inds_1[~test1_1]

        final_s[inds1_1] = 0
        final_t[inds1_1] = 1
        final_dists[inds1_1] = c[inds1_1] + 2.0 * e[inds1_1] + f[inds1_1]

        denom = a[inds1_2] - 2.0 * b[inds1_2] + c[inds1_2]

        test1_21 = (numer[~test1_1] >= denom)
        # print(denom, numer, numer[~test1_1], test1_21, inds1_2)
        inds1_21 = inds1_2[test1_21]
        inds1_22 = inds1_2[~test1_21]

        final_s[inds1_21] = 1
        final_t[inds1_21] = 0
        final_dists[inds1_21] = a[inds1_21] + 2.0 * d[inds1_21] + f[inds1_21]

        final_s[inds1_22] = numer[~test1_1][~test1_21] / denom[~test1_21]
        final_t[inds1_22] = 1 - final_s[inds1_22]
        final_dists[inds1_22] = final_s[inds1_22] * (
                a[inds1_22] * final_s[inds1_22] + b[inds1_22] * final_t[inds1_22] + 2.0 * d[inds1_22]) + \
                                final_t[inds1_22] * (
                                        b[inds1_22] * final_s[inds1_22] + c[inds1_22] * final_t[inds1_22] + 2.0 * e[
                                    inds1_22]) + f[inds1_22]

    final_dists[final_dists < 0] = 0
    final_dists = np.sqrt(final_dists)

    projections = bases + final_s[:, None] * axis1 + final_t[:, None] * axis2
    if return_bary:
        bary_coords = np.concatenate([1 - final_s[:, None] - final_t[:, None], final_s[:, None], final_t[:, None]],
                                     axis=1)
        return final_dists, projections, bary_coords

    return final_dists, projections
