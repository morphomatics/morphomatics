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

from typing import Tuple, Callable

import jax
import jax.numpy as jnp

def two_sample_test(data_A: jnp.ndarray,
                    data_B: jnp.ndarray,
                    dissimilarity_measure: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                    n_permutations: int,
                    key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two-sample permutation test for data in a Riemannian manifold or a Lie Group with affine connection. It is
    intended to be used together with the bi-invariant and Riemannian dissimilarity measure defined in the 'stats'
    package.

    The test is based on the permutation test framework proposed in:
        Hanik, Hege, and von Tycowicz (2020): Bi-invariant Two-Sample Tests in Lie Groups for Shape Analysis

    For the dissimilarity measures in the 'stats' package, the null hypotheses on the underlying distributions are

        'Means of distributions underlying the 2 data sets are equal' if Hotelling T2 statistic is used
    or
        'Means and covariance underlying the 2 data sets are equal' if Bhattacharyya distance is used.

    :param data_A: data array of first set; data is sorted along first axis
    :param data_B: data array of second set; data is sorted along first axis
    :param dissimilarity_measure: dissimilarity measure to be used for the test
    :param n_permutations: number of permutations performed for the test
    :param key: random key
    :return: p-value, original distance d_orig between data, vector d-perm of distances between permuted data sets
    """
    n = jnp.shape(data_A)[0]

    # distance between distributions of data
    d_orig = dissimilarity_measure(data_A, data_B)

    D = jnp.concatenate((data_A, data_B), axis=0)

    def permute_and_recompute(key_):
        # mix data
        D_perm = jax.random.permutation(key_, D)
        # distance between shuffled groups
        return dissimilarity_measure(D_perm[:n], D_perm[n:])

    # vectorize
    random_keys = jax.random.split(key, n_permutations)
    d_perm = jax.vmap(permute_and_recompute)(random_keys)

    # p-value, i.e., approximate probability of observing d_orig under the null hypothesis
    p_value = jnp.count_nonzero(d_perm > d_orig) / (n_permutations + 1)

    return p_value, d_orig, d_perm
