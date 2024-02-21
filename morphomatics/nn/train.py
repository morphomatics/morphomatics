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

from typing import NamedTuple

import jax.numpy as jnp
import jax.ops

from functools import partial

import jraph
import haiku as hk
import optax


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


def weighted_cross_entropy_loss(params: hk.Params, graph: jraph.GraphsTuple, label: jnp.ndarray, network, rn_key,
                                mask, weights: jnp.array = None) -> jnp.ndarray:
    """Weighted cross-entropy classification loss, regularised by L2 weight decay.

    Can be used for both transductive and inductive learning. In the latter case, batch may only consist of one graph,
    and the labels must be zero vectors for test nodes.

    """

    if mask is None:
        mask = jnp.ones(len(graph.n_node), dtype=int)

    logits = network.apply(params, rn_key, graph)
    NUM_CLASSES = logits.shape[-1]
    one_hot = jax.nn.one_hot(label, NUM_CLASSES) * mask[:, None]

    if weights is None:
        weights = jnp.ones(NUM_CLASSES) / NUM_CLASSES

    # jax.debug.print('one_hot: {}.', one_hot)
    # jax.debug.print('weights: {}.', weights)
    # jax.debug.print('labels: {}.', label)
    # jax.debug.print('mask: {}.', mask)
    # jax.debug.print('log_softmax: {}.', jax.nn.softmax(logits))
    # jax.debug.print('logits: {}.', logits)

    # l2_regularizer = 0.5 * sum(
    #     jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    terms = weights[None] * one_hot * jax.nn.log_softmax(logits)
    # jax.debug.print('terms: {}.', terms)
    # jax.debug.print('masked_terms: {}.', jax.lax.select(mask, -terms.sum(axis=1), jnp.zeros_like(mask, dtype=float)))
    # log_likelihood = jax.ops.segment_sum(terms.sum(axis=1), mask, num_segments=2)[1]
    log_likelihood = jax.lax.select(mask, -terms.sum(axis=1), jnp.zeros_like(mask, dtype=float)).sum()
    # jax.debug.print('log_likelihood: {}.', log_likelihood)
    return log_likelihood / jnp.sum(mask)  # + 1e-4 * l2_regularizer


@partial(jax.jit, static_argnames=['num_classes'])
def confusion_matrix(predictions: jnp.array, results: jnp.array, num_classes: int, labels: jnp.array,
                     mask: jnp.ndarray = None) -> jnp.array:
    """Encodes true positives (TP), false positives (FP), and false negatives (FN) of a multi-class classification in a matrix

    :param predictions: array of predicted classes
    :param results: boolean array indicating correct classification
    :param num_classes: number of classes
    :param labels: true labels (enumerated as results)
    :return: matrix of size [num_classes, 3]; rows represent classes; the first colum counts TPs, the second FPs, and
    the third FNs for each class
    """

    def f(prediction, result, label, m):
        def tp(_pl):
            _, _l = _pl
            A = jnp.zeros((num_classes, 3), int)
            return A.at[_l, 0].set(1)

        def fp_and_fn(_pl):
            _p, _l = _pl
            A = jnp.zeros((num_classes, 3), int)
            A = A.at[_p, 1].set(1)
            A = A.at[_l, 2].set(1)
            return A

        return jax.lax.cond(result, tp, fp_and_fn, (prediction, label)) * m

    B = jax.vmap(f)(predictions, results, labels.astype(int), mask)

    return jnp.sum(B, axis=0)


@partial(jax.jit, static_argnames=['num_classes', 'network'])
def evaluate(params: hk.Params, graph: jraph.GraphsTuple, labels: jnp.ndarray, num_classes: int, network, rn_key,
             mask: jnp.ndarray = None) -> jnp.ndarray:
    """Evaluation metric (classification accuracy)."""
    if mask is None:
        mask = jnp.ones(len(graph.n_node))

    logits = network.apply(params, rn_key, graph)
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.sum((predictions == labels) * mask) / jnp.sum(mask)

    # jax.debug.print('max(predictions): {}', jnp.max(predictions))
    return acc


@partial(jax.jit, static_argnames=['num_classes', 'network'])
def evaluate_F1(params: hk.Params, graph: jraph.GraphsTuple, labels: jnp.ndarray, num_classes: int, network, rn_key,
                mask: jnp.ndarray = None) -> jnp.ndarray:
    """Evaluation metric (classification accuracy)."""
    if mask is None:
        mask = jnp.ones(len(graph.n_node))

    logits = network.apply(params, rn_key, graph)
    predictions = jnp.argmax(logits, axis=-1)

    C = confusion_matrix(predictions, (predictions == labels), num_classes, labels, mask)

    # jax.debug.print('confusion_matrix: {}', C)

    def f1_macro(_C):
        def single_class(c):
            return c[0] / (c[0] + 1 / 2 * (c[1] + c[2]))

        return jnp.mean(jax.vmap(single_class)(_C))

    return f1_macro(C)


@partial(jax.jit, static_argnames=['optimizer', 'network'])
def update(state: TrainingState, graph: jraph.GraphsTuple, label: jnp.ndarray, optimizer, network, rn_key,
           mask: jnp.ndarray = None, weights: jnp.ndarray = None) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    value, grads = jax.value_and_grad(weighted_cross_entropy_loss)(state.params, graph, label, network, rn_key, mask,
                                                                   weights)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)

    # jax.debug.print('value: {}', value)
    # jax.debug.print("||grads_i||_inf: {}", jax.tree_util.tree_map(lambda a: jnp.max(jnp.abs(a)), grads))
    # infnrm = jnp.round(jax.tree_util.tree_reduce(lambda a, b: jnp.max(jnp.array(a, b)),
    #                                              jax.tree_util.tree_map(lambda a: jnp.max(jnp.abs(a)), grads)),
    #                    3)
    # jax.debug.print("||grads||_inf: {}", infnrm)
    # jax.debug.print('params: {}', state.params)

    params = optax.apply_updates(state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    avg_params = optax.incremental_update(params, state.avg_params, step_size=0.1)
    return TrainingState(params, avg_params, opt_state)
    # return TrainingState(params, params, opt_state)
