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

from typing import NamedTuple, Dict

import jax.numpy as jnp
import jax.ops

from functools import partial

import jraph
import flax.linen as nn
import optax


class TrainingState(NamedTuple):
    params: Dict
    avg_params: Dict
    opt_state: optax.OptState


def weighted_cross_entropy_loss(params: Dict,
                                graph: jraph.GraphsTuple,
                                label: jnp.ndarray,
                                network: nn.Module,
                                mask: jnp.array,
                                weights: jnp.array = None
                                ) -> jnp.ndarray:
    """Weighted cross-entropy classification loss

    :param params: network parameters
    :param graph: graph, possibly batched, on which the loss is to be evaluated
    :param label: ground truth labels
    :param network: graph neural network
    :param mask: binary mask to mask dummy graphs from batching (use jraph's get_graph_padding_mask if applicable)
    :param weights: class weights (all one by default)
    :return: scalar loss

    Can be used for both transductive and inductive learning. In the latter case, batch may only consist of one graph,
    and the labels must be zero vectors for test nodes.

    """

    logits = network.apply(params, graph)
    NUM_CLASSES = logits.shape[-1]
    one_hot = jax.nn.one_hot(label, NUM_CLASSES) * mask[:, None]

    if weights is None:
        weights = jnp.ones(NUM_CLASSES) / NUM_CLASSES

    # l2_regularizer = 0.5 * sum(
    #     jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    terms = weights[None] * one_hot * jax.nn.log_softmax(logits)
    # log_likelihood = jax.ops.segment_sum(-terms.sum(axis=1), mask.astype(jnp.int32), num_segments=2)[1]
    log_likelihood = jax.lax.select(mask, -terms.sum(axis=1), jnp.zeros_like(mask, dtype=float)).sum()
    return log_likelihood / jnp.sum(mask)  # + 1e-4 * l2_regularizer


@partial(jax.jit, static_argnames=['num_classes'])
def confusion_matrix(predictions: jnp.array,
                     results: jnp.array,
                     num_classes: int,
                     labels: jnp.array,
                     mask: jnp.array
                     ) -> jnp.array:
    """Encodes true positives (TP), false positives (FP), and false negatives (FN) of a multi-class classification in a
    matrix

    :param predictions: array of predicted classes
    :param results: boolean array indicating correct classification
    :param num_classes: number of classes
    :param labels: ground truth labels (enumerated as results)
    :param mask: binary mask to mask dummy graphs from batching
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
def evaluate(params: Dict,
             graph: jraph.GraphsTuple,
             labels: jnp.ndarray,
             num_classes: int,
             network: nn.Module,
             rn_key: jax.random.PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
    """Evaluation metric: classification accuracy

    :param params: network parameters
    :param graph: graph, possibly batched, on which the loss is to be evaluated
    :param labels: ground truth labels
    :param num_classes: only needed for consistency with F1 score
    :param network: graph neural network
    :param rn_key: random number key
    :param mask: binary mask to mask dummy graphs from batching
    :return: score in [0,1]
    """

    logits = network.apply(params, rn_key, graph)
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.sum((predictions == labels) * mask) / jnp.sum(mask)

    return acc


@partial(jax.jit, static_argnames=['num_classes', 'network'])
def evaluate_F1(params: Dict,
                graph: jraph.GraphsTuple,
                labels: jnp.ndarray,
                num_classes: int,
                network: nn.Module,
                mask: jnp.ndarray
                ) -> jnp.ndarray:
    """Evaluation metric: F1 score for classification

    :param params: network parameters
    :param graph: graph, possibly batched, on which the loss is to be evaluated
    :param labels: ground truth labels
    :param num_classes: only needed for consistency with F1 score
    :param network: graph neural network
    :param mask: binary mask to mask dummy graphs from batching
    :return: score in [0,1]
    """

    logits = network.apply(params, graph)
    predictions = jnp.argmax(logits, axis=-1)

    C = confusion_matrix(predictions, (predictions == labels), num_classes, labels, mask)

    # jax.debug.print('confusion_matrix: {}', C)

    def f1_macro(_C):
        def single_class(c):
            return c[0] / (c[0] + 1 / 2 * (c[1] + c[2]))

        return jnp.mean(jax.vmap(single_class)(_C))

    return f1_macro(C)


@partial(jax.jit, static_argnames=['optimizer', 'network', 'verbosity'])
def update(state: TrainingState,
           graph: jraph.GraphsTuple,
           label: jnp.ndarray,
           optimizer: optax.GradientTransformation,
           network: nn.Module,
           mask: jnp.ndarray,
           weights: jnp.ndarray = None,
           verbosity: int = 0
           ) -> TrainingState:
    """Learning rule (stochastic gradient descent)

    :param state: current training state
    :param graph: graph (possibly batched)
    :param label: ground truth labels
    :param optimizer: optimizer function (e.g., optax adam)
    :param network: graph neural network
    :param mask: binary mask to mask dummy graphs from batching (use jraph's get_graph_padding_mask if applicable)
    :param weights: class weights (all one by default)
    :param verbosity: verbosity level between 0 and 2
    :return: updated training state
    """
    value, grads = jax.value_and_grad(weighted_cross_entropy_loss)(state.params, graph, label, network, mask, weights)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)

    if verbosity > 0:
        jax.debug.print('value: {}', value)
    if verbosity > 1:
        jax.debug.print("||grads_i||_inf: {}", jax.tree_util.tree_map(lambda a: jnp.max(jnp.abs(a)), grads))

    params = optax.apply_updates(state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    avg_params = optax.incremental_update(params, state.avg_params, step_size=0.1)
    return TrainingState(params, avg_params, opt_state)
