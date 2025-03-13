__all__ = ["structuredpruning"]

from typing import Callable, List

import numpy as np
import tensorflow as tf


def structuredpruning(
    model: tf.keras.Model,
    heuristic_function: Callable,
    sparsity: float = 20.0,
    previous_sparsity: float = 0.0,
    layer_wise: bool = True,
    aggregate: str = "mean",
    iterative: int = 1,
    **kwargs: List,
) -> tf.keras.Model:
    """
    Summary: Unstructured pruning function

    :param model: model
    :type model: tf.keras.Model
    :param heuristic_function: heuristic (L1 norm, SNIP, Synflow)
    :type heuristic_function: Callable
    :param sparsity: percentage of weights whose values will be 0
    after the function, defaults to 20.0
    :type sparsity: float, optional
    :param previous_sparsity: percentage of weights whose values are 0
    before the function, defaults to 0.0
    :type previous_sparsity: float, optional
    :param layer_wise: local or global pruning, defaults to True
    :type layer_wise: bool, optional
    :param aggregate: aggregation of score across a filter or
    neuron ("mean","sum"), defaults to "mean"
    :type aggregate: str, optional
    :param iterative: number of iteration used in the
    cycle ranking/pruning), defaults to 1
    :type iterative: int, optional
    :return: pruned model
    :rtype: tf.keras.Model
    """
    if sparsity == previous_sparsity:
        return model

    if not iterative > 0:
        msg = "iterative value should be superior to 0"
        raise ValueError(msg)

    for i in range(iterative):
        power = (iterative - (i + 1)) / iterative
        new_sparsity = sparsity + 1 - (sparsity - previous_sparsity) ** power
        dico_score = heuristic_function(model, **kwargs)

        for layer, score in dico_score.items():
            score.pop("bias", None)
            for key in score:
                if aggregate == "mean":
                    dico_score[layer][key] = tf.math.reduce_sum(
                        dico_score[layer][key],
                        axis=list(range(tf.rank(dico_score[layer][key]) - 1)),
                    ) / tf.cast(
                        tf.math.reduce_prod(dico_score[layer][key].shape[:-1]),
                        dtype=tf.float32,
                    )
                elif aggregate == "sum":
                    dico_score[layer][key] = tf.math.reduce_sum(
                        dico_score[layer][key],
                        axis=list(range(tf.rank(dico_score[layer][key]) - 1)),
                    )

        if layer_wise:
            for layer, score in dico_score.items():
                threshold = np.percentile(score["kernel"].numpy(), new_sparsity)
                score_mask = tf.cast(
                    tf.where(score["kernel"] <= threshold, 0, 1),
                    dtype=tf.float32,
                )
                layer.kernel_mask.assign(
                    tf.ones_like(layer.kernel_mask) * score_mask,
                )
                if layer.bias_mask is not None:
                    layer.bias_mask.assign(
                        tf.ones_like(layer.bias_mask) * score_mask,
                    )
        else:
            score_vector = tf.concat(
                [
                    score["kernel"]
                    for score in dico_score.values()
                    if "kernel" in score
                ],
                -1,
            )
            threshold = np.percentile(score_vector.numpy(), new_sparsity)
            for layer, score in dico_score.items():
                score_mask = tf.cast(
                    tf.where(score["kernel"] <= threshold, 0, 1),
                    dtype=tf.float32,
                )
                layer.kernel_mask.assign(
                    tf.ones_like(layer.kernel_mask) * score_mask,
                )
                if layer.bias_mask is not None:
                    layer.bias_mask.assign(
                        tf.ones_like(layer.bias_mask) * score_mask,
                    )

    return model
