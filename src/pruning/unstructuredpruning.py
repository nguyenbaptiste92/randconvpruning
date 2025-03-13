__all__ = ["unstructuredpruning"]

from typing import Callable, List

import numpy as np
import tensorflow as tf


def unstructuredpruning(
    model: tf.keras.Model,
    heuristic_function: Callable,
    sparsity: float = 20.0,
    previous_sparsity: float = 0.0,
    layer_wise: bool = True,
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

        # Compute score with heuristic
        dico_score = heuristic_function(model, **kwargs)

        # Prune
        if layer_wise:
            for layer, dico in dico_score.items():
                list_score = list(dico.values())
                list_score = [tf.reshape(x, [-1]) for x in list_score]
                score_vector = tf.concat(list_score, -1)
                threshold = np.percentile(score_vector.numpy(), new_sparsity)
                new_kernel_mask = tf.cast(
                    tf.where(dico["kernel"] <= threshold, 0, 1),
                    dtype=tf.float32,
                )
                layer.kernel_mask.assign(new_kernel_mask)
                if layer.bias_mask is not None:
                    new_bias_mask = tf.cast(
                        tf.where(dico["bias"] <= threshold, 0, 1),
                        dtype=tf.float32,
                    )
                    layer.bias_mask.assign(new_bias_mask)
        else:
            list_score = [dico["kernel"] for dico in dico_score.values()] + [
                dico.get("bias") for dico in dico_score.values()
            ]
            list_score = [x for x in list_score if x is not None]
            list_score = [tf.reshape(x, [-1]) for x in list_score]
            score_vector = tf.concat(list_score, -1)
            threshold = np.percentile(score_vector.numpy(), new_sparsity)
            for layer, dico in dico_score.items():
                new_kernel_mask = tf.cast(
                    tf.where(dico["kernel"] <= threshold, 0, 1),
                    dtype=tf.float32,
                )
                layer.kernel_mask.assign(new_kernel_mask)
                if layer.bias_mask is not None:
                    new_bias_mask = tf.cast(
                        tf.where(dico["bias"] <= threshold, 0, 1),
                        dtype=tf.float32,
                    )
                    layer.bias_mask.assign(new_bias_mask)

    return model
