__all__ = ["l1_heuristic"]

from typing import Any, Dict

import tensorflow as tf

from .utils import getAllLayer


def l1_heuristic(
    model: tf.keras.Model,
    **kwargs: Dict[str, Any],
) -> Dict[tf.keras.Layer, Dict[str, tf.Tensor]]:
    """
    Summary: get L1 heuristic score

    :param model: model to prune
    :type model: tf.keras.Model
    :param kwargs: usefull arguments
    - prune_last(bool): decide if the score of the last layer is computed or not
    - reset (bool): reset mask for the computation of the score
    :type kwargs: Dict[str, Any]
    :return: score for the parameters of the model
    :rtype: Dict[tf.keras.Layer, Dict[str, tf.Tensor]]
    """
    prune_last = kwargs["prune_last"]
    reset = kwargs["reset"]

    dico_score = {}

    listlayer = getAllLayer(model)
    for layer in listlayer:
        if hasattr(layer, "kernel_mask") and (
            prune_last or (prune_last is False and layer.last is False)
        ):
            dico_score[layer] = {}
            if reset:
                dico_score[layer]["kernel"] = tf.math.abs(layer.kernel)
                if layer.bias_mask is not None:
                    dico_score[layer]["bias"] = tf.math.abs(layer.bias)
            else:
                dico_score[layer]["kernel"] = tf.math.abs(
                    layer.kernel * layer.kernel_mask,
                )
                if layer.bias_mask is not None:
                    dico_score[layer]["bias"] = tf.math.abs(
                        layer.bias * layer.bias_mask,
                    )

    return dico_score
