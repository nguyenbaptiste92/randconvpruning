__all__ = ["snip_heuristic"]

from typing import Any, Callable, Dict, List

import tensorflow as tf

from .utils import getLinkLayerTrainableVariable

###############################################################################################################################################
# STEP TO COMPUTE THE GRADIENT FOR SNIP HEURISTIC (MONOGPU AND MULTIGPU)
###############################################################################################################################################


def compute_gradients_step(
    model: tf.keras.Model,
    elem: Dict[str, tf.Tensor],
    prediction_loss: Callable,
    constitency_loss: Callable,
    mode: str = "simple",
    ratio: int = 10,
    rand_module: tf.keras.layer = None,
    rand_iter: int = 0,
    multi_gpu: bool = False,
) -> List[tf.Tensor]:
    """
    Summary: compute snip step

    :param model: model to prune
    :type model: tf.keras.Model
    :param elem: element of the dataset
    :type elem: Dict[str, tf.Tensor]
    :param prediction_loss: classifiction loss
    :type prediction_loss: Callable
    :param constitency_loss:  consistency loss
    :type constitency_loss: Callable
    :param mode: mode to compute gradients, defaults to "simple"
    :type mode: str, optional
    :param ratio: ratio between prediction loss and consistency loss
    , defaults to 10
    :type ratio: int, optional
    :param rand_module: module to do rand conv, defaults to None
    :type rand_module: tf.keras.layer, optional
    :param rand_iter: number of random convolutions
    use for data_augmentation, defaults to 0
    :type rand_iter: int, optional
    :param multi_gpu: use multi gpu, defaults to False
    :type multi_gpu: bool, optional
    :return: snip score for the batch
    :rtype: List[tf.Tensor]
    """
    inputs = elem["image"]
    target = elem["label"]
    with tf.GradientTape() as tape:
        pred_loss = 0.0
        inv_loss = 0.0
        if mode == "simple":
            prediction = model(inputs, training=False)["label"]
            if multi_gpu:
                pred_loss += tf.reduce_sum(prediction_loss(target, prediction))
            else:
                pred_loss += prediction_loss(target, prediction)

        else:
            if not ((rand_module is not None) and (rand_iter > 0)):
                msg = "rand module should not be None"
                raise ValueError(msg)
            list_prediction = []
            for _ in range(rand_iter):
                input_batch = rand_module(inputs)
                prediction = model(input_batch, training=False)["label"]
                list_prediction.append(prediction)

            if mode in ("rand_conv", "rand_conv_total"):
                for prediction in list_prediction:
                    if multi_gpu:
                        pred_loss += tf.reduce_sum(
                            prediction_loss(target, prediction),
                        )
                    else:
                        pred_loss += prediction_loss(target, prediction)

            if mode in ("rand_conv_consistency", "rand_conv_total"):
                for prediction in list_prediction:
                    if multi_gpu:
                        inv_loss += tf.reduce_sum(
                            constitency_loss(target, prediction),
                        )
                    else:
                        inv_loss += constitency_loss(target, prediction)
        total_loss = pred_loss + ratio * inv_loss

    return tape.gradient(
        total_loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.NONE,
    )


@tf.function
def distributed_compute_gradients_step(
    strategy: tf.distribute.Strategy,
    model: tf.keras.Model,
    elem: Dict[str, tf.Tensor],
    prediction_loss: Callable,
    constitency_loss: Callable,
    mode: str = "simple",
    ratio: int = 10,
    rand_module: tf.keras.layer = None,
    rand_iter: int = 0,
) -> List[tf.Tensor]:
    """
    Summary: compute snip step with distributed strategy

    :param strategy: distributed strategy
    :type strategy: tf.distribute.Strategy
    :param model: model to prune
    :type model: tf.keras.Model
    :param elem: element of the dataset
    :type elem: Dict[str, tf.Tensor]
    :param prediction_loss: classifiction loss
    :type prediction_loss: Callable
    :param constitency_loss:  consistency loss
    :type constitency_loss: Callable
    :param mode: mode to compute gradients, defaults to "simple"
    :type mode: str, optional
    :param ratio: ratio between prediction loss and consistency loss
    , defaults to 10
    :type ratio: int, optional
    :param rand_module: module to do rand conv, defaults to None
    :type rand_module: tf.keras.layer, optional
    :param rand_iter: number of random convolutions
    use for data_augmentation, defaults to 0
    :type rand_iter: int, optional
    :return: snip score for the batch
    :rtype: List[tf.Tensor]
    """
    per_replica_grad = strategy.run(
        compute_gradients_step,
        args=(
            model,
            elem,
            prediction_loss,
            constitency_loss,
            mode,
            ratio,
            rand_module,
            rand_iter,
        ),
    )

    return [
        strategy.reduce("SUM", element, axis=None)
        for element in per_replica_grad
    ]


##################################################################################
# SNIP HEURISTIC
##################################################################################
def snip_heuristic(
    model: tf.keras.Model,
    **kwargs: Dict[str, Any],
) -> Dict[tf.keras.Layer, Dict[str, tf.Tensor]]:
    """
    Summary: get SNIP heuristic score

    :param model: model to prune
    :type model: tf.keras.Model
    :param kwargs: usefull arguments
    -prune_last (bool): decide if the score of the last layer is computed or not
    -dataset (tf.data.Dataset with input and output): dataset on
    which the network will be training
    -mini_batches (integer): number of batches using for the calcul
    of saliency (0 will use the whole dataset)
    -multi_gpu (bool): use multiple gpus to speed up the computations
    -strategy (tf.distribute.MirroredStrategy) : strategy use for multi-gpu
    -mode (string): mode to compute gradients:
        - "simple": compute gradients without random convolutions)
        - "rand_conv": compute gradients with random convolutions with the
        prediction loss
        - "rand_conv_consistency": compute gradients with random convolutions
        with the consistency loss
        - "rand_conv_total": compute gradients with random convolutions
        with the prediction loss and the consistency loss
    -rand_conv (RandConvModule) : random_convolutions module used.
    -prediction_loss (tf.keras.losses): prediction_loss (in general,
    tf.keras.losses.CategoricalCrossentropy)
    -consistency_loss (tf.keras.losses): consistency loss (in general,
    tf.keras.losses.KLDivergence)
    -ratio (float): ratio between prediction loss and consistency loss
    -rand_iter (integer) : number of random convolutions
    use for data_augmentation
    :type kwargs: Dict[str, Any]
    :return: score for the parameters of the model
    :rtype: Dict[tf.keras.Layer, Dict[str, tf.Tensor]]
    """
    prune_last = kwargs["prune_last"]
    dataset = kwargs["dataset"]
    mini_batches = kwargs["mini_batches"]
    multi_gpu = kwargs["multi_gpu"]
    strategy = kwargs["strategy"]
    mode = kwargs["mode"]
    rand_conv = kwargs["rand_conv"]
    prediction_loss = kwargs["prediction_loss"]
    constitency_loss = kwargs["constitency_loss"]
    ratio = kwargs["ratio"]
    rand_iter = kwargs["rand_iter"]

    dico_variable = getLinkLayerTrainableVariable(model)
    # Compute SNIP score of all trainable variables
    cumulated_grads = [
        tf.zeros_like(weight, dtype=tf.dtypes.float32, name=weight.name)
        for weight in model.trainable_variables
    ]
    for i, elem in enumerate(dataset):
        if multi_gpu:
            grads = distributed_compute_gradients_step(
                strategy,
                model,
                elem,
                prediction_loss,
                constitency_loss,
                mode,
                ratio,
                rand_conv,
                rand_iter,
            )
        else:
            grads = compute_gradients_step(
                model,
                elem,
                prediction_loss,
                constitency_loss,
                mode,
                ratio,
                rand_conv,
                rand_iter,
            )
        cumulated_grads = [
            g + tf.math.abs(ag)
            for g, ag in zip(cumulated_grads, grads, strict=False)
        ]
        if i >= mini_batches:
            break

    # Filter the trainable variables
    dico_score = {}
    for variable, grad in zip(
        model.trainable_variables,
        cumulated_grads,
        strict=False,
    ):
        layer = dico_variable[variable.name]
        if hasattr(layer, "kernel_mask") and (
            prune_last or (prune_last is False and layer.last is False)
        ):
            if layer not in dico_score:
                dico_score[layer] = {}
            if variable.shape.is_compatible_with(layer.kernel.shape):
                dico_score[layer]["kernel"] = tf.math.abs(grad * variable)
            else:
                dico_score[layer]["bias"] = tf.math.abs(grad * variable)

    return dico_score
