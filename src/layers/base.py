__all__ = ["dict_basic_func"]

from typing import Callable

import tensorflow as tf

"""
Basic operation of tensorflow layers
Used in pruning.py and quantization.py
"""


def conv2d_passby(
    layer: tf.keras.layers.Conv2D,
    inputs: tf.Tensor,
    kernel: tf.Tensor,
) -> Callable:
    """
    Summary: return basic operation of the layer (conv)

    :param layer: layer to get the base operation
    :type layer: tf.keras.layers.Conv2D
    :param inputs: inputs of the layer
    :type inputs: tf.Tensor
    :param kernel: kernel of the layer
    :type kernel: tf.Tensor
    :return: function linked with the operation
    :rtype: Callable
    """
    data_format = "NHWC" if layer.data_format == "channels_last" else "NCHW"
    padding = "VALID" if layer.padding == "valid" else "SAME"
    return tf.nn.conv2d(
        inputs,
        kernel,
        layer.strides,
        padding,
        data_format=data_format,
        dilations=layer.dilation_rate,
    )


def conv1d_passby(
    layer: tf.keras.layers.Conv1D,
    inputs: tf.Tensor,
    kernel: tf.Tensor,
) -> Callable:
    """
    Summary: return basic operation of the layer (conv)

    :param layer: layer to get the base operation
    :type layer: tf.keras.layers.Conv1D
    :param inputs: inputs of the layer
    :type inputs: tf.Tensor
    :param kernel: kernel of the layer
    :type kernel: tf.Tensor
    :return: function linked with the operation
    :rtype: Callable
    """
    data_format = "NHWC" if layer.data_format == "channels_last" else "NCHW"
    padding = "VALID" if layer.padding == "valid" else "SAME"
    return tf.nn.conv1d(
        inputs,
        kernel,
        layer.strides,
        padding,
        data_format=data_format,
        dilations=layer.dilation_rate,
    )


def dense_passby(
    layer: tf.keras.layers.Dense,
    inputs: tf.Tensor,
    kernel: tf.Tensor,
) -> Callable:
    """
    Summary: return basic operation of the layer (matmul)

    :param layer: layer to get the base operation
    :type layer: tf.keras.layers.Dense
    :param inputs: inputs of the layer
    :type inputs: tf.Tensor
    :param kernel: kernel of the layer
    :type kernel: tf.Tensor
    :return: function linked with the operation
    :rtype: Callable
    """
    rank = inputs.shape.rank
    return tf.tensordot(inputs, kernel, [[rank - 1], [0]])


dict_basic_func = {
    tf.keras.layers.Dense: dense_passby,
    tf.keras.layers.Conv2D: conv2d_passby,
    tf.keras.layers.Conv1D: conv1d_passby,
}
