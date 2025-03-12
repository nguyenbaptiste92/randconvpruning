__all__ = ["quantization"]

import inspect
from typing import Callable, Tuple

import tensorflow as tf  # type: ignore  # noqa: PGH003

from .base import dict_basic_func
from .utils import get_args, get_signature, register_keras_custom_object

"""
- The quantization layer can be chained : quantization(pruning(Dense(10))) will
created a dense layer which pruned its kernel then quantized its kernel before
realized a dot product.
- The quantization layer support "Dense, Conv2D and Conv1D" layers.
- To add more compatible layers, you should:
    - Take a base layer with an attribute "kernel, use_bias, bias
    and activation" like the Dense layer
    (you can manually create these attributes).
    - Add to the dictionary .base.dict_basic_func your base layer and a
    corresponding function which have layer,inputs and kernel as parameters
    and return one output:
        dense_passby(layer,inputs,kernel):
            rank = inputs.shape.rank
            return tf.tensordot(inputs, kernel, [[rank - 1], [0]])

        dict_basic_func[tf.keras.layers.Dense]=dense_passby

This layer supports only the save of weights and not the save of model.
"""


def quantization(
    base_layer: tf.keras.Layer,
    quantizer: Callable,
) -> tf.keras.Layer:
    """
    Summary: Encapsulates a layer to add quantization.

    :param base_layer: The base layer to be quantized.
    :type base_layer: tf.keras.Layer
    :param quantizer: The quantizer to apply, defaults to None.
    :type quantizer: Callable, optional
    :return: A quantized layer.
    :rtype: tf.keras.Layer
    """

    @register_keras_custom_object
    class Quantization(type(base_layer)):
        """
        Generic class to create a quantizable layer.

        :param type: Dynamic type of layer.
        :type type: type(base_layer)
        """

        def __init__(
            self,
            base_layer: tf.keras.Layer,
            quantizer: Callable,
        ) -> tf.keras.Layer:
            """
            Summary: Initializes the quantizable layer.

            :param base_layer: The base layer to be quantized.
            :type base_layer: tf.keras.Layer
            :param quantizer: The quantizer to apply, defaults to None.
            :type quantizer:Callable, optional
            """
            # Add assert to limit the quantizer and base_layer choice
            signature = get_signature(base_layer)
            layer_type, layer_param = get_args(signature, base_layer)
            layer_type.__init__(self, **layer_param)

            self.set_signature(base_layer)
            self.set_basetype(base_layer)
            self.quantizer = quantizer

        def set_signature(self, base_layer: tf.keras.Layer) -> None:
            """
            Summary: Sets the signature of the new layer.

            :param base_layer: The base layer to be quantized.
            :type base_layer: tf.keras.Layer
            """
            # Get signature of base_layer
            signature = get_signature(base_layer)
            self.signature = {"base_layer": (type(self), signature)}
            # Append arguments of current layer to signature
            self.signature["quantizer"] = inspect.Parameter(
                "quantizer",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )

        def set_basetype(self, base_layer: tf.keras.Layer) -> None:
            """
            Summary: Sets the base type of the new layer.

            :param base_layer: The base layer to be quantized.
            :type base_layer: tf.keras.Layer
            """
            if hasattr(base_layer, "base_type"):
                self.base_type = base_layer.base_type
            else:
                self.base_type = type(base_layer)

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Summary: Call function of the new layer.

            :param inputs: Inputs to the layer.
            :type inputs: tf.Tensor
            :return: Outputs of the layer.
            :rtype: tf.Tensor
            """
            kernel = self.kernel
            bias = self.bias
            signature = self.signature

            # Realize the operation of all layer (Quantization, pruning...)
            while signature["base_layer"][0] is not self.base_type:
                inputs, kernel, bias = signature["base_layer"][0].passby(
                    self,
                    inputs,
                    kernel,
                    bias,
                )
                signature = signature["base_layer"][1]

            # Realize the operation of the Layer
            output = dict_basic_func[self.base_type](self, inputs, kernel)
            if self.use_bias:
                output = tf.nn.bias_add(output, bias)
            output = self.activation(output)
            return output

        def passby(
            self,
            inputs: tf.Tensor,
            kernel: tf.Tensor,
            bias: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """
            Summary: Applies quantization to the weight and bias.

            :param inputs: Inputs of the layer.
            :type inputs: tf.Tensor
            :param kernel: Kernel of the layer.
            :type kernel: tf.Tensor
            :param bias: Bias of the layer.
            :type bias: tf.Tensor
            :return: Quantized inputs, bias, and kernel of the layer.
            :rtype: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            """
            if self.quantizer is not None:
                kernel = self.quantizer.call(kernel)
                if self.use_bias:
                    bias = self.quantizer.call(bias)
            return inputs, kernel, bias

    return Quantization(base_layer, quantizer=quantizer)
