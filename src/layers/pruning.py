__all__ = ["pruning"]

import inspect
from typing import Tuple

import tensorflow as tf

from .base import dict_basic_func
from .utils import get_args, get_signature, register_keras_custom_object

"""
- The pruning layer can be chained : pruning(quantization(Dense(10))) will
created a dense layer which pruned its kernel then quantized its kernel before
realized a dot product.
- The pruning layer support "Dense and Conv2D and Conv1D" layers.
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


def pruning(
    base_layer: tf.keras.Layer,
    prune_bias: bool = False,
    last: bool = False,
) -> tf.keras.Layer:
    """
    Summary: Encapsulation of a layer to add pruning

    :param base_layer: layer to prune
    :type base_layer: tf.keras.Layer
    :param prune_bias: prune bias if true, defaults to False
    :type prune_bias: bool, optional
    :param last: indicate if last layer, defaults to False
    :type last: bool, optional
    :return: prunable layer
    :rtype: tf.keras.Layer
    """

    @register_keras_custom_object
    class Pruning(type(base_layer)):
        """
        Summary: Generic class to create a prunable layer

        :param type: Dynamic type of layer
        :type type: type(base_layer)
        """

        def __init__(
            self,
            base_layer: tf.keras.Layer,
            prune_bias: bool = False,
            last: bool = False,
        ) -> tf.keras.Layer:
            """
            Summary: Encapsulation of a layer to add pruning

            :param base_layer: layer to make prunable
            :type base_layer: tf.keras.Layer
            :param prune_bias: prune bias if true, defaults to False
            :type prune_bias: bool, optional
            :param last: indicate if last layer, defaults to False
            :type last: bool, optional
            :return: prunable layer
            :rtype: tf.keras.Layer
            """
            # Add assert to limit the quantizer and base_layer choice

            signature = get_signature(base_layer)
            layer_type, layer_param = get_args(signature, base_layer)
            layer_type.__init__(self, **layer_param)

            self.set_signature(base_layer)
            self.set_basetype(base_layer)
            self.prune_bias = prune_bias
            self.last = last

        def build(self, input_shape: Tuple[int]) -> None:
            """
            Summary: Construct pruning mask

            :param input_shape: input shape
            :type input_shape: Tuple[int]
            """
            self.base_type.build(self, input_shape)
            self.kernel_mask = self.add_weight(
                shape=self.kernel.shape,
                initializer=tf.keras.initializers.Ones,
                name="kernel_mask",
                trainable=False,
            )
            if self.use_bias and self.prune_bias:
                self.bias_mask = self.add_weight(
                    shape=self.bias.shape,
                    initializer=tf.keras.initializers.Ones,
                    name="bias_mask",
                    trainable=False,
                )
            else:
                self.bias_mask = None
            self.built = True

        def set_signature(self, base_layer: tf.keras.Layer) -> None:
            """
            Summary: set signature of the new layer

            :param base_layer: layer to make prunable
            :type base_layer: tf.keras.Layer
            """
            # Get signature of base_layer
            signature = get_signature(base_layer)
            self.signature = {"base_layer": (type(self), signature)}
            # Append arguments of current layer to signature
            self.signature["prune_bias"] = inspect.Parameter(
                "prune_bias",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            self.signature["last"] = inspect.Parameter(
                "last",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )

        def set_basetype(self, base_layer: tf.keras.Layer) -> None:
            """
            Summary: set base type of the new layer

            :param base_layer: layer to make prunable
            :type base_layer: tf.keras.Layer
            """
            if hasattr(base_layer, "base_type"):
                self.base_type = base_layer.base_type
            else:
                self.base_type = type(base_layer)

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Summary: Call function of the new layer

            :param inputs: inputs
            :type inputs: tf.Tensor
            :return: outputs
            :rtype: tf.Tensor
            """
            kernel = self.kernel
            bias = self.bias
            signature = self.signature

            # Realize the operation of all layer (Quantization,pruning...)
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
            Summary: apply pruning mask to the weight and bias

            :param inputs: inputs of the layer
            :type inputs: tf.Tensor
            :param kernel: kernel of the layer
            :type kernel: tf.Tensor
            :param bias: bias of the layer
            :type bias: tf.Tensor
            :return: pruned inputs,bias and kernel of the layer
            :rtype: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            """
            kernel = tf.math.multiply(kernel, self.kernel_mask)
            if self.bias_mask is not None:
                bias = tf.math.multiply(bias, self.bias_mask)
            return inputs, kernel, bias

    return Pruning(base_layer, prune_bias, last)
