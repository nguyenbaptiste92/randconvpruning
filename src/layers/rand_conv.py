__all__ = [
    "MultiScaleRandConv",
    "RandConv1D",
    "RandConv2D",
    "RandConvModule",
    "randconv_printing",
]

from functools import partial
from typing import Any, Dict, List, Tuple

import tensorflow as tf

from .utils import register_keras_custom_object

# Variable to print result for test
randconv_printing = tf.Variable(
    initial_value=False,
    trainable=False,
    dtype=tf.bool,
)

###################################################
# SINGLE_SCALE_RAND_CONV (1D and 2D)
###################################################


@register_keras_custom_object
class RandConv2D(tf.keras.layers.Conv2D):
    """
    Summary: random convolutions

    :param tf: parent layer
    :type tf: tf.keras.layers.Conv2D
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        rand_bias: bool = False,
        distribution: str = "kaiming_normal",
        clamp_output: bool = False,
        range_up: float = 0,
        range_low: float = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Summary: initialisaton

        :param filters: number of filters
        :type filters: int
        :param kernel_size: kernel size
        :type kernel_size: int
        :param rand_bias: use rand bias if true, defaults to False
        :type rand_bias: bool, optional
        :param distribution: initialization distribution, defaults to
        "kaiming_normal"
        :type distribution: str, optional
        :param clamp_output: clamp output if true, defaults to False
        :type clamp_output: bool, optional
        :param range_up: range of clamp, defaults to 0
        :type range_up: float, optional
        :param range_low: range of clamp, defaults to 1
        :type range_low: float, optional
        """
        super(RandConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            use_bias=rand_bias,
            trainable=False,
            padding="same",
            **kwargs,
        )

        # Variable to decide if you randomize the kernel at each forward pass
        self.is_randomizing = tf.Variable(
            initial_value=True,
            trainable=False,
            dtype=tf.bool,
        )

        # Usefull variable for random convolution like in https://github.com/wildphoton/RandConv
        self.rand_bias = rand_bias
        self.distribution = distribution
        self.clamp_output = clamp_output
        self.range_low = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_low,
                trainable=False,
                dtype=tf.float32,
            )
        )
        self.range_up = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_up,
                trainable=False,
                dtype=tf.float32,
            )
        )

    def randomize(self) -> None:
        """Summary: randomize kernel and bias"""
        if self.distribution == "kaiming_uniform":
            initializer = tf.keras.initializers.HeUniform()
        elif self.distribution == "kaiming_normal":
            initializer = tf.keras.initializers.HeNormal()
        elif self.distribution == "glorot_normal":
            initializer = tf.keras.initializers.GlorotNormal()
        else:
            initializer = tf.keras.initializers.HeUniform()
        self.kernel.assign(initializer(self.kernel.shape))
        if self.rand_bias:
            self.bias.assign(initializer(self.bias.shape))

        tf.cond(
            randconv_printing,
            partial(tf.print, "Randomization with ", self.distribution),
            partial(tf.print, end=""),
        )  # Message for test

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Summary: Forward pass

        :param inputs: input
        :type inputs: tf.Tensor
        :return: output
        :rtype: tf.Tensor
        """
        tf.cond(self.is_randomizing, self.randomize, lambda: None)
        output = super(RandConv2D, self).call(inputs)

        if self.clamp_output == "clamp":
            output = tf.clip_by_value(output, self.range_low, self.range_up)

        return output


@register_keras_custom_object
class RandConv1D(tf.keras.layers.Conv1D):
    """
    Summary: random convolutions

    :param tf: parent layer
    :type tf: tf.keras.layers.Conv1D
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        rand_bias: bool = False,
        distribution: str = "kaiming_normal",
        clamp_output: bool = False,
        range_up: float = 0,
        range_low: float = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Summary: initialisaton

        :param filters: number of filters
        :type filters: int
        :param kernel_size: kernel size
        :type kernel_size: int
        :param rand_bias: use rand bias if true, defaults to False
        :type rand_bias: bool, optional
        :param distribution: initialization distribution, defaults to
        "kaiming_normal"
        :type distribution: str, optional
        :param clamp_output: clamp output if true, defaults to False
        :type clamp_output: bool, optional
        :param range_up: range of clamp, defaults to 0
        :type range_up: float, optional
        :param range_low: range of clamp, defaults to 1
        :type range_low: float, optional
        """
        super(RandConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            use_bias=rand_bias,
            trainable=False,
            padding="same",
            **kwargs,
        )

        # Variable to decide if you randomize the kernel at each forward pass
        self.is_randomizing = tf.Variable(
            initial_value=True,
            trainable=False,
            dtype=tf.bool,
        )

        # Usefull variable for random convolution like in https://github.com/wildphoton/RandConv
        self.rand_bias = rand_bias
        self.distribution = distribution
        self.clamp_output = clamp_output
        self.range_low = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_low,
                trainable=False,
                dtype=tf.float32,
            )
        )
        self.range_up = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_up,
                trainable=False,
                dtype=tf.float32,
            )
        )

    def randomize(self) -> None:
        """Summary: randomize kernel and bias"""
        if self.distribution == "kaiming_uniform":
            initializer = tf.keras.initializers.HeUniform()
        elif self.distribution == "kaiming_normal":
            initializer = tf.keras.initializers.HeNormal()
        elif self.distribution == "glorot_normal":
            initializer = tf.keras.initializers.GlorotNormal()
        else:
            initializer = tf.keras.initializers.HeUniform()
        self.kernel.assign(initializer(self.kernel.shape))
        if self.rand_bias:
            self.bias.assign(initializer(self.bias.shape))

        tf.cond(
            randconv_printing,
            partial(tf.print, "Randomization with ", self.distribution),
            partial(tf.print, end=""),
        )  # Message for test

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Summary: Forward pass

        :param inputs: input
        :type inputs: tf.Tensor
        :return: output
        :rtype: tf.Tensor
        """
        tf.cond(self.is_randomizing, self.randomize, lambda: None)
        output = super(RandConv1D, self).call(inputs)

        if self.clamp_output == "clamp":
            output = tf.clip_by_value(output, self.range_low, self.range_up)

        return output


###############################################################################################################################################
# MULTI_SCALE_RAND_CONV
###############################################################################################################################################

"""
MultiScaleRandConv : class which encapsulate multiple random convolutions with
different kernel_size
"""


@register_keras_custom_object
class MultiScaleRandConv(tf.keras.layers.Layer):
    """
    Summary: multi scale random convolution

    :param tf: parent class
    :type tf: tf.keras.layers.Layer
    """

    def __init__(
        self,
        filters: int,
        kernel_sizes: List[int],
        mode: str = "2D",
        rand_bias: bool = False,
        distribution: str = "kaiming_normal",
        clamp_output: bool = False,
        range_up: float = 0,
        range_low: float = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Summary: initialisaton

        :param filters: number of filters
        :type filters: int
        :param kernel_size: kernel size
        :type kernel_size: int
        :param rand_bias: use rand bias if true, defaults to False
        :type rand_bias: bool, optional
        :param distribution: initialization distribution, defaults to
        "kaiming_normal"
        :type distribution: str, optional
        :param clamp_output: clamp output if true, defaults to False
        :type clamp_output: bool, optional
        :param range_up: range of clamp, defaults to 0
        :type range_up: float, optional
        :param range_low: range of clamp, defaults to 1
        :type range_low: float, optional
        """
        super(MultiScaleRandConv, self).__init__()

        if not (isinstance(kernel_sizes, list)):
            msg = "Kernel_sizes should be a list."
            raise TypeError(msg)

        self.filters = filters

        self.kernel_sizes = kernel_sizes
        self.max_num_kernel = len(kernel_sizes)
        self.num_kernel = tf.Variable(
            initial_value=2,
            dtype=tf.int32,
            trainable=False,
        )

        self.rand_bias = rand_bias
        self.distribution = distribution

        self.clamp_output = clamp_output
        self.range_low = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_low,
                trainable=False,
                dtype=tf.float32,
            )
        )
        self.range_up = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=range_up,
                trainable=False,
                dtype=tf.float32,
            )
        )

        if mode == "2D":
            self.layers = [
                RandConv2D(
                    self.filters,
                    kernel_size,
                    rand_bias=self.rand_bias,
                    distribution=self.distribution,
                    clamp_output=self.clamp_output,
                    range_up=self.range_up,
                    range_low=self.range_low,
                    **kwargs,
                )
                for kernel_size in self.kernel_sizes
            ]
        else:
            self.layers = [
                RandConv1D(
                    self.filters,
                    kernel_size,
                    rand_bias=self.rand_bias,
                    distribution=self.distribution,
                    clamp_output=self.clamp_output,
                    range_up=self.range_up,
                    range_low=self.range_low,
                    **kwargs,
                )
                for kernel_size in self.kernel_sizes
            ]

    def convolution(self, inputs: tf.Tensor, i: int) -> tf.Tensor:
        """
        Summary: convolution with kernel i

        :param inputs: inputs
        :type inputs: tf.Tensor
        :param i: index of kernel
        :type i: int
        :return: output
        :rtype: tf.Tensor
        """
        tf.cond(
            randconv_printing,
            partial(tf.print, "Convolutions with kernel:", i),
            partial(tf.print, end=""),
        )  # Message for test

        return self.layers[i](inputs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Summary: convolution with all kernels

        :param inputs: inputs
        :type inputs: tf.Tensor
        :return: output
        :rtype: tf.Tensor
        """
        self.num_kernel.assign(
            tf.random.uniform(
                shape=[],
                minval=0,
                maxval=self.max_num_kernel,
                dtype=tf.int32,
            ),
        )
        outputs = tf.case(
            [
                (
                    tf.equal(self.num_kernel, i),
                    partial(self.convolution, inputs, i),
                )
                for i in range(self.max_num_kernel)
            ],
            exclusive=True,
        )
        return outputs


"""
RandConvModule : class which encapsulate the multiple random convolutions
and other functionalities of "Robust and Generalizable Visual Representation
Learning via Random Convolutions" such as normalization and mixing.
"""


@register_keras_custom_object
class RandConvModule(tf.keras.layers.Layer):
    """
    Summary: RandConvModule

    :param tf: parent class
    :type tf: tf.keras.layers.Layer
    """

    def __init__(
        self,
        filters: int,
        kernel_sizes: List[int],
        mode: str = "2D",
        rand_bias: bool = False,
        mixing: bool = False,
        identity_prob: float = 0.0,
        distribution: str = "kaiming_normal",
        data_mean: List[float] = None,
        data_std: List[float] = None,
        clamp_output: bool = False,
    ) -> None:
        """
        Summary: initialisaton

        :param filters: number of filters
        :type filters: int
        :param kernel_sizes:  kernel sizes
        :type kernel_sizes: List[int]
        :param mode: 1D or 2D, defaults to "2D"
        :type mode: str, optional
        :param rand_bias: use bias if true, defaults to False
        :type rand_bias: bool, optional
        :param mixing: use mixing or not, defaults to False
        :type mixing: bool, optional
        :param identity_prob: probability to use mixing (between 0 an 1),
        defaults to 0.0
        :type identity_prob: float, optional
        :param distribution: random function to assign the value of the
        convolution kernel, defaults to "kaiming_normal"
        :type distribution: str, optional
        :param data_mean: mean of dataset, defaults to None
        :type data_mean: List[float], optional
        :param data_std: std of dataset, defaults to None
        :type data_std: List[float], optional
        :param clamp_output: clamp the output if True, defaults to False
        :type clamp_output: bool, optional
        """
        super(RandConvModule, self).__init__()

        # Variable to decide if you randomize self.alpha at each forward pass
        self.is_randomizing = tf.Variable(
            initial_value=True,
            trainable=False,
            dtype=tf.bool,
        )

        self.filters = filters

        if not (isinstance(kernel_sizes, list)):
            msg = "Kernel_sizes should be a list."
            raise TypeError(msg)

        self.kernel_sizes = kernel_sizes

        self.mode = mode
        self.rand_bias = rand_bias
        self.mixing = mixing
        self.alpha = (
            None
            if not self.mixing
            else tf.Variable(
                initial_value=0.0,
                trainable=False,
                dtype=tf.float32,
            )
        )  # Random coefficient for mixing
        self.identity_prob = identity_prob
        self.distribution = distribution

        # if the input is not normalized, we need to normalized with given mean
        # and std (tensor of size 3)
        self.data_mean = (
            None
            if data_mean is None
            else tf.Variable(
                initial_value=data_mean,
                trainable=False,
                dtype=tf.float32,
            )
        )
        self.data_std = (
            None
            if data_std is None
            else tf.Variable(
                initial_value=data_std,
                trainable=False,
                dtype=tf.float32,
            )
        )

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        self.range_low = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=(tf.zeros_like(self.data_mean) - self.data_mean)
                / self.data_std,
                trainable=False,
                dtype=tf.float32,
            )
        )
        self.range_up = (
            None
            if not self.clamp_output
            else tf.Variable(
                initial_value=(tf.ones_like(self.data_mean) - self.data_mean)
                / self.data_std,
                trainable=False,
                dtype=tf.float32,
            )
        )

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Summary: build kernels and biases

        :param input_shape: input shape
        :type input_shape: Tuple[int]
        """
        if self.mixing:
            self.filters = input_shape[-1]
        print(
            "Add RandConv layer with kernel size {}, output channel {}".format(
                self.kernel_sizes,
                self.filters,
            ),
        )
        self.randconv = MultiScaleRandConv(
            self.filters,
            self.kernel_sizes,
            mode=self.mode,
            rand_bias=self.rand_bias,
            distribution=self.distribution,
            clamp_output=self.clamp_output,
            range_low=self.range_low,
            range_up=self.range_up,
        )
        self.built = True

    def randomize(self) -> None:
        """Summary: randomize kernel and bias"""
        if self.mixing:
            self.alpha.assign(
                tf.random.uniform(
                    shape=[],
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32,
                ),
            )
            tf.cond(
                randconv_printing,
                partial(tf.print, "new alpha:", self.alpha),
                partial(tf.print, end=""),
            )  # Message for test

    def mixing_func(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Summary: mix original input and generated input

        :param inputs: inputs
        :type inputs: tf.Tensor
        :return: outputs
        :rtype: tf.Tensor
        """
        outputs = self.randconv(inputs)
        if self.mixing:
            outputs = self.alpha * outputs + (1 - self.alpha) * inputs

        if self.clamp_output:
            outputs = tf.clip_by_value(outputs, self.range_low, self.range_up)

        return outputs

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Summary: forward pass

        :param inputs: inputs
        :type inputs: tf.Tensor
        :return: outputs
        :rtype: tf.Tensor
        """
        # assume that the input is whightened
        tf.cond(self.is_randomizing, self.randomize, lambda: None)
        outputs = tf.cond(
            tf.math.less(
                tf.random.uniform(
                    shape=[],
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32,
                ),
                self.identity_prob,
            ),
            partial(self.mixing_func, inputs),
            lambda: inputs,
        )
        return outputs
