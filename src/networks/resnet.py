__all__ = ["resnet_layer", "resnet_v1"]


from typing import Tuple

import tensorflow as tf  # type: ignore  # noqa: PGH003

from src.layers.pruning import pruning

################################################################################
# Resnet : network of "Deep Residual Learning for Image Recognition"
# Code taken from https://keras.io/zh/examples/cifar10_resnet/ and adapted
# Normal version and Pruned version (all layers ar pruned except the last layer)
################################################################################


def resnet_layer(
    inputs: tf.Tensor,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    activation: str = "relu",
    batch_normalization: bool = True,
    conv_first: bool = True,
    prune: bool = False,
) -> tf.Tensor:
    """
    Summary: 2D Convolution-Batch Normalization-Activation stack builder

    :param inputs: input tensor from input image or previous layer
    :type inputs: tf.Tensor
    :param num_filters: Conv2D number of filters, defaults to 16
    :type num_filters: int, optional
    :param kernel_size: Conv2D square kernel dimensions, defaults to 3
    :type kernel_size: int, optional
    :param strides: Conv2D square stride dimensions, defaults to 1
    :type strides: int, optional
    :param activation: activation name, defaults to "relu"
    :type activation: str, optional
    :param batch_normalization:  whether to include batch normalization,
    defaults to True
    :type batch_normalization: bool, optional
    :param conv_first: conv-bn-activation (True) or
            bn-activation-conv (False), defaults to True
    :type conv_first: bool, optional
    :param prune: use pruning if True, defaults to False
    :type prune: bool, optional
    :return: tensor as input to the next layer
    :rtype: tf.Tensor
    """
    conv = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    if prune:
        conv = pruning(conv)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(
    input_shape: Tuple[int],
    depth: int,
    num_classes: int = 10,
    prune: bool = False,
) -> tf.keras.Model:
    """
    Summary: ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    :param input_shape: input_shape of the model, defaults to (32, 32, 3)
    :type input_shape: Tuple[int], optional
    :param num_classes: number of classes, defaults to 10
    :type num_classes: int, optional
    :param prune: prune model if true, defaults to False
    :type prune: bool, optional
    :return: model
    :rtype: tf.keras.Model
    """
    if (depth - 2) % 6 != 0:
        msg = "depth should be 6n+2 (eg 20, 32, 44 in [a])"
        raise ValueError(msg)
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, prune=prune)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters,
                strides=strides,
                prune=prune,
            )
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters,
                activation=None,
                prune=prune,
            )
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                    prune=prune,
                )
            x = tf.keras.layers.Add()([x, y])
            x = tf.keras.layers.Activation("relu")(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    y = tf.keras.layers.Flatten()(x)
    if prune:
        y = pruning(
            tf.keras.layers.Dense(
                num_classes,
                activation="linear",
                kernel_initializer="he_normal",
            ),
            last=True,
        )(y)
    else:
        y = tf.keras.layers.Dense(
            num_classes,
            activation="linear",
            kernel_initializer="he_normal",
        )(y)
    outputs = tf.keras.layers.Activation(activation="softmax")(y)

    # Instantiate model.
    model = tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
    )
    return model
