__all__ = ["PrunedTemporalCNN1D"]


from typing import Tuple

import tensorflow as tf

from src.layers.pruning import pruning

#######################################################################
# PrunedTemporalCNN1D : network of "A Systematic Study of Unsupervised
# Domain Adaptation for Robust Human-Activity Recognition" for the
# RealWorld HAR dataset
# (instance normalization layer are replace with batch normalization)
# Pruned version (all layers ar pruned except the last layer)
#######################################################################


def PrunedTemporalCNN1D(  # noqa: N802
    input_shape: Tuple[int] = (150, 3),
    output_units: int = 14,
) -> tf.keras.Model:
    """
    Summary: PrunedTemporalCNN1D model

    :param input_shape: input_shape of the model, defaults to (150, 3)
    :type input_shape: Tuple[int], optional
    :param output_units: numbr of classes, defaults to 14
    :type output_units: int, optional
    :return: model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = pruning(
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti1 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn1 = tf.keras.layers.BatchNormalization()
    conv2 = pruning(
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti2 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn2 = tf.keras.layers.BatchNormalization()
    conv3 = pruning(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=4,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti3 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn3 = tf.keras.layers.BatchNormalization()
    conv4 = pruning(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti4 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn4 = tf.keras.layers.BatchNormalization()
    conv5 = pruning(
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=4,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti5 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn5 = tf.keras.layers.BatchNormalization()
    conv6 = pruning(
        tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=5,
            activation="linear",
            use_bias=True,
            padding="valid",
        ),
    )
    acti6 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn6 = tf.keras.layers.BatchNormalization()

    gap1 = tf.keras.layers.GlobalAveragePooling1D()
    dense1 = tf.keras.layers.Dense(
        units=output_units,
        activation="linear",
        use_bias=True,
    )
    acti7 = tf.keras.layers.Activation(activation="softmax")

    x = conv1(inputs)
    x = acti1(x)
    x = bn1(x)
    x = conv2(x)
    x = acti2(x)
    x = bn2(x)
    x = conv3(x)
    x = acti3(x)
    x = bn3(x)
    x = conv4(x)
    x = acti4(x)
    x = bn4(x)
    x = conv5(x)
    x = acti5(x)
    x = bn5(x)
    x = conv6(x)
    x = acti6(x)
    x = bn6(x)

    y = gap1(x)
    y = dense1(y)
    outputs = acti7(y)

    return tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
        name="PrunedTemporalCNN1D",
    )
