__all__ = ["DigitNet", "PrunedDigitNet", "SmallDigitNet", "SmallPrunedDigitNet"]


from typing import Tuple

import tensorflow as tf  # type: ignore  # noqa: PGH003

from src.layers.pruning import pruning


#################################################################################
# DIGITNET : network of "ROBUST AND GENERALIZABLE VISUAL REPRESENTATION LEARNING
# VIA RANDOM CONVOLUTIONS" for the digits benchmark
# Normal version and Pruned version (all layers ar pruned except the last layer)
#################################################################################
def DigitNet(  # noqa: N802
    input_shape: Tuple[int] = (32, 32, 3),
    output_units: int = 10,
) -> tf.keras.Model:
    """
    Summary: DigitNet model

    :param input_shape: input_shape of the model, defaults to (32, 32, 3)
    :type input_shape: Tuple[int], optional
    :param output_units: numbr of classes, defaults to 10
    :type output_units: int, optional
    :return: model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1 = tf.keras.layers.Conv2D(64, (5, 5), activation="relu")
    mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    conv2 = tf.keras.layers.Conv2D(128, (5, 5), activation="relu")
    mp2 = tf.keras.layers.MaxPooling2D((2, 2))

    flatt = tf.keras.layers.Flatten()
    fc1 = tf.keras.layers.Dense(1024, activation="relu")
    fc2 = tf.keras.layers.Dense(1024, activation="relu")
    fc3 = tf.keras.layers.Dense(output_units, activation="linear")
    acti1 = tf.keras.layers.Activation(activation="softmax")

    x = conv1(inputs)
    x = mp1(x)
    x = conv2(x)
    x = mp2(x)
    x = flatt(x)
    y = fc1(x)
    y = fc2(y)
    y = fc3(y)
    outputs = acti1(y)

    return tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
        name="DigitNet",
    )


def PrunedDigitNet(  # noqa: N802
    input_shape: Tuple[int] = (32, 32, 3),
    output_units: int = 10,
) -> tf.keras.Model:
    """
    Summary: Pruned DigitNet model

    :param input_shape: input_shape of the model, defaults to (32, 32, 3)
    :type input_shape: Tuple[int], optional
    :param output_units: numbr of classes, defaults to 10
    :type output_units: int, optional
    :return: model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1 = pruning(tf.keras.layers.Conv2D(64, (5, 5), activation="relu"))
    mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    conv2 = pruning(tf.keras.layers.Conv2D(128, (5, 5), activation="relu"))
    mp2 = tf.keras.layers.MaxPooling2D((2, 2))
    flatt = tf.keras.layers.Flatten()
    fc1 = pruning(tf.keras.layers.Dense(1024, activation="relu"))
    fc2 = pruning(tf.keras.layers.Dense(1024, activation="relu"))
    fc3 = pruning(
        tf.keras.layers.Dense(output_units, activation="linear"),
        last=True,
    )
    acti1 = tf.keras.layers.Activation(activation="softmax")

    x = conv1(inputs)
    x = mp1(x)
    x = conv2(x)
    x = mp2(x)
    y = flatt(x)
    y = fc1(y)
    y = fc2(y)
    y = fc3(y)
    outputs = acti1(y)

    return tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
        name="PrunedDigitNet",
    )


################################################################################
# SMALLDIGITNET : network of "ROBUST AND GENERALIZABLE VISUAL REPRESENTATION
# LEARNING VIA RANDOM CONVOLUTIONS" scaled to have a number of
# parameters equivalent to Resnet20
# Normal version and Pruned version (all layers ar pruned except the last layer)
################################################################################


def SmallDigitNet(  # noqa: N802
    input_shape: Tuple[int] = (32, 32, 3),
    output_units: int = 10,
) -> tf.keras.Model:
    """
    Summary: Small DigitNet model

    :param input_shape: input_shape of the model, defaults to (32, 32, 3)
    :type input_shape: Tuple[int], optional
    :param output_units: numbr of classes, defaults to 10
    :type output_units: int, optional
    :return: model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation="relu")
    mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation="relu")
    mp2 = tf.keras.layers.MaxPooling2D((2, 2))

    flatt = tf.keras.layers.Flatten()
    fc1 = tf.keras.layers.Dense(128, activation="relu")
    fc2 = tf.keras.layers.Dense(128, activation="relu")
    fc3 = tf.keras.layers.Dense(output_units, activation="linear")
    acti1 = tf.keras.layers.Activation(activation="softmax")

    x = conv1(inputs)
    x = mp1(x)
    x = conv2(x)
    x = mp2(x)
    x = flatt(x)
    y = fc1(x)
    y = fc2(y)
    y = fc3(y)
    outputs = acti1(y)

    return tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
        name="SmallDigitNet",
    )


def SmallPrunedDigitNet(  # noqa: N802
    input_shape: Tuple[int] = (32, 32, 3),
    output_units: int = 10,
) -> tf.keras.Model:
    """
    Summary: Small Pruned DigitNet model

    :param input_shape: input_shape of the model, defaults to (32, 32, 3)
    :type input_shape: Tuple[int], optional
    :param output_units: numbr of classes, defaults to 10
    :type output_units: int, optional
    :return: model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1 = pruning(tf.keras.layers.Conv2D(32, (5, 5), activation="relu"))
    mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    conv2 = pruning(tf.keras.layers.Conv2D(64, (5, 5), activation="relu"))
    mp2 = tf.keras.layers.MaxPooling2D((2, 2))
    flatt = tf.keras.layers.Flatten()
    fc1 = pruning(tf.keras.layers.Dense(128, activation="relu"))
    fc2 = pruning(tf.keras.layers.Dense(128, activation="relu"))
    fc3 = pruning(
        tf.keras.layers.Dense(output_units, activation="linear"),
        last=True,
    )
    acti1 = tf.keras.layers.Activation(activation="softmax")

    x = conv1(inputs)
    x = mp1(x)
    x = conv2(x)
    x = mp2(x)
    y = flatt(x)
    y = fc1(y)
    y = fc2(y)
    y = fc3(y)
    outputs = acti1(y)

    return tf.keras.Model(
        inputs={"image": inputs},
        outputs={"label": outputs, "output": y, "feature": x},
        name="SmallPrunedDigitNet",
    )
