__all__ = [
    "get_args",
    "get_signature",
    "register_alias",
    "register_keras_custom_object",
]

import inspect
from typing import Any, Dict

import tensorflow as tf


def register_keras_custom_object(cls: Any) -> Any:
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


def register_alias(name: str) -> Any:
    """
    Summary: A decorator to register a custom keras object under a given alias.

    :param name: alias name
    :type name: str
    :return: object to register
    :rtype: Any
    """

    def register_func(cls: Any) -> Any:
        tf.keras.utils.get_custom_objects()[name] = cls
        return cls

    return register_func


def get_signature(base_layer: tf.keras.Layer) -> Dict:
    """Summary: get signature of layer"""
    Type = type(base_layer)
    if hasattr(base_layer, "signature"):
        return base_layer.signature
    return {
        "base_layer": (Type, dict(inspect.signature(Type.__init__).parameters)),
    }


def get_args(signature: Dict, base_layer: tf.keras.Layer) -> Dict:
    """
    Summary: Get the parameters used during the creation of a tensorflow layer

    :param signature: signature of layer
    :type signature: Dict
    :param base_layer: layer
    :type base_layer: tf.keras.Layer
    :return: args
    :rtype: Dict
    """
    Type = type(base_layer)
    layer_dict = base_layer.__dict__

    if "base_layer" not in signature["base_layer"][1]:
        parameter_value_dict = {
            k: layer_dict[k]
            for k in signature["base_layer"][1]
            if k in layer_dict
        }
        return (signature["base_layer"][0], parameter_value_dict)
    layer_type, layer_param = get_args(signature["base_layer"][1], base_layer)
    copy_signature = signature.copy()
    copy_signature.pop("base_layer", None)
    parameter_value_dict = {
        k: layer_dict[k] for k in copy_signature if k in layer_dict
    }
    parameter_value_dict["base_layer"] = layer_type(**layer_param)
    return (signature["base_layer"][0], parameter_value_dict)
