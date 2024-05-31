__all__ = ['register_keras_custom_object','register_alias','get_signature','get_args']

import tensorflow as tf
import inspect

def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls
    
def register_alias(name: str):
    """A decorator to register a custom keras object under a given alias.
    !!! example
        ```python
        @utils.register_alias("degeneration")
        class Degeneration(tf.keras.metrics.Metric):
            pass
        ```
    """

    def register_func(cls):
        tf.keras.utils.get_custom_objects()[name] = cls
        return cls

    return register_func
    
"""
Get signature of a tensorflow layer
Used in pruning.py and quantization.py
"""
    
def get_signature(base_layer):
    
    Type = type(base_layer)
    if hasattr(base_layer,"signature"):
        return base_layer.signature
    else:
        return {'base_layer':(Type,dict(inspect.signature(Type.__init__).parameters))}

"""
Get the parameters used during the creation of a tensorflow layer
Used in pruning.py and quantization.py
"""

def get_args(signature,base_layer):

    Type = type(base_layer)
    layer_dict=base_layer.__dict__
    
    if 'base_layer' not in signature['base_layer'][1]:
        parameter_value_dict={k:layer_dict[k] for k in signature['base_layer'][1] if k in layer_dict}
        return (signature['base_layer'][0],parameter_value_dict)    
    else:
        layer_type,layer_param=get_args(signature['base_layer'][1],base_layer)
        copy_signature = signature.copy()
        copy_signature.pop('base_layer', None)
        parameter_value_dict = {k:layer_dict[k] for k in copy_signature if k in layer_dict}
        parameter_value_dict['base_layer'] = layer_type(**layer_param)
        return (signature['base_layer'][0],parameter_value_dict)