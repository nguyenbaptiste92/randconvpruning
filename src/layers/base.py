__all__ = ['dict_basic_func']

import tensorflow as tf

"""
Basic operation of tensorflow layers
Used in pruning.py and quantization.py
"""

def conv2d_passby(layer,inputs,kernel):
    data_format="NHWC" if layer.data_format=="channels_last" else "NCHW"
    padding = "VALID" if layer.padding=="valid" else "SAME"
    return tf.nn.conv2d(inputs, kernel, layer.strides, padding,data_format=data_format, dilations=layer.dilation_rate)
    
def conv1d_passby(layer,inputs,kernel):
    data_format="NHWC" if layer.data_format=="channels_last" else "NCHW"
    padding = "VALID" if layer.padding=="valid" else "SAME"
    return tf.nn.conv1d(inputs, kernel, layer.strides, padding,data_format=data_format, dilations=layer.dilation_rate)
    
def dense_passby(layer,inputs,kernel):
    rank = inputs.shape.rank
    return tf.tensordot(inputs, kernel, [[rank - 1], [0]])
    
dict_basic_func={tf.keras.layers.Dense:dense_passby,tf.keras.layers.Conv2D:conv2d_passby,tf.keras.layers.Conv1D:conv1d_passby}