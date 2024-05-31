__all__ = ['pruning']

import tensorflow as tf
import inspect

from .utils import register_keras_custom_object, get_signature, get_args
from .base import dict_basic_func

"""
- The pruning layer can be chained : pruning(quantization(Dense(10))) will created a dense layer which pruned its kernel then quantized its kernel before
realized a dot product.
- The pruning layer support "Dense and Conv2D and Conv1D" layers.
- To add more compatible layers, you should:
    - Take a base layer with an attribute "kernel, use_bias, bias and activation" like the Dense layer (you can manually create these attributes).
    - Add to the dictionary .base.dict_basic_func your base layer and a corresponding function which have layer,inputs and kernel as parameters and return one output:
        dense_passby(layer,inputs,kernel):
            rank = inputs.shape.rank
            return tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            
        dict_basic_func[tf.keras.layers.Dense]=dense_passby
        
This layer supports only the save of weights and not the save of model.
"""

def pruning(base_layer,prune_bias=False,last=False):
 
    @register_keras_custom_object
    class Pruning(type(base_layer)):
    
        def __init__(self, base_layer,prune_bias=False,last=False):
        
            #Add assert to limit the quantizer and base_layer choice
            
            signature=get_signature(base_layer)
            layer_type,layer_param = get_args(signature,base_layer)
            layer_type.__init__(self,**layer_param)
            
            self.set_signature(base_layer)
            self.set_basetype(base_layer)
            self.prune_bias = prune_bias
            self.last=last
            
        def build(self,input_shape):
        
            self.base_type.build(self,input_shape)
            self.kernel_mask = self.add_weight(shape=self.kernel.shape,
                                         initializer=tf.keras.initializers.Ones,
                                         name='kernel_mask',
                                         trainable=False)
            if self.use_bias and prune_bias:
                self.bias_mask = self.add_weight(shape=self.bias.shape,
                                         initializer=tf.keras.initializers.Ones,
                                         name='bias_mask',
                                         trainable=False)
            else:
                self.bias_mask = None
            self.built = True
            
         
        # Set the signature of the pruning Dense Layer    
        def set_signature(self,base_layer):
        
            #Get signature of base_layer
            signature = get_signature(base_layer)
            self.signature = {"base_layer":(type(self),signature)}
            # Append arguments of current layer to signature
            self.signature["prune_bias"]=inspect.Parameter('prune_bias', inspect.Parameter.POSITIONAL_OR_KEYWORD)
            self.signature["last"]=inspect.Parameter('last', inspect.Parameter.POSITIONAL_OR_KEYWORD)
            
        def set_basetype(self,base_layer):
        
            if hasattr(base_layer,"base_type"):
                self.base_type = base_layer.base_type
            else:
                self.base_type = type(base_layer)
                
        def call(self,inputs):
        
            kernel = self.kernel
            bias = self.bias
            signature = self.signature
            
            #Realize the operation of all layer (Quantization,pruning...)
            while signature['base_layer'][0] is not self.base_type:
                inputs,kernel,bias = signature['base_layer'][0].passby(self,inputs,kernel,bias)
                signature = signature['base_layer'][1]
                
            #Realize the operation of the Layer
            output = dict_basic_func[self.base_type](self,inputs,kernel)
            if self.use_bias:
                output = tf.nn.bias_add(output, bias)
            output=self.activation(output) 
            return output
            
        def passby(self,inputs,kernel,bias):
            kernel = tf.math.multiply(kernel,self.kernel_mask)
            if self.bias_mask is not None:
                bias = tf.math.multiply(bias,self.bias_mask)
            return inputs,kernel,bias
            
    return Pruning(base_layer,prune_bias,last)