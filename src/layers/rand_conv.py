__all__=['RandConv2D','RandConv1D','MultiScaleRandConv','RandConvModule','randconv_printing']

import tensorflow as tf
from functools import partial

from .utils import register_keras_custom_object

#Variable to print result for test
randconv_printing = tf.Variable(initial_value=False, trainable=False,dtype=tf.bool)

###############################################################################################################################################
#SINGLE_SCALE_RAND_CONV (1D and 2D)
###############################################################################################################################################

"""
Random 2D-convolutions layer

Arguments : -base argument of Conv2D(filters,kernel_size,strides, padding...)
            -rand_bias (boolean): use bias or not for the random convolution
            -distribution (string): random function to assign the value of the convolution kernel (support kaiming_uniform, kaiming_normal and glorot_normal distribution)
            -clamp_output(boolean): clamp the output of the random convolution or not
            -range_up (float): upper bound for clamping
            -range_low (float): lower bound for clamping
"""
@register_keras_custom_object
class RandConv2D(tf.keras.layers.Conv2D):

    def __init__(self, filters, kernel_size, rand_bias=False,
                 distribution='kaiming_normal',
                 clamp_output=None, range_up=None, range_low=None, **kwargs):
        super(RandConv2D, self).__init__(filters=filters, kernel_size=kernel_size, use_bias=rand_bias,trainable=False,padding='same', **kwargs)
        
        #Variable to decide if you randomize the kernel at each forward pass
        self.is_randomizing = tf.Variable(initial_value=True, trainable=False,dtype=tf.bool)
        
        #Usefull variable for random convolution like in https://github.com/wildphoton/RandConv
        self.rand_bias = rand_bias
        self.distribution = distribution
        self.clamp_output = clamp_output
        self.range_low = None if not self.clamp_output else tf.Variable(initial_value=range_low, trainable=False,dtype=tf.float32)
        self.range_up = None if not self.clamp_output else tf.Variable(initial_value=range_up, trainable=False,dtype=tf.float32)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"
            
    def randomize(self):
        if self.distribution == 'kaiming_uniform':
            initializer=tf.keras.initializers.HeUniform()
        elif self.distribution == 'kaiming_normal':
            initializer=tf.keras.initializers.HeNormal()
        elif self.distribution == 'glorot_normal':
            initializer=tf.keras.initializers.GlorotNormal()
        else:
            raise NotImplementedError()
        self.kernel.assign(initializer(self.kernel.shape))
        if self.rand_bias:
            self.bias.assign(initializer(self.bias.shape))
            
        tf.cond(randconv_printing,partial(tf.print,"Randomization with ",self.distribution),partial(tf.print,end=''))#Message for test
            
    def call(self, inputs):
        tf.cond(self.is_randomizing,self.randomize,lambda:None)
        output = super(RandConv2D, self).call(inputs)

        if self.clamp_output == 'clamp':
            output=tf.clip_by_value(output,self.range_low,self.range_up)

        return output
        
        
"""
Random 1D-convolutions layer

Arguments : -base argument of Conv2D(filters,kernel_size,strides, padding...)
            -rand_bias (boolean): use bias or not for the random convolution
            -distribution (string): random function to assign the value of the convolution kernel (support kaiming_uniform, kaiming_normal and glorot_normal distribution)
            -clamp_output(boolean): clamp the output of the random convolution or not
            -range_up (float): upper bound for clamping
            -range_low (float): lower bound for clamping
"""
        
@register_keras_custom_object
class RandConv1D(tf.keras.layers.Conv1D):

    def __init__(self, filters, kernel_size, rand_bias=False,
                 distribution='kaiming_normal',
                 clamp_output=None, range_up=None, range_low=None, **kwargs):
        super(RandConv1D, self).__init__(filters=filters, kernel_size=kernel_size, use_bias=rand_bias,trainable=False,padding='same', **kwargs)
        
        #Variable to decide if you randomize the kernel at each forward pass
        self.is_randomizing = tf.Variable(initial_value=True, trainable=False,dtype=tf.bool)
        
        #Usefull variable for random convolution like in https://github.com/wildphoton/RandConv
        self.rand_bias = rand_bias
        self.distribution = distribution
        self.clamp_output = clamp_output
        self.range_low = None if not self.clamp_output else tf.Variable(initial_value=range_low, trainable=False,dtype=tf.float32)
        self.range_up = None if not self.clamp_output else tf.Variable(initial_value=range_up, trainable=False,dtype=tf.float32)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"
            
    def randomize(self):
        if self.distribution == 'kaiming_uniform':
            initializer=tf.keras.initializers.HeUniform()
        elif self.distribution == 'kaiming_normal':
            initializer=tf.keras.initializers.HeNormal()
        elif self.distribution == 'glorot_normal':
            initializer=tf.keras.initializers.GlorotNormal()
        else:
            raise NotImplementedError()
        self.kernel.assign(initializer(self.kernel.shape))
        if self.rand_bias:
            self.bias.assign(initializer(self.bias.shape))
            
        tf.cond(randconv_printing,partial(tf.print,"Randomization with ",self.distribution),partial(tf.print,end=''))#Message for test
            
    def call(self, inputs):
        tf.cond(self.is_randomizing,self.randomize,lambda:None)
        output = super(RandConv1D, self).call(inputs)

        if self.clamp_output == 'clamp':
            output=tf.clip_by_value(output,self.range_low,self.range_up)

        return output
        
###############################################################################################################################################
#MULTI_SCALE_RAND_CONV
###############################################################################################################################################
   
"""
MultiScaleRandConv : class which encapsulate multiple random convolutions with different kernel_size

Arguments : -base argument of Conv2D(filters,strides, padding...) except for kernel_size
            -kernel_sizes (list of integers): kernel sizes of the different random convolutions
            -mode (string): use 1D convolutions ("1D") or 2D convolutions ("2D")
            -rand_bias (boolean): use bias or not for the random convolution
            -distribution (string): random function to assign the value of the convolution kernel (support kaiming_uniform, kaiming_normal and glorot_normal distribution)
            -clamp_output(boolean): clamp the output of the random convolution or not
            -range_up (float): upper bound for clamping
            -range_low (float): lower bound for clamping
"""

@register_keras_custom_object        
class MultiScaleRandConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_sizes,mode="2D",
                 rand_bias=False, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs):
                 
        super(MultiScaleRandConv, self).__init__()
        
        assert isinstance(kernel_sizes, list),"Kernel_sizes should be a list."
        
        self.filters=filters
        
        self.kernel_sizes=kernel_sizes
        self.max_num_kernel=len(kernel_sizes)
        self.num_kernel=tf.Variable(initial_value=2,dtype=tf.int32, trainable=False)
        
        self.rand_bias=rand_bias
        self.distribution=distribution
        
        self.clamp_output = clamp_output
        self.range_low = None if not self.clamp_output else tf.Variable(initial_value=range_low, trainable=False,dtype=tf.float32)
        self.range_up = None if not self.clamp_output else tf.Variable(initial_value=range_up, trainable=False,dtype=tf.float32)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"
            
        if mode=="2D":   
            self.layers=[RandConv2D(self.filters, kernel_size, rand_bias=self.rand_bias,distribution=self.distribution,clamp_output=self.clamp_output, range_up=self.range_up, range_low=self.range_low, **kwargs) for kernel_size in self.kernel_sizes]
        else:
            self.layers=[RandConv1D(self.filters, kernel_size, rand_bias=self.rand_bias,distribution=self.distribution,clamp_output=self.clamp_output, range_up=self.range_up, range_low=self.range_low, **kwargs) for kernel_size in self.kernel_sizes]
    
        
    def convolution(self,inputs,i):
        tf.cond(randconv_printing,partial(tf.print,"Convolutions with kernel:",i),partial(tf.print,end=''))#Message for test
        
        return self.layers[i](inputs)
    
        
    def call(self,inputs):
        self.num_kernel.assign(tf.random.uniform(shape=[], minval=0, maxval=self.max_num_kernel, dtype=tf.int32))
        outputs = tf.case([(tf.equal(self.num_kernel, i),partial(self.convolution,inputs,i)) for i in range(self.max_num_kernel)], exclusive=True)
        return outputs
        
        
"""
RandConvModule : class which encapsulate the multiple random convolutions and other functionalities of "Robust and Generalizable Visual Representation Learning via Random Convolutions" such as normalization and mixing.

Arguments : -filters (integer): number of output filters for the random convolutions
            -kernel_sizes (list of integers): kernel sizes of the different random convolutions
            -mode (string): use 1D convolutions ("1D") or 2D convolutions ("2D")
            -rand_bias (boolean): use bias or not for the random convolution
            -mixing (boolean): use mixing or not
            -identity_prob (boolean between 0 and 1): probability to use mixing
            -distribution (string): random function to assign the value of the convolution kernel (support kaiming_uniform, kaiming_normal and glorot_normal distribution)
            -clamp_output(boolean): clamp the output of the random convolution or not
            -data_mean (float): mean of dataset (use for clamping)
            -data_std (float): std of dataset (use for clamping)
"""       
        
@register_keras_custom_object        
class RandConvModule(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_sizes, mode="2D",
                 rand_bias=False,mixing=False,
                 identity_prob=0.0, distribution='kaiming_normal',
                 data_mean=None, data_std=None, clamp_output=False):
                 
        super(RandConvModule, self).__init__()
        
        #Variable to decide if you randomize self.alpha at each forward pass
        self.is_randomizing = tf.Variable(initial_value=True, trainable=False,dtype=tf.bool)
        
        self.filters=filters
        assert isinstance(kernel_sizes, list),"Kernel_sizes should be a list."
        self.kernel_sizes=kernel_sizes
        
        self.mode = mode
        self.rand_bias=rand_bias
        self.mixing=mixing
        self.alpha = None if not self.mixing else tf.Variable(initial_value=0.0, trainable=False,dtype=tf.float32)#Random coefficient for mixing
        self.identity_prob=identity_prob
        self.distribution=distribution
        
        # if the input is not normalized, we need to normalized with given mean and std (tensor of size 3)
        self.data_mean = None if data_mean is None else tf.Variable(initial_value=data_mean, trainable=False,dtype=tf.float32)
        self.data_std = None if data_std is None else tf.Variable(initial_value=data_std, trainable=False,dtype=tf.float32)

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std to do output range adjust"
        self.range_low = None if not self.clamp_output else tf.Variable(initial_value=(tf.zeros_like(self.data_mean)-self.data_mean)/self.data_std, trainable=False,dtype=tf.float32)
        self.range_up = None if not self.clamp_output else tf.Variable(initial_value=(tf.ones_like(self.data_mean)-self.data_mean)/self.data_std, trainable=False,dtype=tf.float32)
        
    def build(self, input_shape):
        
        #assert (self.mixing==False) or (self.mixing==True and input_shape[-1]==self.filters),"Input channels and output channels should be equal in mixing mode."
        if self.mixing:
            self.filters=input_shape[-1]
        print("Add RandConv layer with kernel size {}, output channel {}".format(self.kernel_sizes, self.filters))
        self.randconv = MultiScaleRandConv(self.filters, self.kernel_sizes, mode=self.mode, rand_bias=self.rand_bias,
                                             distribution=self.distribution,
                                             clamp_output=self.clamp_output,
                                             range_low=self.range_low,
                                             range_up=self.range_up)
        self.built=True
        
    def randomize(self):
        if self.mixing:
            self.alpha.assign(tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32))
            tf.cond(randconv_printing,partial(tf.print,"new alpha:",self.alpha),partial(tf.print,end=''))#Message for test
            
        
    def mixing_func(self,inputs):
    
        outputs = self.randconv(inputs)
        if self.mixing:    
            outputs = self.alpha*outputs + (1-self.alpha)*inputs
            
        if self.clamp_output:
            outputs = tf.clip_by_value(outputs,self.range_low,self.range_up)
            
        return outputs
        
        
    def call(self,inputs):
        
        #assume that the input is whightened
        tf.cond(self.is_randomizing,self.randomize,lambda:None)
        outputs=tf.cond(tf.math.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32),self.identity_prob),partial(self.mixing_func,inputs),lambda : inputs)
        return outputs