__all__ = ['IdentityTransform','Cast','Rescale','Reshape','ToGrayScale','GreyToColor','Normalize','DeNormalize','ColorJitter','RandomGrayscale','Filter_label']

import tensorflow as tf

def IdentityTransform():
    """do nothing"""

    def func(x):
        return x
        
    return func
    
"""
Realized one hot encoding
"""
def Cast(num_label):

    def func(x):
        im, lbl = x["image"], x["label"]
        im=tf.cast(im, tf.float32)
        lbl = tf.cast(tf.one_hot(lbl, num_label),tf.float32)
        return {"image":im,"label":lbl}
        
    return func
  
"""
Rescaled the input of the dataset
"""  
def Rescale(value):

    def func(x):
        im, lbl = x["image"], x["label"]
        im = im / value
        return {"image":im,"label":lbl}
        
    return func
      
"""
Reshaped the input of the dataset : work for image, need to be adapted for temporal series
"""  
def Reshape(shape):
    
    def func(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(tf.rank(im),3,message="images should have a rank of 3")
        tf.debugging.assert_equal(len(shape),2,message="shape should have a rank of 2")
        im = tf.image.resize(im, shape)
        return {"image":im,"label":lbl}
    
    return func

"""
Put the input of the dataset on grey scale: work for image
"""    
def ToGrayScale():

    def func(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(tf.rank(im),3,message="images should have a rank of 3")
        im = tf.image.rgb_to_grayscale(im)
        return {"image":im,"label":lbl}
        
    return func
    
"""
Put the input of the dataset on color scale: work for image
"""
def GreyToColor():
    
    def func(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(tf.rank(im),3,message="images should have a rank of 3")
        shape=im.shape
        im = tf.cond(tf.math.equal(im.shape[-1],1),lambda: tf.broadcast_to(im, im.shape[:-1]+[3]),lambda: im)
        im = tf.reshape(im, shape[:-1]+[3])
        return {"image":im,"label":lbl}
        
    return func

"""
Normalize the input of the dataset
"""
def Normalize(mean,std):

    def func(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(im.shape[-1],len(mean),message="Mean dimension should be equal to the number of channels of the input")
        tf.debugging.assert_equal(im.shape[-1],len(std),message="Std dimension should be equal to the number of channels of the input")
        im = (im - mean) / std
        return {"image":im,"label":lbl}
          
    return func

"""
Unnormalize the input of the dataset
"""    
def DeNormalize(mean,std):

    def func(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(im.shape[-1],len(mean),message="Mean dimension should be equal to the number of channels of the input")
        tf.debugging.assert_equal(im.shape[-1],len(std),message="Std dimension should be equal to the number of channels of the input")
        im = im * std + mean
        return {"image":im,"label":lbl}
          
    return func
    
def ColorJitter(brightness=0, contrast=0, saturation=0, hue=0):

    def func(x):
        im, lbl = x["image"], x["label"]
        
        tf.debugging.assert_non_negative(brightness,message="Brightness should be positive")
        im=tf.image.random_brightness(im,brightness)
        
        tf.debugging.assert_non_negative(contrast,message="Contrast should be positive")
        im=tf.image.random_contrast(im,max(0,1-contrast),1+contrast)
        
        tf.debugging.assert_non_negative(saturation,message="Saturation should be positive")
        im=tf.image.random_saturation(im,max(0,1-saturation),1+saturation)
        
        tf.debugging.assert_non_negative(hue,message="Hue should be positive")
        tf.debugging.assert_less_equal(hue,0.5,message="Hue should be inferior to 0.5")
        im=tf.image.random_hue(im,hue)
        
        return {"image":im,"label":lbl}
        
    return func
    
"""
Put the input of the dataset on grayscale then put it on color with a probability p: only for image
"""
def RandomGrayscale(p=0.1):

    def compose(x):
        im, lbl = x["image"], x["label"]
        tf.debugging.assert_equal(tf.rank(im),3,message="images should have a rank of 3")
        im = tf.image.rgb_to_grayscale(im)
        im = tf.broadcast_to(im, im.shape[:-1]+[3])
        return {"image":im,"label":lbl}
        

    def func(x):
        tf.debugging.assert_equal(im.shape[-1],3,message="Colored images should have 3 channels")
        random_value=tf.random.uniform(shape=[])
        shape=im.shape
        x = tf.cond(tf.math.less(random_value,p),lambda: compose(x),lambda: x)
        im = tf.reshape(im, shape[:-1]+[3])
        return x
    
    return func
    
"""
Filter the dataset by label
"""
def Filter_label(list_label):
    
    def func(x):
        im, lbl = x["image"], x["label"]
        label = tf.math.argmax(lbl)
        return tf.math.reduce_any(tf.equal(list_label, label))
    
    return func
