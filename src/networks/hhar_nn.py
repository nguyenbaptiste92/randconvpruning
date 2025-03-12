__all__ = ['PrunedTemporalCNN1D']

import numpy as np
import tensorflow as tf
import os,sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from ..layers.pruning import pruning
    
###############################################################################################################################################
#PrunedTemporalCNN1D : network of "A Systematic Study of Unsupervised Domain Adaptation for Robust
#Human-Activity Recognition" for the RealWorld HAR dataset (instance normalization layer are replace with batch normalization)
#Pruned version (all layers ar pruned except the last layer)
###############################################################################################################################################
    
def PrunedTemporalCNN1D(inputs_shape=[150,3],classes=14):

    inputs= tf.keras.Input(shape=inputs_shape)
    conv1 = pruning(tf.keras.layers.Conv1D(filters=16,kernel_size=3,activation='linear',use_bias=True,padding='valid'))
    acti1 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn1= tf.keras.layers.BatchNormalization()
    conv2 = pruning(tf.keras.layers.Conv1D(filters=16,kernel_size=3,activation='linear',use_bias=True,padding='valid'))
    acti2 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn2 = tf.keras.layers.BatchNormalization()
    conv3 = pruning(tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=4,activation='linear',use_bias=True,padding='valid'))
    acti3 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn3 = tf.keras.layers.BatchNormalization()
    conv4 = pruning(tf.keras.layers.Conv1D(filters=32,kernel_size=3,activation='linear',use_bias=True,padding='valid'))
    acti4 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn4 = tf.keras.layers.BatchNormalization()
    conv5 = pruning(tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=4,activation='linear',use_bias=True,padding='valid'))
    acti5 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn5 = tf.keras.layers.BatchNormalization()
    conv6 = pruning(tf.keras.layers.Conv1D(filters=100,kernel_size=5,activation='linear',use_bias=True,padding='valid'))
    acti6 = tf.keras.layers.LeakyReLU(alpha=0.3)
    bn6 = tf.keras.layers.BatchNormalization()
    
    gap1=tf.keras.layers.GlobalAveragePooling1D()
    dense1 = tf.keras.layers.Dense(units=classes,activation='linear',use_bias=True)
    acti7 = tf.keras.layers.Activation(activation='softmax')

    
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
    outputs=acti7(y)

    
    return tf.keras.Model(inputs={"image":inputs}, outputs={"label":outputs,"output":y,"feature":x},name="PrunedTemporalCNN1D")
