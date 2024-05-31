__all__ = ['DigitNet','PrunedDigitNet','SmallDigitNet','SmallPrunedDigitNet']

import tensorflow as tf
import os,sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from ..layers.pruning import pruning


###############################################################################################################################################
#DIGITNET : network of "ROBUST AND GENERALIZABLE VISUAL REPRESENTATION LEARNING VIA RANDOM CONVOLUTIONS" for the digits benchmark
#Normal version and Pruned version (all layers ar pruned except the last layer)
###############################################################################################################################################
def DigitNet(input_shape=(32,32,3),output_units=10):
  
    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1=tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
    mp1=tf.keras.layers.MaxPooling2D((2, 2))
    conv2=tf.keras.layers.Conv2D(128, (5, 5), activation='relu')
    mp2=tf.keras.layers.MaxPooling2D((2, 2))

    flatt=tf.keras.layers.Flatten()
    fc1=tf.keras.layers.Dense(1024, activation='relu')
    fc2=tf.keras.layers.Dense(1024, activation='relu')
    fc3=tf.keras.layers.Dense(output_units, activation='linear')
    acti1 = tf.keras.layers.Activation(activation='softmax')
        
    x=conv1(inputs)
    x=mp1(x)
    x=conv2(x)
    x=mp2(x)  
    x=flatt(x)
    y=fc1(x)
    y=fc2(y)
    y=fc3(y)
    outputs=acti1(y)
    
    return tf.keras.Model(inputs={"image":inputs}, outputs={"label":outputs,"output":y,"feature":x},name="DigitNet")
    
def PrunedDigitNet(input_shape=(32,32,3),output_units=10):
 
    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1=pruning(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    mp1=tf.keras.layers.MaxPooling2D((2, 2))
    conv2=pruning(tf.keras.layers.Conv2D(128, (5, 5), activation='relu'))
    mp2=tf.keras.layers.MaxPooling2D((2, 2))
    flatt=tf.keras.layers.Flatten()
    fc1=pruning(tf.keras.layers.Dense(1024, activation='relu'))
    fc2=pruning(tf.keras.layers.Dense(1024, activation='relu'))
    fc3=pruning(tf.keras.layers.Dense(output_units, activation='linear'),last=True)
    acti1 = tf.keras.layers.Activation(activation='softmax')
        
    x=conv1(inputs)
    x=mp1(x)
    x=conv2(x)
    x=mp2(x)
    y=flatt(x)
    y=fc1(y)
    y=fc2(y)
    y=fc3(y)
    outputs=acti1(y)
    
    return tf.keras.Model(inputs={"image":inputs}, outputs={"label":outputs,"output":y,"feature":x},name="PrunedDigitNet")
    
 
###############################################################################################################################################
#SMALLDIGITNET : network of "ROBUST AND GENERALIZABLE VISUAL REPRESENTATION LEARNING VIA RANDOM CONVOLUTIONS" scaled to have a number of
#parameters equivalent to Resnet20
#Normal version and Pruned version (all layers ar pruned except the last layer)
###############################################################################################################################################
    
def SmallDigitNet(input_shape=(32,32,3),output_units=10):

    inputs = tf.keras.Input(shape=input_shape)
    conv1=tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
    mp1=tf.keras.layers.MaxPooling2D((2, 2))
    conv2=tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
    mp2=tf.keras.layers.MaxPooling2D((2, 2))

    flatt=tf.keras.layers.Flatten()
    fc1=tf.keras.layers.Dense(128, activation='relu')
    fc2=tf.keras.layers.Dense(128, activation='relu')
    fc3=tf.keras.layers.Dense(output_units, activation='linear')
    acti1 = tf.keras.layers.Activation(activation='softmax')
        
    x=conv1(inputs)
    x=mp1(x)
    x=conv2(x)
    x=mp2(x)  
    x=flatt(x)
    y=fc1(x)
    y=fc2(y)
    y=fc3(y)
    outputs=acti1(y)
    
    return tf.keras.Model(inputs={"image":inputs}, outputs={"label":outputs,"output":y,"feature":x},name="SmallDigitNet")
    
def SmallPrunedDigitNet(input_shape=(32,32,3),output_units=10):

    inputs = tf.keras.Input(shape=input_shape, name="input")
    conv1=pruning(tf.keras.layers.Conv2D(32, (5, 5), activation='relu'))
    mp1=tf.keras.layers.MaxPooling2D((2, 2))
    conv2=pruning(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    mp2=tf.keras.layers.MaxPooling2D((2, 2))
    flatt=tf.keras.layers.Flatten()
    fc1=pruning(tf.keras.layers.Dense(128, activation='relu'))
    fc2=pruning(tf.keras.layers.Dense(128, activation='relu'))
    fc3=pruning(tf.keras.layers.Dense(output_units, activation='linear'),last=True)
    acti1 = tf.keras.layers.Activation(activation='softmax')
        
    x=conv1(inputs)
    x=mp1(x)
    x=conv2(x)
    x=mp2(x)
    y=flatt(x)
    y=fc1(y)
    y=fc2(y)
    y=fc3(y)
    outputs=acti1(y)
    
    return tf.keras.Model(inputs={"image":inputs}, outputs={"label":outputs,"output":y,"feature":x},name="SmallPrunedDigitNet")
