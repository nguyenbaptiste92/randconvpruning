#File to test the RandConvModule in rand_conv.py

import tensorflow as tf
import sys
import os

from ..rand_conv import *

randconv_printing.assign(True)

inputs=tf.random.uniform(shape=[1,32,32,3], minval=0.0, maxval=1.0, dtype=tf.float32)

def randconv2d_model(input_shape=(32,32,3)):

    inputs = tf.keras.Input(shape=input_shape, name="input_source")
    conv=RandConvModule(3,[1,2,3],identity_prob=0.5,mixing=True,mode="2d")
    
    y=conv(inputs)
    return tf.keras.Model(inputs={"input_source":inputs}, outputs={"pred":y},name="randconv_model")
    
model=randconv2d_model()
model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),loss=tf.keras.losses.CategoricalCrossentropy())
model.summary()


pred1=model(inputs)["pred"]
pred2=model(inputs)["pred"]

tf.debugging.assert_equal(pred1.shape[:-1], inputs.shape[:-1], message="The input and the output should have the same image size")
tf.debugging.assert_equal(pred2.shape[:-1], inputs.shape[:-1], message="The input and the output should have the same image size")
tf.debugging.assert_none_equal(pred1,pred2, message="The two outputs should be different due to the randomization of the RandConvModule")
print("RandConvModule2d fonctionne bien.")

inputs=tf.random.uniform(shape=[1,150,3], minval=0.0, maxval=1.0, dtype=tf.float32)

def randconv1d_model(input_shape=(150,3)):

    inputs = tf.keras.Input(shape=input_shape, name="input_source")
    conv=RandConvModule(3,[1,2,3],identity_prob=0.5,mixing=True,mode="1d")
    
    y=conv(inputs)
    return tf.keras.Model(inputs={"input_source":inputs}, outputs={"pred":y},name="randconv_model")
    
model=randconv1d_model()
model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),loss=tf.keras.losses.CategoricalCrossentropy())
model.summary()


pred1=model(inputs)["pred"]
pred2=model(inputs)["pred"]

tf.debugging.assert_equal(pred1.shape[:-1], inputs.shape[:-1], message="The input and the output should have the same image size")
tf.debugging.assert_equal(pred2.shape[:-1], inputs.shape[:-1], message="The input and the output should have the same image size")
tf.debugging.assert_none_equal(pred1,pred2, message="The two outputs should be different due to the randomization of the RandConvModule")
print("RandConvModule1d fonctionne bien.")




