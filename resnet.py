#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:20:35 2022

@author: Deveshwar Singh
"""

import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as k

def IOA(y_true, y_pred):
    #x = K.sum(K.square(y_true-y_pred))
    #y = K.sum(K.square(K.abs(y_pred-K.mean(y_true))+K.abs(y_true-K.mean(y_true))))
    ioa = 1 -(k.sum((y_true-y_pred)**2))/(k.sum((k.abs(y_pred-k.mean(y_true))+k.abs(y_true-k.mean(y_true)))**2))
    #ioa = 1 - (x/y)
    return ioa

def COR(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = k.mean(x)
    my = k.mean(y)
    xm, ym = x - mx, y - my
    r_num = k.sum(tf.multiply(xm, ym))
    r_den = k.sqrt(tf.multiply(k.sum(k.square(xm)), k.sum(k.square(ym))))
    r = r_num / r_den
    r = k.maximum(k.minimum(r, 1.0), -1.0)
    return k.square(r)

def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    The equation is:
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    
    Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
    two values (4.75) as MAX_I        
    """        
    #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
    return - 10.0 * k.log(k.mean(k.square(y_pred - y_true))) / k.log(10.0)

def relu_batch_norm(inputs):
    relu = layers.ReLU()(inputs)
    batch_norm = layers.BatchNormalization()(relu)
    return batch_norm
def conv_layer(x, filters: int,kernel_initializer: str,activity_regularizer:float,  kernel_size: int = 3, strides: int = 1):
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,kernel_initializer= kernel_initializer,activity_regularizer=activity_regularizer,
               padding="same")(x)
    y = relu_batch_norm(y)
    return y
def residual_block(x, filters: int,kernel_initializer: str,activity_regularizer:float,  kernel_size: int = 3, strides: int = 1):
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,kernel_initializer= kernel_initializer,activity_regularizer=activity_regularizer,
               padding="same")(x)
    y = relu_batch_norm(y)
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,kernel_initializer= kernel_initializer,activity_regularizer=activity_regularizer,
               padding="same")(y)
    y = relu_batch_norm(y)
    
    out = layers.Add()([x, y])
    out = relu_batch_norm(out)
    
    return out
def conv_layer_transpose(x, filters: int,kernel_initializer: str,activity_regularizer:float,  kernel_size: int = 4, strides: int = 2):
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides=strides,
               filters=filters,kernel_initializer= kernel_initializer,activity_regularizer=activity_regularizer,
               padding="same")(x)
    y = relu_batch_norm(y)
    return y
def resnet(self, input_shape):
    
    self.input_shape = input_shape
    
    input_shape = input_shape

    inputs = Input(shape=input_shape,dtype='float32')
    
    activity_regularizer = tf.keras.regularizers.l1(4.5206720345029323e-13)
    activation = 'softplus'
    initialiser = "he_normal"
    
    t = layers.BatchNormalization()(inputs)
    t = conv_layer(t, kernel_size=9,
           strides=1,
           filters=64,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=64,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=128,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=256,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=512,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    
    for i in range(0,3):
        t = residual_block(t, kernel_size=3,
               strides=1,
               filters=512,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=512,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=256,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=128,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64,kernel_initializer=initialiser,activity_regularizer = activity_regularizer)
    
    t = layers.Conv2DTranspose(kernel_size=4,
           strides= 1,
           filters=1,
           padding="same")(t)
    t = layers.Activation(activation=activation)(t)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.007171796122571094)

    loss = "mse"
    output = (t) 
    model = Model(inputs, output)
    model.compile(optimizer=optimizer, loss=loss,metrics = [IOA,PSNR,COR])
    return model
