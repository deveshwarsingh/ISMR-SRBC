#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:05:45 2022

@author: Deveshwar Singh
"""

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D,  AveragePooling2D
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

def autoencoder_decoder(self,input_shape):
    input_shape = self.input_shape

    input_layer = Input(shape=input_shape,dtype='float32')
    
    activity_regularizer = tf.keras.regularizers.l1(6.417239985918604e-11)
    activation = tf.keras.layers.Activation('sigmoid')
    initialiser = "he_normal"
    
    l1 = Conv2D(128, kernel_size = (3,3), activation=activation,padding= "same",kernel_initializer=initialiser,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(input_layer)

    l2 = Conv2D(128, kernel_size = (3,3), activation=activation,padding= "same",kernel_initializer=initialiser,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l1)

    l3 = AveragePooling2D(padding="same")(l2)

    l4 = Conv2D(256,kernel_size = (3,3), activation=activation,padding="same",kernel_initializer=initialiser,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l3)

    l5 = Conv2D(256,kernel_size = (3,3), activation=activation,padding= "same",kernel_initializer=initialiser,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l4)

    l6 = AveragePooling2D(padding='same')(l5)

    l7 = Conv2D(512, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l6)

    l8 = UpSampling2D()(l7)
    l9 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l8)
    l10 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l9)

    l11 = tf.keras.layers.add([l10, l5])

    l12 = UpSampling2D()(l11)
    l13 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l12)
    l14 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l13)

    l15 = tf.keras.layers.add([l14, l2])
    l16 = UpSampling2D()(l15)

    #l17 = tf.keras.layers.add([l16,l01])
    l18 = UpSampling2D()(l16)
    decoded_image =  Conv2D(1, (3, 3), padding='same', kernel_initializer=initialiser, activation=activation,bias_initializer=initialiser,activity_regularizer=activity_regularizer)(l18)

    model = Model(inputs=(input_layer), outputs=decoded_image)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.009740970406057836)
    
    loss = "mse"
    
    model.compile(optimizer=optimizer, loss=loss,metrics = [IOA,PSNR,COR])
    return model 