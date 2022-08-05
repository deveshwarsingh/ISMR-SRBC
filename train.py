#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:42:08 2022

@author: dsingh24
"""
# example of an encoder-decoder generator for the cyclegan
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, InputLayer
#from Extract_Features_Label import extract_features_labels
#from Scaler import Scaler
#from Extract_Data_Array import extract_data_array
import cv2
import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from sklearn.base import TransformerMixin
import collections
#%%
def extract_data_array(dataset,
                       variables,
                       lev):
    
    # allocate the array
    arr = np.empty(shape=[dataset.time.size, 
                          dataset.lat.size, 
                          dataset.lon.size, 
                          len(variables)],
                   dtype=np.float64)
    
    # for each variable we'll extract the values 
    for var_index, var in enumerate(variables):
        
        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            values = dataset[var].values[:, lev, :, :]
        elif dimensions == ('time', 'lat', 'lon'):
            values = dataset[var].values[:, :, :]
        else:
            raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))
        
        # add the values into the array at the variable's position
        arr[:, :, :, var_index] = values
    
    return arr

def extract_features_labels(netcdf_features, 
                            netcdf_labels,
                            feature_vars,
                            label_vars,
                            level=0):
    """
    Extracts feature and label data from specified NetCDF files for a single level as numpy arrays.
    
    The feature and label NetCDFs are expected to have matching time, level, lat, and lon coordinate variables.
    
    Returns two arrays: the first for features and the second for labels. Arrays will have shape (time, lat, lon, var),
    where var is the number of feature or label variables. For example if the dimensions of feature data variables in 
    the NetCDF is (time: 360, lev: 26, lat: 120, lon: 180) and the features specified are ["T", "U"] then the resulting
    features array will have shape (360, 120, 180, 2), with the first feature variable "T" corresponding to array[:, :, :, 0]
    and the second feature variable "U" corresponding to array[:, :, :, 1].
    
    :param netdcf_features: one or more NetCDF files containing feature variables, can be single file or list
    :param netdcf_features: one or more NetCDF files containing label variables, can be single file or list
    :param feature_vars: list of feature variable names to be extracted from the features NetCDF
    :param label_vars: list of label variable names to be extracted from the labels NetCDF
    :param level: index of the level to be extracted (all times/lats/lons at this level for each feature/label variable)
    :return: two 4-D numpy arrays, the first for features and the second for labels
    """
    
    # open the features (flows) and labels (tendencies) as xarray DataSets
    ds_features = xr.open_mfdataset(paths=netcdf_features)
    ds_labels = xr.open_mfdataset(paths=netcdf_labels)
    
    # confirm that we have datasets that match on the time, lev, lat, and lon dimension/coordinate
    # if np.any(ds_features.variables['time'].values != ds_labels.variables['time'].values):
    #     raise ValueError('Non-matching time values between feature and label datasets')
    # if np.any(ds_features.variables['lev'].values != ds_labels.variables['lev'].values):
    #     raise ValueError('Non-matching level values between feature and label datasets')
    # if np.any(ds_features.variables['lat'].values != ds_labels.variables['lat'].values):
    #     raise ValueError('Non-matching lat values between feature and label datasets')
    # if np.any(ds_features.variables['lon'].values != ds_labels.variables['lon'].values):
    #     raise ValueError('Non-matching lon values between feature and label datasets')
    
    # extract feature and label arrays at the specified level
    array_features = extract_data_array(ds_features, feature_vars, level)
    array_labels = extract_data_array(ds_labels, label_vars, level)
    
    return array_features, array_labels

class Scaler(TransformerMixin):
    
    def __init__(self, features):
        
        # initialize an ordered dict to store scalers for each feature
        self.scalers = collections.OrderedDict().fromkeys(features, MinMaxScaler(feature_range=(0, 1)))
    
    def transform(self, values):
        """
        Transforms a 4-D array of values, expected to have shape:
        (times, lats, lons, vars).
        
        :param values:
        :return:
        """
        
        # make new arrays to contain the scaled values we'll return
        scaled_features = np.empty(shape=values.shape)
        
        # data is 4-D with shape (times, lats, lons, vars), scalers can only
        # work on 2-D arrays, so for each variable we scale the corresponding
        # 3-D array of values after flattening it, then reshape back into
        # the original shape
        var_ix = 0
        for variable, scaler in self.scalers.items():
            variable = values[:, :, :, var_ix].flatten().reshape(-1, 1)
            scaled_feature = scaler.fit_transform(variable)
            reshaped_scaled_feature = np.reshape(scaled_feature,
                                                 newshape=(values.shape[0],
                                                           values.shape[1],
                                                           values.shape[2]))
            scaled_features[:, :, :, var_ix] = reshaped_scaled_feature
            var_ix += 1
        
        # return the scaled values (the scalers have been fitted to the data)
        return scaled_features
    
    def fit(self, x=None):
        
        return self


#Defining CUSTOM LOSS Function
def customLoss1(o,p):
    ioa = 1 -(k.sum((o-p)**2))/(k.sum((k.abs(p-k.mean(o))+k.abs(o-k.mean(o)))**2))
    return (-ioa)

def IOA(y_true, y_pred):
    #x = K.sum(K.square(y_true-y_pred))
    #y = K.sum(K.square(K.abs(y_pred-K.mean(y_true))+K.abs(y_true-K.mean(y_true))))
    ioa = 1 -(k.sum((y_true-y_pred)**2))/(k.sum((k.abs(y_pred-k.mean(y_true))+k.abs(y_true-k.mean(y_true)))**2))
    #ioa = 1 - (x/y)
    return ioa
#%% Reading the data and necessary preprocessing
features_dir = "/project/ychoi/dsingh/biasCorrectionISMR/Data/data_preparation/py_test/input/"
labels_dir = "/project/ychoi/dsingh/biasCorrectionISMR/Data/data_preparation/py_test/output/"

#files used as feature inputs for model training
netcdf_features_train = features_dir + "input_train_test.nc"

# filter used as label inputs for model training
netcdf_labels_train = labels_dir + "output_train_test.nc"



features = ["evspsbl", "hfls", "hfss", "hurs", "pr","ps","rsds",
            "ta200","ta850","ua200","ua850","va850","va200",
            "zg500","tas"]
labels = ["APCP-sfc"]

train_x, train_y = extract_features_labels(netcdf_features=netcdf_features_train,
                                           netcdf_labels=netcdf_labels_train,
                                           feature_vars=features,
                                           label_vars= labels,
                                           level=0)

train_x = train_x[:-1,:,:]

scaled_train_x = Scaler(features).transform(train_x)

time_stamps = scaled_train_x.shape[0]
variables = scaled_train_x.shape[-1]

# loop to reshape input of the model
reshaped_train_x = np.empty([9861,128,128,15])*np.nan
for time in range(time_stamps):
    for var in range(variables):
        reshaped_train_x[time,:,:,var] = cv2.resize(scaled_train_x[time,:,:,var],(128,128),interpolation= cv2.INTER_CUBIC)
        print(time,var)
     

reshaped_train_y = np.empty([9861,512,512])*np.nan
for time in range(time_stamps):
    reshaped_train_y[time,:,:] = cv2.resize(train_y[time,:,:,:],(512,512),interpolation= cv2.INTER_CUBIC)
    print(time)
    
del scaled_train_x, train_x, train_y
#%% Model architecture and development


input_shape = reshaped_train_x.shape[1:]

output_shape = reshaped_train_y.shape[1:]

input_layer = tf.keras.layers.Input(shape=input_shape,dtype='float32')

# l01 = tf.keras.layers.Conv2D(256, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_layer)

# l02 = tf.keras.layers.Conv2D(256, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l01)

# l03 = tf.keras.layers.MaxPool2D(padding="same")(l02)

l1 = tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_layer)

l2 = tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)

l3 = tf.keras.layers.MaxPool2D(padding="same")(l2)

l4 = tf.keras.layers.Conv2D(256,kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)

l5 = tf.keras.layers.Conv2D(256,kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)

l6 = tf.keras.layers.MaxPool2D(padding='same')(l5)

l7 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

l8 = tf.keras.layers.UpSampling2D()(l7)
l9 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
l10 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

l11 = tf.keras.layers.add([l10, l5])

l12 = tf.keras.layers.UpSampling2D()(l11)
l13 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
l14 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)

l15 = tf.keras.layers.add([l14, l2])
l16 = tf.keras.layers.UpSampling2D()(l15)

#l17 = tf.keras.layers.add([l16,l01])
l18 = tf.keras.layers.UpSampling2D()(l16)
decoded_image =  tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l18)

auto_encoder = tf.keras.models.Model(inputs=(input_layer), outputs=decoded_image)

auto_encoder.compile(optimizer='Adam', loss='mse',metrics = [IOA])

auto_encoder.summary()
#%% Training model

trainX, valX, trainY, valY = train_test_split(reshaped_train_x, reshaped_train_y, 
                                                        test_size=0.2, 
                                                        random_state=0,shuffle=True)
datagen = ImageDataGenerator() 

train = datagen.flow(
    trainX,
    trainY,
    batch_size=4, 
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)

val = datagen.flow(
    valX,
    valY,
    batch_size=4, 
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)

key = locals().keys()

save_dir = "path"
checkpoint = ModelCheckpoint(save_dir+"BC_pr_test.h5", monitor='val_loss', mode='min', verbose=1, 
                             save_best_only=True, save_weights_only=False) 


### Training time logger 

class TimeLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
time_callback = TimeLogger()

# Create CSV file with epoch number and training and validation scores        
csv_logger = CSVLogger(save_dir+"BC_pr_model_history.csv",append=True)

### Training PCNN Model ###
history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//8,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//8, 
    epochs=30,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])

history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//64,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//64, 
    epochs=10,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])

history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//32,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//32, 
    epochs=10,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])


history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//8,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//8, 
    epochs=15,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])
#%% ResNet Test
def relu_batch_norm(inputs):
    relu = layers.ReLU()(inputs)
    batch_norm = layers.BatchNormalization()(relu)
    return batch_norm
def conv_layer(x, filters: int, kernel_size: int = 3, strides: int = 1):
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,
               padding="same")(x)
    y = relu_batch_norm(y)
    return y
def residual_block(x, filters: int, kernel_size: int = 3, strides: int = 1):
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,
               padding="same")(x)
    y = relu_batch_norm(y)
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,
               padding="same")(y)
    y = relu_batch_norm(y)
    
    out = layers.Add()([x, y])
    out = relu_batch_norm(out)
    
    return out
def conv_layer_transpose(x, filters: int, kernel_size: int = 4, strides: int = 2):
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides=strides,
               filters=filters,
               padding="same")(x)
    y = relu_batch_norm(y)
    return y
def create_res_net():
    
    inputs = tf.keras.layers.Input((128, 128, 15))
    
    t = layers.BatchNormalization()(inputs)
    t = conv_layer(t, kernel_size=9,
           strides=1,
           filters=64)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=64)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=128)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=256)
    t = conv_layer(t, kernel_size=4,
           strides=2,
           filters=512)
    
    for i in range(0,3):
        t = residual_block(t, kernel_size=3,
               strides=1,
               filters=512)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=512)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=256)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=128)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64)
    t = conv_layer_transpose(t, kernel_size=4,
           strides=2,
           filters=64)
    
    t = layers.Conv2DTranspose(kernel_size=4,
           strides= 1,
           filters=1,
           padding="same")(t)
    t = layers.Activation(activation='tanh')(t)
    
    output = (t)
        
    model = Model(inputs, output)

    return model
model = create_res_net()
model.compile(optimizer='Adam', loss='mse',metrics = [IOA])
model.summary()
#%%
trainX, valX, trainY, valY = train_test_split(reshaped_train_x, reshaped_train_y, 
                                                        test_size=0.2, 
                                                        random_state=0,shuffle=True)
datagen = ImageDataGenerator() 

train = datagen.flow(
    trainX,
    trainY,
    batch_size=4, 
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)

val = datagen.flow(
    valX,
    valY,
    batch_size=4, 
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)

key = locals().keys()

save_dir = "path"
checkpoint = ModelCheckpoint(save_dir+"BC_pr_resnet_test.h5", monitor='val_loss', mode='min', verbose=1, 
                             save_best_only=True, save_weights_only=False) 


### Training time logger 

class TimeLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
time_callback = TimeLogger()

# Create CSV file with epoch number and training and validation scores        
csv_logger = CSVLogger(save_dir+"BC_pr_resnet_model_history.csv",append=True)

### Training PCNN Model ###
history = model.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//8,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//8, 
    epochs=30,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])

history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//64,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//64, 
    epochs=10,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])

history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//32,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//32, 
    epochs=10,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])


history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(reshaped_train_x)//8,  
    validation_data=val,
    validation_steps=len(reshaped_train_y)//8, 
    epochs=15,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=True,
    callbacks=[checkpoint, csv_logger])
