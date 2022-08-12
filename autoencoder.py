#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:42:08 2022

@author: dsingh24
"""
import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras.backend as k
from sklearn.base import TransformerMixin
import collections
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Lambda, MaxPool2D
from tensorflow.keras.models import Model
#%% Functions

def IOA(y_true, y_pred):
    #x = K.sum(K.square(y_true-y_pred))
    #y = K.sum(K.square(K.abs(y_pred-K.mean(y_true))+K.abs(y_true-K.mean(y_true))))
    ioa = 1 -(K.sum((y_true-y_pred)**2))/(K.sum((K.abs(y_pred-K.mean(y_true))+K.abs(y_true-K.mean(y_true)))**2))
    #ioa = 1 - (x/y)
    return ioa

def SSIM(y_true, y_pred):
    ssim = tf.image.ssim(y_pred,y_true, 1.0)
    
    return ssim

def l1(y_true, y_pred):
    """Calculate the L1 loss used in all loss calculations"""
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
        
def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""
    
    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
    
    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]
    
    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    gram = K.batch_dot(features, features, axes=2)
    
    # Normalize with channels, height and width
    gram = gram /  K.cast(C * H * W, x.dtype)
    
    return gram

def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    The equation is:
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    
    Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
    two values (4.75) as MAX_I        
    """        
    #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
    return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

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
netcdf_features_train = features_dir + "input_best_8.nc"

# filter used as label inputs for model training
netcdf_labels_train = labels_dir + "output_best_all.nc"



# features = ["evspsbl", "hfls", "hfss", "hurs", "pr","ps","rsds",
#             "ta200","ta850","ua200","ua850","va850","va200",
#             "zg500","tas"]

features = ["hurs", "pr","ps",
            "ua200","ua850","va850","va200",
            "tas"]
labels = ["APCP-sfc"]

train_x, train_y = extract_features_labels(netcdf_features=netcdf_features_train,
                                           netcdf_labels=netcdf_labels_train,
                                           feature_vars=features,
                                           label_vars= labels,
                                           level=0)

#train_x = train_x[:-1,:,:]

scaled_train_x = Scaler(features).transform(train_x)

time_stamps = scaled_train_x.shape[0]
variables = scaled_train_x.shape[-1]  
del  train_x
#%% VGG16
img_rows=512
img_cols=512
mean = [0.485, 0.456, 0.406] # Scaling for VGG input
std = [0.229, 0.224, 0.225] # Scaling for VGG input
vgg_layers = [6, 10, 13] # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
def build_vgg():
    """
    Load pre-trained VGG16 from keras applications
    Extract features to be used in loss function from last conv layer, see architecture at:
    https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
    """        
        
    # Input image to extract features from
    img = tf.keras.layers.Input(shape=(img_rows, img_cols, 3))

    # Mean center and rescale by variance as in PyTorch
    processed = Lambda(lambda x: (x-mean) / std)(img)
    
                
    # Get the vgg network from Keras applications
    vgg = VGG16(weights="imagenet", include_top=False)
   

    # Output the first three pooling layers
    vgg.outputs = [vgg.layers[i].output for i in vgg_layers]        
    
    # Create model and compile
    loss_model = Model(inputs=img, outputs=vgg[processed].outputs)
    loss_model.trainable = False
    loss_model.compile(loss='mse', optimizer='adam')

    return loss_model
vgg=build_vgg()
#%% Loss
def perceptual_style_loss(y_true, y_pred):
    vgg_out = vgg(y_pred)
    vgg_gt = vgg(y_true)
    # Compute loss components
    l3 = loss_perceptual(vgg_out, vgg_gt)
    l4 = loss_style(vgg_out, vgg_gt)

    # Return loss function
    return l3 + l4

    return perceptual_style_loss


def loss_perceptual(vgg_out, vgg_gt): 
    """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
    loss = 0
    for o,  g in zip(vgg_out,  vgg_gt):
        loss += l1(o, g) 
    return loss
    
def loss_style( output, vgg_gt):
    """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
    loss = 0
    for o, g in zip(output, vgg_gt):
        loss += l1(gram_matrix(o), gram_matrix(g))
    return loss

#%% Model architecture and development


def auto_encoder():
    input_shape = scaled_train_x.shape[1:]

    input_layer = Input(shape=input_shape,dtype='float32')


    l1 = Conv2D(128, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_layer)

    l2 = Conv2D(128, kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)

    l3 = MaxPool2D(padding="same")(l2)

    l4 = Conv2D(256,kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)

    l5 = Conv2D(256,kernel_size = (3,3), activation="relu",padding= "same",kernel_initializer='he_uniform',activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)

    l6 = MaxPool2D(padding='same')(l5)

    l7 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

    l8 = UpSampling2D()(l7)
    l9 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
    l10 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

    l11 = tf.keras.layers.add([l10, l5])

    l12 = UpSampling2D()(l11)
    l13 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
    l14 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)

    l15 = tf.keras.layers.add([l14, l2])
    l16 = UpSampling2D()(l15)

    #l17 = tf.keras.layers.add([l16,l01])
    l18 = UpSampling2D()(l16)
    decoded_image =  Conv2D(1, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l18)

    auto_encoder = Model(inputs=(input_layer), outputs=decoded_image)

    auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss,metrics = [IOA,PSNR,"mse"])
    return auto_encoder

auto_encoder = auto_encoder()
auto_encoder.summary()

        
#%% Training model

trainX, valX, trainY, valY = train_test_split(scaled_train_x, train_y, 
                                                        test_size=0.2, 
                                                        random_state=0,shuffle=True)
datagen = ImageDataGenerator() 

train = datagen.flow(
    trainX,
    trainY,
    batch_size=16, 
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
    batch_size=16, 
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)


key = locals().keys()

save_dir = "/path/"
checkpoint = ModelCheckpoint(save_dir+"BC_pr_test.h5", monitor='val_loss', mode='min', verbose=1, 
                             save_best_only=True, save_weights_only=False) 


### Training time logger 

# class TimeLogger(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.times = []
#     def on_epoch_begin(self, epoch, logs={}):
#         self.epoch_time_start = time.time()
#     def on_epoch_end(self, epoch, logs={}):
#         self.times.append(time.time() - self.epoch_time_start)
        
# time_callback = TimeLogger()

# Create CSV file with epoch number and training and validation scores        
csv_logger = CSVLogger(save_dir+"BC_pr_model_history.csv",append=True)

### Training PCNN Model ###
history = auto_encoder.fit_generator(train,
    steps_per_epoch=len(scaled_train_x)//16,  
    validation_data=val,
    validation_steps=len(train_y)//16, 
    epochs=10,
    verbose=1, shuffle=True,
    #workers=1, 
    use_multiprocessing=False,
    callbacks=[checkpoint, csv_logger])
