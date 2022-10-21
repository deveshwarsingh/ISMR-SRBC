#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:15:51 2022

@author: dsingh24
"""
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import collections
import numpy as np
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