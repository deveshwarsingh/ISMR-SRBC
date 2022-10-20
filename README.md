# ISMR-SRBC
A library for super-resolution and bias-correction (SRBC) of Indian Summer Monsoon Rainfall (ISMR). 

This repository is a tool for SRBC of climatic simulations obtained from CORDEX-SA climatic simulations for the Representative Concentration Pathways (RCP) viz. RCP 2.6, 4.5 and 8.0. For accurate and reliable projections of ISMR, climatic simulations obtained current state-of-the-art dynamical numerical models such as GCMs and RCMs suffer from three challenges viz.
1. Systematic biases associated with GCMs and RCMs due to several issues such as parameterization etc.
2. Coarse resolution of climatic simulations which is insuffient for regional-to-local scale climate policy making.
3. Inaccurate representation of extreme rainfall events in the climatic simulations of the precipitation.

Therefore, in order to address these challenges, this repo, uses Deep-learning algorithms such as AutoEncoder-Decoder (ACDC) and Residual Neural Networks (ResNet). Later on, several other algorithms for SRBC are also planned to be added. This repo uses python libraries such as Tensorflow, Keras, Sklearn, Xarray, Numpy and Pandas. 

These algorithms are trained using the Renalysis Datasets such as ERA5 and Indian Monsoon Dataset Assimilation and Analysis (IMDAA) along with CORDEX-SA climatic simulation of precipitation over the domain of South-Asia i.e. CORDEX-SA. Additionally, for bias-correction and reducing topographical information is also provided to the model in the form of Digital Elevation Model (DEM) obtained from Shuttle Radar Topography Mission (SRTM). 


The DL-ased SRBC algorithms are trained in two phases:
1. Pre-training: 
2. Transfer-learning

