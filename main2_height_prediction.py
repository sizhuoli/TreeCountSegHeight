#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:56:33 2021

@author: sizhuo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

import imageio
import os
import time
import rasterio.warp             
from functools import reduce
from tensorflow.keras.models import load_model

from core2.UNet_attention_CHM import UNet

from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.frame_info_CHM import FrameInfo
from core2.dataset_generator_CHM import DataGenerator
from core2.split_frames import split_dataset
from core2.visualize import display_images

import json
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# %matplotlib inline
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from tqdm import tqdm
import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from config import UNetTraining_CHM
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

print(tf.config.list_physical_devices('GPU'))
config = UNetTraining_CHM.Configuration()

# Loading data (train and val): read all images/frames into memory
frames = []
all_files = os.listdir(config.base_dir)
all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
for i, fn in tqdm(enumerate(all_files_c1)):
    comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
    if config.single_raster or not config.aux_data:
        for c in range(config.image_channels-1):
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
    else: # multi raster
        for c in range(config.image_channels-2): 
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
        comb_img = np.append(comb_img, chm, axis = 0)

    comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
    chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn).replace("png", "tif"))).read()
    chm = np.squeeze(chm)
    f = FrameInfo(comb_img, chm)
    frames.append(f)

frames_val = []
all_files = os.listdir(config.base_dir_val)
all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]

for i, fn in tqdm(enumerate(all_files_c1)):
    comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
    if config.single_raster or not config.aux_data:
        for c in range(config.image_channels-1):
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
    else: # multi raster
        for c in range(config.image_channels-2): 
                #loop through raw channels
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
        comb_img = np.append(comb_img, chm, axis = 0)

    comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
    chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn).replace("png", "tif"))).read()
    chm = np.squeeze(chm)
    f = FrameInfo(comb_img, chm)
    frames_val.append(f)

training_frames = list(range(len(frames)))
validation_frames = list(range(len(frames_val)))
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, augmenter = "iaa").random_generator(config.BATCH_SIZE, normalize = config.normalize, maxmin_norm=config.maxmin_norm, gb_norm = config.gb_norm, gb_norm_FI = config.gb_norm_FI, robust_scale = config.robust_scale)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames_val, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize, maxmin_norm=config.maxmin_norm, gb_norm = config.gb_norm, gb_norm_FI = config.gb_norm_FI, robust_scale = config.robust_scale)


# Defining model config
def W_mae(y_true, y_pred, clip_delta=10, wei = 5):
    # weight higher for large heights
    cond  = y_true < clip_delta
    loss_small = K.mean(K.abs(y_pred - y_true)) # no spatial weights
    loss_large = K.mean(wei*K.abs(y_pred - y_true))
    return tf.where(cond, loss_small, loss_large)

OPTIMIZER = adam
OPTIMIZER_NAME = 'Adam'
LOSS_NAME = 'Wmae'
timestr = time.strftime("%Y%m%d-%H%M")
chs = reduce(lambda a,b: a+str(b), config.extracted_filenames, '')
if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)
model_path = os.path.join(config.model_path,
                          'trees_RGB2CHM_{}_{}_{}_{}_{}_{}_frames_{}.h5'.format(
                              timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], 
                              len(all_files_c1), config.model_name))

# Ini model
model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
model.compile(optimizer=OPTIMIZER, loss=W_mae, 
              metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
print(model.summary())

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min', save_weights_only = False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                    patience=4, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)
log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,
                                                                       OPTIMIZER_NAME,LOSS_NAME,chs, 
                                                                       config.input_shape[0], 
                                                                       len(all_files_c1)))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, 
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                          embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
callbacks_list = [checkpoint, tensorboard] 

tf.config.run_functions_eagerly(True)

# Training
loss_history = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          epochs=config.NB_EPOCHS, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]

