#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:56:33 2021

@author: rscph
"""

# UNet as regressor to map from NIRGB to CHM



# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import rasterio
# import imgaug as ia
# from imgaug import augmenters as iaa
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import imageio
# import os
# import time
# import rasterio.warp             # Reproject raster samples
# from functools import reduce
# from tensorflow.keras.models import load_model
# import tensorflow.keras.backend as K

# from core.UNet_attention_CHM import UNet
# from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
# from core.optimizers import adaDelta, adagrad, adam, nadam
# from core.frame_info_CHM import FrameInfo
# from core.dataset_generator_CHM import DataGenerator
# from core.split_frames import split_dataset
# from core.visualize import display_images

# import json
# from sklearn.model_selection import train_test_split
# from skimage.transform import resize

# # %matplotlib inline
# import matplotlib.pyplot as plt  # plotting tools
# import matplotlib.patches as patches
# from matplotlib.patches import Polygon

# import warnings                  # ignore annoying warnings
# warnings.filterwarnings("ignore")
# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)

# # %reload_ext autoreload
# # %autoreload 2
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# from config import UNetTraining_CHM
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard


# config = UNetTraining_CHM.Configuration()

# frames = []

# all_files = os.listdir(config.base_dir)
# # image channel 1
# all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
# print(all_files_c1)
    

# for i, fn in enumerate(all_files_c1):
#     # loop through rectangles
#     comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
#     if config.single_raster or not config.aux_data:
#         # print('If single raster or multi raster without aux')
#         for c in range(config.image_channels-1):
#             #loop through raw channels
#             comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
#     else: # multi raster
#         print('Multi raster with aux data')
#         for c in range(config.image_channels-1): 
#                 #loop through raw channels
#             comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
#         #for aux chm channel, upsample by 2 
#         # 0.4m resolution compared to 0.2m resolution
#         # chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
#         # #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
#         # # handle resolution diff: resize chm layer to the same shape as core image
#         # chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
#         # comb_img = np.append(comb_img, chm, axis = 0)

    
    
#     # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
#     # np.asarray(annotation_im)
#     # annotation = np.array(annotation_im)
#     # annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn))).read()
#     # annotation = np.squeeze(annotation)
#     # weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.weight_fn))).read()
#     # weight = np.squeeze(weight)
#     chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn))).read()
#     chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
#     chm = np.squeeze(chm)
#     center = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],'center'))).read()
#     center = np.squeeze(center)
#     comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
#     f = FrameInfo(comb_img, chm, center)
#     frames.append(f)



# training_frames = validation_frames = testing_frames  = list(range(len(frames)))

# # divide fixed train val test for 31 frames in v2
# # training_frames = [22, 2, 15, 27, 7, 18, 24, 19, 11, 3, 13, 12, 14, 4, 0, 30, 23, 9, 6, 21, 1, 25, 10, 8, 17]
# # validation_frames = [26, 20, 16, 29, 5, 28]
# # testing_frames = [6, 21, 1, 25, 10, 8, 17] # not used

# annotation_channels = config.input_label_channel
# train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, config.center_weights, augmenter = "iaa").random_generator(config.BATCH_SIZE, normalize = config.normalize)
# val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, config.center_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
# test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, config.center_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)


# for _ in range(1):
#     train_images, chms = next(train_generator)
#     print(train_images.shape, chms.shape)
#     chmm = chms[..., 0][..., np.newaxis]
#     center = chms[..., 1][..., np.newaxis]
#     print(np.unique(center))
#     # print(train_images.shape, chm.shape)
#     # print(type(train_images), type(chm))
#     # chm = chm[..., np.newaxis]
    
#     #overlay of annotation with boundary to check the accuracy
#     #8 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    
#     display_images(np.concatenate((train_images,chmm, center), axis = -1))
    


# def mse(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true))


# def weighted_mse(y_true, y_pred):
#     # y_pred = np.squeeze(y_pred)
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
#     # weights
#     y_weights = y_true[...,1]
#     y_weights = y_weights[...,np.newaxis]
#     # print(np.unique(y_weights))
#     sqr = K.square(y_pred - y_t)
#     # print(np.mean(sqr))
#     mul = sqr*y_weights
#     # print(np.mean(mul))
    
#     return K.mean(mul)
    
# def root_weighted_mse(y_true, y_pred):
#     return K.sqrt(weighted_mse(y_true, y_pred))
    
# # OPTIMIZER = adaDelta
# OPTIMIZER = adam
# # LOSS = 'mse'
# LOSS = 'weighted_mse'
# #Only for the name of the model in the very end
# # OPTIMIZER_NAME = 'AdaDelta'
# OPTIMIZER_NAME = 'Adam'
# LOSS_NAME = 'weighted_mse'

# # Declare the path to the final model
# # If you want to retrain an exising model then change the cell where model is declared. 
# # This path is for storing a model after training.

# timestr = time.strftime("%Y%m%d-%H%M")
# # chf = config.input_image_channel + config.input_label_channel
# # chs = reduce(lambda a,b: a+str(b), chf, '')
# # chs = reduce(lambda a,b: a+str(b), config.input_image_channel, '')
# chs = reduce(lambda a,b: a+str(b), config.extracted_filenames, '')


# if not os.path.exists(config.model_path):
#     os.makedirs(config.model_path)
# model_path = os.path.join(config.model_path,'trees_NIRGB2CHM_{}_{}_{}_{}_{}_{}_frames_unet_attention.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], len(all_files_c1)))


# model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
# # model.compile(optimizer=OPTIMIZER, loss='mse', metrics=tf.keras.metrics.RootMeanSquaredError())
# model.compile(optimizer=OPTIMIZER, loss=weighted_mse, metrics=root_weighted_mse)
# # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


# # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
#                              save_best_only=True, mode='min', save_weights_only = False)

# #reduceonplatea; It can be useful when using adam as optimizer
# #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
# #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
#                                    patience=4, verbose=1, mode='min',
#                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)

# #early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

# log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0], len(all_files_c1)))
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

# tf.config.run_functions_eagerly(True)

# loss_history = [model.fit(train_generator, 
#                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                          epochs=config.NB_EPOCHS, 
#                          validation_data=val_generator,
#                          validation_steps=config.VALID_IMG_COUNT,
#                          callbacks=callbacks_list,
#                          workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )]






# for i in range(1):
#     test_images, real_label = next(test_generator)
#     #5 images per row: pan, ndvi, label, weight, prediction
#     prediction = model.predict(test_images, steps=1)
#     print(prediction.shape)
#     # real_label = resize(real_label[:, :, :], (config.BATCH_SIZE, test_images.shape[1], test_images.shape[2]))
#     # prediction = resize(prediction[:, :, :], (config.BATCH_SIZE, test_images.shape[1], test_images.shape[2]))
#     display_images(np.concatenate((test_images, real_label, prediction), axis = -1))

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu


import tensorflow as tf
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import imageio
import os
import time
import rasterio.warp             # Reproject raster samples
from functools import reduce
from tensorflow.keras.models import load_model

from core.UNet_attention_CHM import UNet
from core.Xception_CHM import Xception


from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info_CHM import FrameInfo
from core.dataset_generator_CHM import DataGenerator
from core.split_frames import split_dataset
from core.visualize import display_images

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

print(tf.__version__)


print(tf.config.list_physical_devices('GPU'))

# Required configurations (including the input and output paths) are stored in a separate file (such as config/UNetTraining.py)
# Please provide required info in the file before continuing with this notebook. 
 

# In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder 
# Eg. from configLargeCluster import UNetTraining
config = UNetTraining_CHM.Configuration()



# split test 
# import random
# random.seed(2)
# testframes = random.sample(list(np.arange(63)), k = 13)


# Read all images/frames into memory
frames = []

all_files = os.listdir(config.base_dir)
# image channel 1
all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
print(all_files_c1)


for i, fn in tqdm(enumerate(all_files_c1)):
    # loop through rectangles
    comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
    if config.single_raster or not config.aux_data:
        # print('If single raster or multi raster without aux')
        for c in range(config.image_channels-1):
            #loop through raw channels
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
    else: # multi raster
        print('Multi raster with aux data')
        for c in range(config.image_channels-2): 
                #loop through raw channels
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
        #for aux chm channel, upsample by 2 
        # 0.4m resolution compared to 0.2m resolution
        chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
        #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
        # handle resolution diff: resize chm layer to the same shape as core image
        # chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
        comb_img = np.append(comb_img, chm, axis = 0)

    
    comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
    # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
    # np.asarray(annotation_im)
    # annotation = np.array(annotation_im)
    # annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn))).read()
    # annotation = np.squeeze(annotation)
    # weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.weight_fn))).read()
    # weight = np.squeeze(weight)
    chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn).replace("png", "tif"))).read()
    chm = np.squeeze(chm)
    f = FrameInfo(comb_img, chm)
    frames.append(f)


#####
# separate val set
frames_val = []

all_files = os.listdir(config.base_dir_val)
# image channel 1
all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
print(all_files_c1)


for i, fn in tqdm(enumerate(all_files_c1)):
    # loop through rectangles
    comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
    if config.single_raster or not config.aux_data:
        # print('If single raster or multi raster without aux')
        for c in range(config.image_channels-1):
            #loop through raw channels
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
    else: # multi raster
        print('Multi raster with aux data')
        for c in range(config.image_channels-2): 
                #loop through raw channels
            comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
        #for aux chm channel, upsample by 2 
        # 0.4m resolution compared to 0.2m resolution
        chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
        #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
        # handle resolution diff: resize chm layer to the same shape as core image
        # chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
        comb_img = np.append(comb_img, chm, axis = 0)

    
    comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
    # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
    # np.asarray(annotation_im)
    # annotation = np.array(annotation_im)
    # annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn))).read()
    # annotation = np.squeeze(annotation)
    # weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.weight_fn))).read()
    # weight = np.squeeze(weight)
    chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn).replace("png", "tif"))).read()
    chm = np.squeeze(chm)
    f = FrameInfo(comb_img, chm)
    frames_val.append(f)

# training_frames = validation_frames = list(range(len(frames)))
training_frames = list(range(len(frames)))
validation_frames = list(range(len(frames_val)))
# divide fixed train val test for 31 frames in v2
# training_frames = [22, 2, 15, 27, 7, 18, 24, 19, 11, 3, 13, 12, 14, 4, 0, 30, 23, 9, 6, 21, 1, 25, 10, 8, 17]
# validation_frames = [26, 20, 16, 29, 5, 28]
# testing_frames = [6, 21, 1, 25, 10, 8, 17] # not used

# annotation_channels = config.input_label_channel + config.input_weight_channel
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, augmenter = "iaa").random_generator(config.BATCH_SIZE, normalize = config.normalize, maxmin_norm=config.maxmin_norm, gb_norm = config.gb_norm, gb_norm_FI = config.gb_norm_FI, robust_scale = config.robust_scale)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames_val, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize, maxmin_norm=config.maxmin_norm, gb_norm = config.gb_norm, gb_norm_FI = config.gb_norm_FI, robust_scale = config.robust_scale)
# test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)


for _ in range(1):
    train_images, chm = next(train_generator)
    print(train_images.shape, chm.shape)
    print(train_images.mean(axis = (0, 1, 2)))
    print(train_images.std(axis = (0, 1, 2)))
    print(train_images.max(), train_images.min())
    print('chm mean', chm.mean(axis = (1, 2, 3)))
    print('chm max', chm.max())
    chm = resize(chm[:, :, :], (config.BATCH_SIZE, train_images.shape[1], train_images.shape[2]))
    print(train_images.shape, chm.shape)
    print(type(train_images), type(chm))
    # chm = chm[..., np.newaxis]
    
    #overlay of annotation with boundary to check the accuracy
    #8 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    
    display_images(np.concatenate((train_images,chm), axis = -1))
    
   
# =============================================================================
# all sequential patches for pixel2pixel GANs
import random
from tqdm import tqdm

random.seed(1)
training_frames = list(random.sample(list(range(len(frames))), int(len(frames)*0.7)))

validation_frames = list(i for i in list(range(len(frames))) if i not in training_frames)

step_size = (196, 196)

imgs, labels = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, augmenter = 'iaa').all_sequential_patches(step_size, normalize = 0)

# save pngs, resample CHM to higher reso
tt = imgs[0, :,:,:]
tt.shape

np.save('./testt.npy', tt)

basePath = '/mnt/ssdc/Denmark/color2CHM/training_data2018/extracted_refined_cleaned_smooth8_pix2pix/'
basePhotos = os.path.join(basePath, 'photos/train/')
baseCHMs = os.path.join(basePath, 'CHMs/train/')


counter = 0

for i in tqdm(range(len(imgs))):
    curim = imgs[i, :,:,:]
    curlab = labels[i, :,:,:]
    curlab = resize(curlab[:, :, :], (int(curlab.shape[0]*2), int(curlab.shape[1]*2), 1), preserve_range = 1)
    curAname = basePhotos + str(i) + '.npy'
    curBname = baseCHMs + str(i) + '.npy'
    # print(curAname)
    # print(curBname)
    np.save(curAname, curim)
    np.save(curBname, curlab)
    counter += 1
    
    
    
tt2 = np.load(curBname)
tt2.shape

# =============================================================================
   
    
   
    
   
    
   
    
   
import tensorflow.keras.backend as K
def W_mae(y_true, y_pred, clip_delta=10, wei = 5):
    # weight more for high heights
    cond  = y_true < clip_delta
    loss_small = K.mean(K.abs(y_pred - y_true)) # no spatial weights
    loss_large = K.mean(wei*K.abs(y_pred - y_true))
    
    return tf.where(cond, loss_small, loss_large)

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def root_mse(y_true, y_pred):
    return K.sqrt(mse(y_true, y_pred))

# # OPTIMIZER = adaDelta
# OPTIMIZER = adam
# LOSS = 'smoothl1'

# #Only for the name of the model in the very end
# # OPTIMIZER_NAME = 'AdaDelta'
# OPTIMIZER_NAME = 'Adam'
# LOSS_NAME = 'smoothl1loss'
def mse4(y_true, y_pred):
    return K.mean(K.square(K.square(y_pred - y_true)))

def mse24(y_true, y_pred, wei = 0.0001):
    return mse(y_true, y_pred)+ wei*mse4(y_true, y_pred)

def mse24huber(y_true, y_pred, wei = (1, 5)):
    total = wei[0] * mse24(y_true, y_pred) + wei[1] * tf.keras.losses.huber(y_true, y_pred)
    return total


# def huber_loss(y_true, y_pred, c=1):
#     # https://www.astroml.org/book_figures/chapter8/fig_huber_loss.html
    
#     t = abs((K.batch_flatten(tf.dtypes.cast(y_true, K.floatx())) - K.batch_flatten(tf.dtypes.cast(y_pred, K.floatx()))))
#     flag = t > c
#     print('flag2', flag)
#     return np.sum((~flag) *(0.5 * t ** 2), -1)
#     # return np.sum((~flag) * (0.5 * t ** 2) - (flag / 1.0) * c * (0.5 * c - t), -1)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    # https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    
    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss) # element wise condition, but how to single number?


def weighted_huber_loss(y_true, y_pred, clip_delta=1.0):
    # https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    wei = y_true + 0.01
    squared_loss = 0.5 * K.square(error) * wei
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta) * wei

    return tf.where(cond, squared_loss, linear_loss) # element wise condition, but how to single number?


def weihuber_mse(y_true, y_pred):
    return weighted_huber_loss(y_true, y_pred) * 2 + mse(y_true, y_pred)

def huber_mse(y_true, y_pred):
    return huber_loss(y_true, y_pred) * 5 + mse(y_true, y_pred)

# def huber(y_true, y_pred, delta=1.0):
#   """Computes Huber loss value.
#   For each value x in `error = y_true - y_pred`:
#   ```
#   loss = 0.5 * x^2                  if |x| <= d
#   loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
#   ```
#   where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
#   Args:
#     y_true: tensor of true targets.
#     y_pred: tensor of predicted targets.
#     delta: A float, the point where the Huber loss function changes from a
#       quadratic to linear.
#   Returns:
#     Tensor with one scalar loss entry per sample.
#   """
#   y_pred = math_ops.cast(y_pred, dtype=K.floatx())
#   y_true = math_ops.cast(y_true, dtype=K.floatx())
#   delta = math_ops.cast(delta, dtype=K.floatx())
#   error = math_ops.subtract(y_pred, y_true)
#   abs_error = math_ops.abs(error)
#   half = ops.convert_to_tensor_v2_with_dispatch(0.5, dtype=abs_error.dtype)
#   return K.mean(
#       array_ops.where_v2(
#           abs_error <= delta, half * math_ops.pow(error, 2),
#           half * math_ops.pow(delta, 2) + delta * (abs_error - delta)),
#       axis=-1)




subArrayX = 3
subArrayY = 3
inputChannels = 1
outputChannels = 1
convFilter = K.ones((subArrayX, subArrayY, inputChannels, outputChannels))

def local_loss(y_true, y_pred):

    diff = K.abs(y_true-y_pred) #you might also try K.square instead of abs

    localSums = K.conv2d(diff, convFilter)
    localSums = K.batch_flatten(localSums) 
        #if using more than 1 channel, you might want a different thing here

    return K.max(localSums, axis=-1)   


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        

def mae_ssim(y_true, y_pred, delta = 2):
    ssiml = ssim_loss(y_true, y_pred)
    mael = mae(y_true, y_pred)
    return mael + delta * ssiml

# OPTIMIZER = adaDelta
OPTIMIZER = adam

OPTIMIZER = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9)


# OPTIMIZER = tf.keras.optimizers.RMSprop(
#     learning_rate=0.001,
#     momentum=0.9)

LOSS = 'huber'

#Only for the name of the model in the very end
# OPTIMIZER_NAME = 'AdaDelta'
# OPTIMIZER_NAME = 'SGD001'
OPTIMIZER_NAME = 'Adam'
LOSS_NAME = 'Wmae'

# Declare the path to the final model
# If you want to retrain an exising model then change the cell where model is declared. 
# This path is for storing a model after training.

timestr = time.strftime("%Y%m%d-%H%M")
# chf = config.input_image_channel + config.input_label_channel
# chs = reduce(lambda a,b: a+str(b), chf, '')
# chs = reduce(lambda a,b: a+str(b), config.input_image_channel, '')
chs = reduce(lambda a,b: a+str(b), config.extracted_filenames, '')


if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)
model_path = os.path.join(config.model_path,'trees_GBNIR2CHM_{}_{}_{}_{}_{}_{}_frames_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], len(all_files_c1), config.model_name))

print(model_path)
##############################################################################################################
##############################################################################################################

# if not os.path.exists(config.model_path):
#     os.makedirs(config.model_path)
# model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}_{}_frames_build.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], len(all_files_c1)))

##############################################################################################################
##############################################################################################################


model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
model = Xception([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
model.compile(optimizer=OPTIMIZER, loss=W_mae, metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
# model.compile(optimizer=OPTIMIZER, loss='mae', metrics=[tf.keras.metrics.MeanSquaredError()])

# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing
model.summary()

# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min', save_weights_only = False)

#reduceonplatea; It can be useful when using adam as optimizer
#Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
#cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                    patience=4, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)




#early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0], len(all_files_c1)))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

tf.config.run_functions_eagerly(True)

loss_history = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          epochs=config.NB_EPOCHS, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]

print(f"Model learning rate is: {K.get_value(model.optimizer.lr):.4f}")
weight_values = model.optimizer.get_weights()
# if continue training
model_path = os.path.join(config.model_path,'trees_RGBNIRNDVI2CHM_20210906-0036_Adam_mae_ssim2_redgreenblueinfraredndvi_512_50_frames_unet_attention.h5')
log_dir = os.path.join('./logs','UNet_20210906-0036_Adam_mae_ssim2_redgreenblueinfraredndvi_512_50_frames')


checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta
tf.config.run_functions_eagerly(True)


loaded_model = load_model(model_path, compile=False)
loaded_model.compile(optimizer=OPTIMIZER, loss=mae_ssim, metrics=[tf.keras.metrics.RootMeanSquaredError(), ssim_loss])
loaded_model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          initial_epoch = 336,
                          epochs=1000, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )








# The weights without the model architecture can also be saved. Just saving the weights is more efficent.

# weight_path="./saved_weights/UNet/{}/".format(timestr)
# if not os.path.exists(weight_path):
#     os.makedirs(weight_path)
# weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
# print(weight_path)

# Define the model and compile it
model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
model.compile(optimizer=OPTIMIZER, loss=tf.keras.losses.Huber(), metrics=tf.keras.metrics.RootMeanSquaredError())
# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min', save_weights_only = False)

#reduceonplatea; It can be useful when using adam as optimizer
#Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
#cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                    patience=4, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)

#early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0], len(all_files_c1)))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

tf.config.run_functions_eagerly(True)

loss_history = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          epochs=config.NB_EPOCHS, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]


loss_history2 = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          initial_epoch = 1001,
                          epochs = 2000, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]




#####
# change lr
###

step = tf.Variable(0, trainable=False)
boundaries = [500, 800, 1000]
values = [0.00001, 0.000005, 0.000001, 0.0000005]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

# Later, whenever we perform an optimization step, we pass in the step.
learning_rate = learning_rate_fn(step)

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Nadam

OPTIMIZER = Adam(lr= learning_rate, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)

model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
model.compile(optimizer=OPTIMIZER, loss= tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min', save_weights_only = False)

#reduceonplatea; It can be useful when using adam as optimizer
#Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
#cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                    patience=4, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)

#early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0], len(all_files_c1)))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

tf.config.run_functions_eagerly(True)

loss_history = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          epochs=config.NB_EPOCHS, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]

loss_history2 = [model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          initial_epoch = 1607,
                          epochs = 3500, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]





model_path = os.path.join(config.model_path,'trees_RGBNIR2CHM_20210505-1653_Adam_huber_redgreenblueinfrared_128_972_frames_unet_attention.h5')
log_dir = os.path.join('./logs','UNet_20210505-1653_Adam_huber_redgreenblueinfrared_128_972_frames')
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta
tf.config.run_functions_eagerly(True)


loaded_model = load_model(model_path, compile=False)
loaded_model.compile(optimizer=OPTIMIZER, loss= tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
loaded_model.fit(train_generator, 
                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
                          initial_epoch = 4301,
                          epochs=4500, 
                          validation_data=val_generator,
                          validation_steps=config.VALID_IMG_COUNT,
                          callbacks=callbacks_list,
                          workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )












# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # first gpu


# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import rasterio
# import imgaug as ia
# from imgaug import augmenters as iaa
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import imageio
# import os
# import time
# import rasterio.warp             # Reproject raster samples
# from functools import reduce
# from tensorflow.keras.models import load_model

# from core.UNet_attention_CHM import UNet
# from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
# from core.optimizers import adaDelta, adagrad, adam, nadam
# from core.frame_info_CHM import FrameInfo
# from core.dataset_generator_CHM import DataGenerator
# from core.split_frames import split_dataset
# from core.visualize import display_images

# import json
# from sklearn.model_selection import train_test_split
# from skimage.transform import resize

# # %matplotlib inline
# import matplotlib.pyplot as plt  # plotting tools
# import matplotlib.patches as patches
# from matplotlib.patches import Polygon

# import warnings                  # ignore annoying warnings
# warnings.filterwarnings("ignore")
# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)

# # %reload_ext autoreload
# # %autoreload 2
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# from config import UNetTraining_CHM
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

# print(tf.__version__)


# print(tf.config.list_physical_devices('GPU'))

# # Required configurations (including the input and output paths) are stored in a separate file (such as config/UNetTraining.py)
# # Please provide required info in the file before continuing with this notebook. 
 

# # In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder 
# # Eg. from configLargeCluster import UNetTraining
# config = UNetTraining_CHM.Configuration()

# # Read all images/frames into memory
# frames = []

# all_files = os.listdir(config.base_dir)
# # image channel 1
# all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
# print(all_files_c1)
# #all_files_c1: [c1_0.png, c1_1.png, c1_2.png,...]

# # for i, fn in enumerate(all_files_c1):
# #     # loop through rectangles

# #     comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
    
# #     for c in range(config.image_channels-2):
# #         #loop through raw channels
# #         comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
    
# #     #for aux chm channel, upsample by 2 
# #     # 0.4m resolution compared to 0.2m resolution
# #     chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
# #     #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
# #     # handle resolution diff: resize chm layer to the same shape as core image
# #     chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
# #     comb_img = np.append(comb_img, chm, axis = 0)
    
# #     comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
# #     annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
# #     annotation = np.array(annotation_im)
# #     weight_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.weight_fn)))
# #     weight = np.array(weight_im)
# #     f = FrameInfo(comb_img, annotation, weight)
# #     frames.append(f)
    
# ######################################################################################################    
# ######################################################################################################    


# for i, fn in enumerate(all_files_c1):
#     # loop through rectangles
#     comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
#     if config.single_raster or not config.aux_data:
#         # print('If single raster or multi raster without aux')
#         for c in range(config.image_channels-1):
#             #loop through raw channels
#             comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
#     else: # multi raster
#         print('Multi raster with aux data')
#         for c in range(config.image_channels-2): 
#                 #loop through raw channels
#             comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
#         #for aux chm channel, upsample by 2 
#         # 0.4m resolution compared to 0.2m resolution
#         chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
#         #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
#         # handle resolution diff: resize chm layer to the same shape as core image
#         chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
#         comb_img = np.append(comb_img, chm, axis = 0)

    
#     comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
#     # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
#     # np.asarray(annotation_im)
#     # annotation = np.array(annotation_im)
#     annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn))).read()
#     annotation = np.squeeze(annotation)
#     weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.weight_fn))).read()
#     weight = np.squeeze(weight)
#     chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.chm_fn))).read()
#     chm = np.squeeze(chm)
#     f = FrameInfo(comb_img, annotation, weight, chm)
#     frames.append(f)
# # comb_img = rasterio.open(os.path.join(config.base_dir, all_files_c1[0])).read()
# # for c in range(config.image_channels-2):
# #     #loop through channels
# #     comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
        
# # aa = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
# # aa2 = aa.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
# # aa2 = aa2[:, :327, :]

# # print(aa.shape)
# # plt.figure(figsize=(10, 15))
# # plt.imshow(aa[0, :, :])

# # plt.figure(figsize=(10, 15))
# # plt.imshow(aa2[0, :, :])


# # aa3 = resize(aa[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))

# # plt.figure(figsize=(10, 15))
# # plt.imshow(aa3[0, :, :])



# # b = np.array([[1,2], [3, 4]])

# # c = b.repeat(2, axis = 0).repeat(2, axis = 1)


# # def image_normalize(im, axis = (0,1), c = 1e-8):
# #     '''
# #     Normalize to zero mean and unit standard deviation along the given axis'''
# #     return (im - im.mean(axis)) / (im.std(axis) + c)
   
# # nor = 0.4
# # test_im = comb_img[:, :, :2]

# # r = np.random.random(1)
# # if nor >= r[0]:
# #     print('yes, the patch is normalized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# #     print('previous max', test_im.max())
# #     test_im = image_normalize(test_im, axis=(0, 1))
# #     print('after max', test_im.max())
    
# # r = np.random.random(1)
# # if nor >= r[0]:
# #     print('yes, the patch is normalized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# #     print('previous max', comb_img.max())
# #     comb_im = image_normalize(comb_img, axis=(0, 1))
# #     print('after max', comb_im.max())

# # test_im.mean((0,1))
# # comb_img.mean((0,1))
# ######################################################################################################
# ######################################################################################################

# # training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
# training_frames = validation_frames = testing_frames  = list(range(len(frames)))

# # divide fixed train val test for 31 frames in v2
# # training_frames = [22, 2, 15, 27, 7, 18, 24, 19, 11, 3, 13, 12, 14, 4, 0, 30, 23, 9, 6, 21, 1, 25, 10, 8, 17]
# # validation_frames = [26, 20, 16, 29, 5, 28]
# # testing_frames = [6, 21, 1, 25, 10, 8, 17] # not used

# annotation_channels = config.input_label_channel + config.input_weight_channel
# train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, config.boundary_weights, augmenter = "iaa").random_generator(config.BATCH_SIZE, normalize = config.normalize)
# val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
# test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)


# for _ in range(1):
#     train_images, chm = next(train_generator)
#     print(train_images.shape, chm.shape)
#     chm = resize(chm[:, :, :], (config.BATCH_SIZE, train_images.shape[1], train_images.shape[2]))
#     print(train_images.shape, chm.shape)
#     print(type(train_images), type(chm))
#     # chm = chm[..., np.newaxis]
    
#     #overlay of annotation with boundary to check the accuracy
#     #8 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    
#     display_images(np.concatenate((train_images,chm), axis = -1))
    
    
# ##################################################################################
# ##################################################################################   
# # train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, config.boundary_weights, augmenter = None).random_generator(config.BATCH_SIZE, normalize = config.normalize)

    
# # for _ in range(1):
# #     train_images, real_label = next(train_generator)
# #     print('shape of training image', train_images.shape)
# #     # a = np.max(train_images, axis = (1, 2, 3))
# #     # b = np.min(train_images, axis = (1, 2, 3))
# #     # c = np.mean(train_images, axis = (1, 2, 3))
# #     for i in train_images:
# #         print('new\n')
# #         print(i.mean((0,1)))
# #     # print(c)
        

# #################################################################################    
# ##################################################################################    
# import tensorflow.keras.backend as K

# def mse(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true))

# # OPTIMIZER = adaDelta
# OPTIMIZER = adam
# LOSS = 'mse'

# #Only for the name of the model in the very end
# # OPTIMIZER_NAME = 'AdaDelta'
# OPTIMIZER_NAME = 'Adam'
# LOSS_NAME = 'mse'

# # Declare the path to the final model
# # If you want to retrain an exising model then change the cell where model is declared. 
# # This path is for storing a model after training.

# timestr = time.strftime("%Y%m%d-%H%M")
# # chf = config.input_image_channel + config.input_label_channel
# # chs = reduce(lambda a,b: a+str(b), chf, '')
# # chs = reduce(lambda a,b: a+str(b), config.input_image_channel, '')
# chs = reduce(lambda a,b: a+str(b), config.extracted_filenames, '')


# if not os.path.exists(config.model_path):
#     os.makedirs(config.model_path)
# model_path = os.path.join(config.model_path,'trees_NIRGB2CHM_{}_{}_{}_{}_{}_{}_frames_unet_attention.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], len(all_files_c1)))


# ##############################################################################################################
# ##############################################################################################################

# # if not os.path.exists(config.model_path):
# #     os.makedirs(config.model_path)
# # model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}_{}_frames_build.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0], len(all_files_c1)))

# ##############################################################################################################
# ##############################################################################################################



# # The weights without the model architecture can also be saved. Just saving the weights is more efficent.

# # weight_path="./saved_weights/UNet/{}/".format(timestr)
# # if not os.path.exists(weight_path):
# #     os.makedirs(weight_path)
# # weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
# # print(weight_path)

# # Define the model and compile it
# model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
# model.compile(optimizer=OPTIMIZER, loss='mse', metrics=tf.keras.metrics.RootMeanSquaredError())
# # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


# # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing


# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
#                              save_best_only=True, mode='min', save_weights_only = False)

# #reduceonplatea; It can be useful when using adam as optimizer
# #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
# #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
#                                    patience=4, verbose=1, mode='min',
#                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)

# #early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

# log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_frames'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0], len(all_files_c1)))
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

# tf.config.run_functions_eagerly(True)

# loss_history = [model.fit(train_generator, 
#                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                          epochs=config.NB_EPOCHS, 
#                          validation_data=val_generator,
#                          validation_steps=config.VALID_IMG_COUNT,
#                          callbacks=callbacks_list,
#                          workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )]


# loss_history2 = [model.fit(train_generator, 
#                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                          initial_epoch = 2000,
#                          epochs = 2500, 
#                          validation_data=val_generator,
#                          validation_steps=config.VALID_IMG_COUNT,
#                          callbacks=callbacks_list,
#                          workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )]

# # Load model after training
# # If you load a model with different python version, than you may run into a problem: https://github.com/keras-team/keras/issues/9595#issue-303471777
# model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity}, compile=False)

# # In case you want to use multiple GPU you can uncomment the following lines.
# # from tensorflow.python.keras.utils import multi_gpu_model
# # model = multi_gpu_model(model, gpus=2, cpu_merge=False)

# model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])

# # Print one batch on the training/test data!
# for i in range(1):
#     test_images, real_label = next(test_generator)
#     #5 images per row: pan, ndvi, label, weight, prediction
#     prediction = model.predict(test_images, steps=1)
#     print(prediction.shape)
#     real_label = resize(real_label[:, :, :], (config.BATCH_SIZE, test_images.shape[1], test_images.shape[2]))
#     prediction = resize(prediction[:, :, :], (config.BATCH_SIZE, test_images.shape[1], test_images.shape[2]))
#     display_images(np.concatenate((test_images, real_label, prediction), axis = -1))