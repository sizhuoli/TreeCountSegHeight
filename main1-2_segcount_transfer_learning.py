#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:07:16 2021

@author: sizhuo
"""

# segcount density


import os

import ipdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # first gpu

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
import rasterio.warp  # Reproject raster samples
from functools import reduce
from tensorflow.keras.models import load_model

from core2.UNet_attention_segcount import UNet
from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.frame_info_segcount import FrameInfo
from core2.dataset_generator_segcount import DataGenerator

from core2.split_frames import split_dataset
from core2.visualize import display_images

import json
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# %matplotlib inline
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings  # ignore annoying warnings

warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import tensorflow.keras.backend as K
from config import UNetTrainingFinland
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, \
    TensorBoard
import scipy

tf.config.run_functions_eagerly(True)


print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

config = UNetTrainingFinland.Configuration()

# Read all local / finetuning data

all_files = os.listdir(config.base_dir)
# image channel 1
all_files_c1 = [fn for fn in all_files if
                fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
print(all_files_c1)

frames = []

# oversample f times to match with the initial data if needed
oversample_times = 2
for oo in range(oversample_times):
    print('loading local data')
    for i, fn in enumerate(all_files_c1):
        # loop through rectangles
        comb_img = rasterio.open(os.path.join(config.base_dir, fn)).read()
        # print('bef', np.min(comb_img), np.mean(comb_img), np.max(comb_img))
        # print(comb_img.shape)
        if config.upsample:
            comb_img = resize(comb_img[:, :, :], (1, int(comb_img.shape[1] * 2), int(comb_img.shape[2] * 2)),
                              preserve_range=1)

        if config.single_raster or not config.aux_data:
            # print('If single raster or multi raster without aux')
            for c in range(1, config.image_channels):
                # loop through raw channels
                # comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
                cur = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],
                                                                             config.extracted_filenames[c]))).read()
                # print('bef', np.min(cur), np.mean(cur))
                if config.upsample:
                    cur = resize(cur[:, :, :], (1, int(cur.shape[1] * 2), int(cur.shape[2] * 2)), preserve_range=1)
                # print('aft', np.min(cur))
                comb_img = np.append(comb_img, cur, axis=0)

        else:  # multi raster
            print('Multi raster with aux data')
            for c in range(1, config.image_channels):
                # loop through raw channels
                cur = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],
                                                                             config.extracted_filenames[c]))).read()
                # print(np.min(cur), np.mean(cur), np.max(cur))
                if config.upsample:
                    cur = resize(cur[:, :, :], (1, int(comb_img.shape[1] * 2), int(comb_img.shape[2] * 2)),
                                 preserve_range=1)
                # print(np.min(cur))
                comb_img = np.append(comb_img, cur, axis=0)


        comb_img = np.transpose(comb_img, axes=(1, 2, 0))  # Channel at the end
        # print('statis', comb_img.min(), comb_img.mean(), comb_img.max())
        # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
        # np.asarray(annotation_im)
        # annotation = np.array(annotation_im)
        annotation = rasterio.open(
            os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0], config.annotation_fn))).read()
        annotation = np.squeeze(annotation)
        # print('bef', np.unique(annotation))
        if config.upsample:
            annotation = scipy.ndimage.zoom(annotation, 2, order=1)
            # annotation = resize(annotation[:, :], (int(annotation.shape[0]*2), int(annotation.shape[1]*2)), preserve_range = 1).astype(int)
        # print('aft', np.unique(annotation))
        weight = rasterio.open(
            os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0], config.weight_fn))).read()
        weight = np.squeeze(weight)
        # print('bef', np.unique(weight))
        if config.upsample:
            weight = scipy.ndimage.zoom(weight, 2, order=1)
            # weight = resize(weight[:, :], (int(weight.shape[0]*2), int(weight.shape[1]*2)), preserve_range = 1).astype(int)
        density = rasterio.open(
            os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0], config.density_fn))).read()
        density = np.squeeze(density)

        if config.upsample:
            density = resize(density[:, :], (int(density.shape[0] * 2), int(density.shape[1] * 2)),
                             preserve_range=1).astype(float)
            # print('aft', density.sum())
            density = density * (density.shape[0] / float(density.shape[0] * 2)) * (
                        density.shape[1] / float(density.shape[1] * 2))

        f = FrameInfo(comb_img, annotation, weight, density)
        frames.append(f)

print('local data loaded, total no. frames: ', len(frames))

# read all initial training data
print('loading initial training data')
all_files2 = os.listdir(config.base_dir2)
# image channel 1
all_files_c12 = [fn for fn in all_files2 if
                 fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
print(all_files_c12)

for i, fn in enumerate(all_files_c12):
    # loop through rectangles
    img1 = rasterio.open(os.path.join(config.base_dir2, fn)).read()
    # print(img1.mean())
    # print(img1.std())
    # print(np.min(img1), np.mean(img1), np.max(img1))
    if config.single_raster or not config.aux_data:
        # print('If single raster or multi raster without aux')
        for c in range(1, config.image_channels):
            # loop through raw channels
            img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir2,
                                                              fn.replace(config.extracted_filenames[0],
                                                                         config.extracted_filenames[c]))).read(),
                             axis=0)
            # print(np.min(img1), np.mean(img1), np.max(img1))
    else:  # multi raster
        print('Multi raster with aux data')
        for c in range(config.image_channels1 - 1):
            # loop through raw channels
            img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],
                                                                                          config.channel_names1[
                                                                                              c + 1]))).read(), axis=0)

        img2 = rasterio.open(
            os.path.join(config.base_dir, fn.replace(config.channel_names1[0], config.channel_names2[0]))).read()

    img1 = np.transpose(img1, axes=(1, 2, 0))  # Channel at the end
    # img2 = np.transpose(img2, axes=(1,2,0))
    annotation = rasterio.open(
        os.path.join(config.base_dir2, fn.replace(config.extracted_filenames[0], config.annotation_fn))).read()
    annotation = np.squeeze(annotation)
    weight = rasterio.open(
        os.path.join(config.base_dir2, fn.replace(config.extracted_filenames[0], config.weight_fn))).read()
    weight = np.squeeze(weight)
    density = rasterio.open(
        os.path.join(config.base_dir2, fn.replace(config.extracted_filenames[0], config.density_fn))).read()
    density = np.squeeze(density)
    f = FrameInfo(img1, annotation, weight, density)
    frames.append(f)

print('initial data loaded, total no. frames: ', len(frames))




training_frames = validation_frames = list(range(len(frames)))

annotation_channels = config.input_label_channel + config.input_weight_channel + config.input_density_channel
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames,
                                annotation_channels, config.boundary_weights, augmenter='iaa').random_generator(
    config.BATCH_SIZE, normalize=config.normalize)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames,
                              annotation_channels, config.boundary_weights, augmenter=None).random_generator(
    config.BATCH_SIZE, normalize=config.normalize)



train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames,
                                annotation_channels, config.boundary_weights, augmenter='iaa').random_generator(
    config.BATCH_SIZE, normalize=config.normalize)

for _ in range(3):
    train_images, real_label = next(train_generator)
    ann = real_label['output_seg'][..., 0]
    wei = real_label['output_seg'][..., 1]
    print('count', real_label['output_dens'].sum(axis=(1, 2)))
    print('Boundary highlighted weights:', np.unique(wei))
    print(np.unique(wei))
    print(np.unique(ann))
    print(real_label['output_dens'].shape)
    print('max', real_label['output_dens'].max())
    # overlay of annotation with boundary to check the accuracy
    # 8 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    overlay = ann + wei
    # overlay = overlay[:,:,:,np.newaxis]
    overlay = overlay[..., np.newaxis]
    display_images(
        np.concatenate((train_images, real_label['output_seg'], overlay, real_label['output_dens']), axis=-1))

OPTIMIZER = adam  # tf.keras.optimizers.Adam(lr= 0.00001, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8) #adagrad #tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) #adam
LOSS = tversky

##### finetune model##############
model_path = config.model_path
loaded_model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss': dice_loss,
                                                      'accuracy': accuracy, 'specificity': specificity,
                                                      'sensitivity': sensitivity, 'miou': miou,
                                                      'weight_miou': weight_miou, 'K': K}, compile=False)



# save logs
timestr = time.strftime("%Y%m%d-%H%M")

# log_dir = config.log_dir
new_model_path = os.path.join(config.new_model_path, 'finetune_{}.h5'.format(timestr))
checkpoint = ModelCheckpoint(new_model_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=False)


log_dir = os.path.join(config.log_dir, 'finetune_{}'.format(timestr))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                          embeddings_data=None, update_freq='epoch')


callbacks_list = [checkpoint, tensorboard]

# =============================================================================
# finetune FIDK
loaded_model.compile(optimizer=OPTIMIZER, loss={'output_seg': LOSS, 'output_dens': 'mse'},
                     loss_weights={'output_seg': 1., 'output_dens': 10000},
                     metrics={
                         'output_seg': [dice_coef, specificity, sensitivity, miou, weight_miou, accuracy],
                         'output_dens': [tf.keras.metrics.RootMeanSquaredError()]})

loss_history = [loaded_model.fit(train_generator,
                                 steps_per_epoch=config.MAX_TRAIN_STEPS,
                                 initial_epoch=1401,
                                 epochs=1700,
                                 validation_data=val_generator,
                                 validation_steps=config.VALID_IMG_COUNT,
                                 callbacks=callbacks_list,
                                 workers=1,
                                 )]

