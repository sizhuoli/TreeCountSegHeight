#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 01:33:25 2021

@author: sizhuo
"""

import os

# import keras.saving.save

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
# import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps
import glob
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape
from skimage.transform import resize

import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
import time

from itertools import product
import cv2
from sklearn.metrics import mean_absolute_error, median_absolute_error

import sys
import math
# from core.UNet_multires import UNet

# from rasterstats import zonal_stats
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from collections import defaultdict
import random
from rasterio.enums import Resampling
# from osgeo import ogr, gdal

from scipy.optimize import curve_fit
from matplotlib import colors
import glob
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
import csv
from shapely.geometry import shape
from rasterio.windows import Window
from rasterio.features import shapes
# import multiprocessing
from itertools import product
import tensorflow as tf
from pathlib import Path
import ipdb
from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.UNet_attention_segcount import UNet
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.frame_info import image_normalize
# from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# ipdb.set_trace()
class anaer:
    def __init__(self, config):
        self.config = config
        self.all_files = load_files(self.config)

    def load_model(self):
        OPTIMIZER = adam
        if not self.config.change_input_size:

            # tf.config.threading.set_intra_op_parallelism_threads(
            #     1
            # )
            # K.clear_session()
            # ipdb.set_trace()
            # self.model = load_model(self.config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity, 'K': K}, compile=False)
            # self.model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])

            # in case different python versions used for training and testing
            if self.config.multires:
                from core2.UNet_multires_attention_segcount import UNet
            elif not self.config.multires:
                from core2.UNet_attention_segcount import UNet
            # ipdb.set_trace()
            self.model = UNet([self.config.BATCH_SIZE, *self.config.input_shape],
                              self.config.input_label_channel, inputBN=self.config.inputBN)
            self.model.load_weights(self.config.trained_model_path)
            self.model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            # prediction mode
            self.model.trainable = False
            self.model_chm = None


        elif self.config.change_input_size:
            # self.models = []
            # modeli = load_model(self.config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            # modeli.summary()
            # self.weiwei = modeli.get_weights()
            # # print(modeli.input[0])
            # print(modeli.layers[0]._batch_input_shape)
            # if self.config.rgb2gray:
            #     modeli.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, 1)
            # else:
            #     modeli.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channel_names1))
            #
            # print(modeli.layers[0]._batch_input_shape)
            # # modeli.layers[0]._batch_input_shape = oldInputShape
            # new_model = tf.keras.models.model_from_json(modeli.to_json())
            #
            #
            # # copy weights from old model to new one
            # for layer in new_model.layers:
            #     try:
            #         layer.set_weights(modeli.get_layer(name=layer.name).get_weights())
            #     except:
            #         print("Could not transfer weights for layer {}".format(layer.name))
            #
            #
            #
            # self.weiwei2 = new_model.get_weights()
            # print(new_model.summary())
            #
            # new_model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            # self.models.append(new_model)
            #     # self.models.append(modeli)
            # # change input size for chm model as well
            # self.model_chm = load_model(self.config.trained_model_path_chm, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            # # self.model_chm.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_chm': tf.keras.losses.Huber()},
            # #               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
            # #                     'output_chm':[tf.keras.metrics.RootMeanSquaredError()]})
            #
            # if self.config.addndvi:
            #     self.model_chm.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channels)+1)
            #
            # elif not self.config.addndvi:
            #     self.model_chm.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channels))
            #
            # self.new_model_chm = tf.keras.models.model_from_json(self.model_chm.to_json())
            #
            #
            # # copy weights from old model to new one
            # for layer in self.new_model_chm.layers:
            #     try:
            #         layer.set_weights(self.model_chm.get_layer(name=layer.name).get_weights())
            #     except:
            #         print("Could not transfer weights for layer {}".format(layer.name))
            # print(self.new_model_chm.summary())
            # self.model_chm = self.new_model_chm
            # self.new_model_chm.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            raise NotImplementedError('not supporting yet')
        print('Model(s) loaded')


    def segcount_RUN(self):
        predict_ready_run(self.config, self.all_files, self.model, self.model_chm, self.config.output_dir, eva = 0, th = self.config.threshold, rgb2gray = self.config.rgb2gray)
        return



def load_files(config):
    exclude = set(['water_new', 'md5', 'pred', 'test_kay'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    if config.fillmiss: # only fill the missing predictions (smk) # while this include north
        doneff = gps.read_file(config.grids)
        donef2 = list(doneff['filepath'])
        done_names= set([os.path.basename(f)[:6] for f in donef2])
        all_files = [f for f in all_files if os.path.splitext(f[1])[0] not in done_names]
    # print(all_files)
    # print('Number of missing tif to predict:', len(all_files))

    return all_files




def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
    # set the 4 borderlines to 0 to remove the border effect
    newPredictions[:10, :] = 0
    newPredictions[-10:, :] = 0
    newPredictions[:, :10] = 0
    newPredictions[:, -10:] = 0

# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get.
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    return (res)

def addTOResult_chm(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    # plt.figure()
    # plt.imshow(np.squeeze(currValue))
    newPredictions = prediction[:he, :wi]
    # plt.figure()
    # plt.imshow(np.squeeze(newPredictions))
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    elif operator == "MIX": # alpha blending # note do not combine with empty regions
        # print('mix')
        mm1 = currValue!=0
        try:
            currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
            mm2 = (currValue==0)
            currValue[mm2] = newPredictions[mm2]
            resultant = currValue
        except:
            resultant = newPredictions[:256, :256]

    else: #operator == 'REPLACE':
        resultant = newPredictions
    res[row:row+he, col:col+wi] =  resultant
    return (res)


# 2 tasks
def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        # print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        seg, dens = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        seg, dens = model.predict(tm1) # tm is a list []

    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens


def predict_using_model_segcount_fi(model, batch, batch_pos, maskseg, maskdens, operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    b1 = batch[0]
    tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
    seg, dens = model.predict(tm1, workers = 10, use_multiprocessing = True)
    # ipdb.set_trace()
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)

        if upsample and downsave:   # no preserve range
            # print('**********************UPsample for prediction***************************')
            # print('**********************Downsample for saving***********************')
            # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
            # print(upscale)
            p = resize(p[:, :], (int(p.shape[0]/upscale), int(p.shape[1]/upscale)), preserve_range=True)
            c = resize(c[:, :], (int(c.shape[0]/upscale), int(c.shape[1]/upscale)), preserve_range=True)
            if rescale_values:
                # rescale values
                p = p * (p.shape[0] / float(p.shape[0]/upscale)) * (p.shape[1] / float(p.shape[1]/upscale))
                c = c * (c.shape[0] / float(c.shape[0]/upscale)) * (c.shape[1] / float(c.shape[1]/upscale))

        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens





def predict_using_model_chm_fi(model, batch, batch_pos, mask, operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200

    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        prediction = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        # print(tm1.mean(axis = (0, 1, 2)))
        prediction = model.predict(tm1) # tm is a list []
        # print('pred', prediction.min(), prediction.max())
        # print(prediction.mean())
        # print('pred from model', prediction.shape)
        
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # if upsample and downsave:   # no preserve range for chm model, as already a resolution reduce
            
        #     p = resize(p[:, :], (int(p.shape[0]/upscale), int(p.shape[1]/upscale)), preserve_range=True)
        #     if rescale_values:
        #         # rescale values
        #         p = p * (p.shape[0] / float(p.shape[0]/upscale)) * (p.shape[1] / float(p.shape[1]/upscale))
                
        # print('before add tore ', p.shape)
        mask = addTOResult_chm(mask, p, row, col, he, wi, operator)
    
    return mask


def detect_tree_segcount_fi(config, model, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1, multires = 1, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1, rgb2gray = 0):
    if 'chm' in config.channel_names1:
        raise NotImplementedError('not supporting chm as input yet')
    else:
        CHM = 0
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()
    # tile normalize:
    if config.segcount_tilenorm:
        print('tile norm')
        temp_imm = img.read()
        temp_imm = np.transpose(temp_imm, axes=(1,2,0))
        means = np.mean(temp_imm, axis = (0, 1))
        stds = np.std(temp_imm, axis = (0, 1))


    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    if downsave or not upsample:
        masksegs = np.zeros((nrows, nols), dtype=np.float32)
        maskdenss = np.zeros((nrows, nols), dtype=np.float32)

    elif not downsave:
        masksegs = np.zeros((int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        maskdenss = np.zeros((int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        meta.update(
                    {'width': int(nols*upscale),
                     'height': int(nrows*upscale)
                    }
                    )

    if rgb2gray:
        meta.update(
                    {'count': 1,
                    }
                    )

    batch = []
    batch_pos = [ ]
    # ipdb.set_trace()
    for col_off, row_off in tqdm(offsets):

        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)

        if upsample:

            patch1 = np.zeros((int(height*upscale), int(width*upscale), len(config.channels)))
            if config.band_switch:
                patch1 = np.zeros((int(height*upscale), int(width*upscale), int(len(config.channels))))

            temp_im1 = img.read(
                                out_shape=(
                                img.count,
                                int(window.height*upscale),
                                int(window.width*upscale)
                            ),
                            resampling=Resampling.bilinear, window = window)
        else:
            # no upsample

            patch1 = np.zeros((height, width, len(config.channels))) #Add zero padding in case of corner images
            if config.band_switch:
                patch1 = np.zeros((height, width, int(len(config.channels))))

            temp_im1 = img.read(window = window)

        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        try:
            temp_im1 = temp_im1[:,:,config.channels]
        except:
            ipdb.set_trace()


        if rgb2gray:
            temp_im1 = rgb2gray_convert(temp_im1)[..., np.newaxis]


        if config.segcount_tilenorm:
            temp_im1 = (temp_im1-means)/stds

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel

        if upsample:
            patch1[:int(window.height*upscale), :int(window.width*upscale)] = temp_im1
        else:
            patch1[:int(window.height), :int(window.width)] = temp_im1
        # ipdb.set_trace()

        batch.append(patch1)

        if downsave or not upsample:
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        elif not downsave:
            batch_pos.append((int(window.col_off*upscale), int(window.row_off*upscale), int(window.width*upscale), int(window.height*upscale)))

        if (len(batch) == config.BATCH_SIZE):
            # print('processing one batch')
            masksegs, maskdenss = predict_using_model_segcount_fi(model, batch, batch_pos, masksegs, maskdenss, 'MAX', upsample = upsample, downsave = downsave, upscale = upscale, rescale_values=rescale_values)

            batch = []
            batch_pos = []

    if batch:
        masksegs, maskdenss = predict_using_model_segcount_fi(model, batch, batch_pos, masksegs, maskdenss, 'MAX', upsample = upsample, downsave = downsave, upscale = upscale, rescale_values=rescale_values)
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta


def detect_tree_rawtif_fi(config, model, img, channels,  width=256, height=256, stride = 128, normalize=0, maxnorm = 0, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # for FI, upsample input images to deal with resolution diff

    #single rater or multi raster without aux
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])

    if downsave and upsample:
        mask = np.zeros((int(nrows), int(nols)), dtype=meta['dtype'])

    # if for denmark-like resolution
    else:
        mask = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])
        # print('HERE!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(mask.shape)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        if upsample:
            # patch1 = np.zeros((height, width, len(img)))
            if config.addndvi:
                patch = np.zeros((height*2, width*2, len(channels)+1))
            else:
                patch = np.zeros((height*2, width*2, len(channels)))

            read_im = img.read(
                                out_shape=(
                                img.count,
                                int(window.height*2), # upsample by 2
                                int(window.width*2)
                            ),
                            resampling=Resampling.bilinear, window = window)
        else:
            # no upsample
            if config.addndvi:
                patch = np.zeros((height, width, len(channels)+1)) #Add zero padding in case of corner images
            else:
                patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images

            read_im = img.read(window = window)
        # #single rater or multi raster without aux # must be multi raster
        # patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        # read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels] # swap channels if needed
        # print(temp_im.mean(axis = (0,1)))

        # print(temp_im.mean(axis = (0, 1)))
        # print('size', patch.shape)
        if config.addndvi:
            NDVI = (temp_im[:, :, -1].astype(float) - temp_im[:, :, 0].astype(float)) / (temp_im[:, :, -1].astype(float) + temp_im[:, :, 0].astype(float))
            NDVI = NDVI[..., np.newaxis]
            # print('NDVI', NDVI.max(), NDVI.min())
            # print('bands', temp_im.max(), temp_im.min())
            temp_im = np.append(temp_im, NDVI, axis = -1)


        # print('max rescaling', temp_im.max)
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        # if len(config.channels) > 3:
        if len(channels) > 3:
            # print('channel > 3')
            if config.gbnorm: # all bands
                logging.info('all bands - standarization - gb')
                temp_im = temp_im / 255
                temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])

                # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
            elif config.robustscale: # all bands robust scale
                logging.info('incomplete')
                # todo: robust scale for all bands
                # temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                if normalize:
                    temp_im = image_normalize(temp_im, axis=(0,1))

        # elif len(config.channels) == 3: # 3 bands
        elif len(channels) == 3: # 3 bands, by default this is for FI so only rg+NIR bands
            if channels[0] == 1: # to confirm: this is the case for FI, first band is green
                if config.robustscale: # DK rebust scale
                    logging.info('3 bands - robust scale - DK - gb')
                    # /179 * 255
                    # # only for inf
                    # temp_im = np.array([0, 0, 29]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-29])
                    # q01 and q99 from DK training data
                    # temp_im = np.array([38, 49, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([167-38, 151-49, 212-62])
                    # # q001 q999
                    # temp_im = np.array([29, 41, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([191-29, 175-41, 225-39])
                    # # q001 q999 v2
                    # temp_im = np.array([29, 41, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255-29, 255-41, 255-39])
                    # # only for inf , q01, q99
                    # temp_im = np.array([0, 0, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 212-62])
                    # # only for inf , q01, q99
                    # temp_im = np.array([0, 0, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-62])
                    # # only for inf , q1, max
                    # temp_im = np.array([0, 0, 119]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-119])
                    # # only for inf , q15, max
                    # temp_im = np.array([0, 0, 130]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-130])

                    # # only for inf , q25, max
                    # temp_im = np.array([0, 0, 145]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-145])
                    # only for inf , q20, max
                    temp_im = np.array([0, 0, 139]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-139])

                    # # # q001 q999 inf
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 225-39])
                    # # q001 max inf
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-39])

                    temp_im = (temp_im - np.array([[73.0, 72.0, 145.0]]))/ np.array([[113.0-73.0, 96.0-72.0, 182.0-145.0]])


                if config.robustscaleFI_gb: # 3 bands FI self stats gbly
                    logging.info('3 bands - robust scale - FI - gb')
                    # q1s = np.array([46, 65, 21])
                    # q3s = np.array([ 99, 106, 130])
                    temp_im = (temp_im - q1s)/ (q3s - q1s)

                if config.robustscaleFI_local: # 3 bands FI self stats locally
                    logging.info('3 bands - robust scale - FI - local')
                    q1s = np.quantile(temp_im, 0.25, axis = (0,1))
                    q3s = np.quantile(temp_im, 0.75, axis = (0,1))
                    assert len(q1s) == 3 and len(q3s) == 3
                    temp_im = (temp_im - q1s)/ (q3s - q1s)

                if config.gbnorm: # 3 bands DK gb norm from training set
                    # only for inf , q20, max
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-39])

                    logging.info('3 bands - gb norm - DK')
                    # temp_im = temp_im / 255
                    temp_im = temp_im / 255 # int16 instead of int8
                    # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                    temp_im = (temp_im - np.array([[0.350, 0.321, 0.560]]))/ np.array([[0.895, 0.703, 1.107]])

                if config.gbnorm_FI:
                    # normalize globally gb inf from FI training data
                    logging.info('3 bands - gb norm - FI training data')
                    temp_im = temp_im / 255
                    # temp_im = (temp_im - np.array([[0.300, 0.338, 0.343]]))/ np.array([[0.168, 0.153, 0.146]])
                    temp_im = (temp_im - np.array([[0.253, 0.300, 0.321]]))/ np.array([[0.122, 0.118, 0.127]])

                if config.localtifnorm:
                    logging.info('3 bands - local tif norm - DK')
                    temp_im = temp_im / 255
                    temp_im = image_normalize(temp_im, axis=(0,1))
            elif channels[0]==0: # this is the case for RGB trained model
                if config.gbnorm: # 3 bands DK gb norm from training set
                    # print('3 bands - gb norm - DK')
                    temp_im = temp_im / 255
                    temp_im = (temp_im - np.array([[0.317, 0.350, 0.321]]))/ np.array([[0.985, 0.895, 0.703]])
                    # print(temp_im.mean(axis = (0, 1)))
                    # print(temp_im.std(axis = (0,1)))

        # print('read', temp_im.shape)
        if upsample:
            patch[:int(window.height*2), :int(window.width*2)] = temp_im
        else:
            patch[:int(window.height), :int(window.width)] = temp_im
        # patch[:window.height, :window.width] = temp_im
        batch.append(patch)

        # this only applies to coraser resolution
        # if downsave or not upsample:
        #     batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        # elif not downsave:
        #     print('upsave-----------------')
        #     batch_pos.append((int(window.col_off*2), int(window.row_off*2), int(window.width*2), int(window.height*2)))

        if upsample:
            batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        elif not upsample: # dk-like chm prediction
            batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        
        # batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1)
            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1)
        batch = []
        batch_pos = []

    if maxnorm:
        mask = mask * 97.19

    return(mask, meta)



def predict_ready_run(config, all_files, model_segcount, model_chm, output_dir, eva = 0, th = 0.5, rgb2gray = 0):
    counter = 1
    th = th
    counts = {}
    if eva:
        heights_gt = np.array([])
        heights_pr = np.array([])
    outputFiles = []
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        print(outputFile)
        outputFile2 = outputFile.replace('seg.tif', 'chm.tif')
        # ipdb.set_trace()
        if not os.path.exists(outputFile) or not os.path.exists(outputFile2):
            # print(outputFile)
            outputFiles.append(outputFile)
            # outputFileChm = os.path.join(output_dir, filename.replace(config.input_image_type, config.output_image_type))

            with rasterio.open(fullPath) as img:
                # for only south tifs
                # print(raw.profile['transform'][5])
                if config.segcountpred:
                    print('creating file', outputFile)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(config, model_segcount, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize,
                                                                                                auxData = config.aux_data, singleRaster=config.single_raster, multires = config.multires, upsample = config.upsample, downsave = config.downsave, upscale = config.upscale, rescale_values=config.rescale_values, rgb2gray = rgb2gray)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg.tif', 'density.tif'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()

                if config.chmpred:
                    print('creating file', outputFile2)
                    detectedMaskChm, detectedMetaChm = detect_tree_rawtif_fi(config, model_chm, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave, upscale = config.upscale, rescale_values = config.rescale_values) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    writeMaskToDiskChm(detectedMaskChm, detectedMetaChm, outputFile.replace('seg.tif', 'chm.tif'), image_type = config.output_image_type, write_as_type = config.output_dtype_chm, scale = 0)

                    if eva:
                        chmPath = config.gt_chm_dir  + config.chm_pref + filename[:-4] + config.chm_sufix
                        # print(chmPath)
                        with rasterio.open(chmPath) as chm:
                            pseg = np.squeeze(detectedMaskSeg)
                            ggt = np.squeeze(chm.read())
                else:
                    continue


            counter += 1


        else:
            print('Skipping: File already analysed!', fullPath)

    return counter





def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, output_shapefile_type, write_as_type = 'uint8', th = 0.5, create_countors = False, convert = 1, rescale = 0):
    # Convert to correct required before writing
    meta = detected_meta.copy()
    if convert:
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask<th]=0
            detected_mask[detected_mask>=th]=1

    if rescale:
        # for densty masks, multiply 10e4
        detected_mask = detected_mask*10000

    detected_mask = detected_mask.astype(write_as_type)
    if detected_mask.ndim != 2:
        detected_mask = detected_mask[0]

    meta['dtype'] =  write_as_type
    meta['count'] = 1
    if rescale:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 32767
                            }
                        )
    else:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 255
                            }
                        )
        ##################################################################################################
        ##################################################################################################
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
    if create_countors:
        wp = wp.replace(image_type, output_shapefile_type)
        # create_contours_shapefile(detected_mask, detected_meta, wp)


def writeMaskToDiskChm(detected_mask, detected_meta, wp, image_type, write_as_type = 'float32', scale = 1):
    # Convert to correct required before writing

    # scale by 100 and then store
    print('range height', detected_mask.min(), detected_mask.max())
    if scale:
        detected_mask = detected_mask * 100

    # print(detected_mask.max())

    print('mask', detected_mask.shape)
    print('meta', detected_meta['height'])
    detected_mask = detected_mask.astype(write_as_type)
    detected_meta['dtype'] =  write_as_type
    detected_meta['count'] = 1
    detected_meta.update(
                        {'compress':'lzw',
                            'nodata': 9999
                        }
                    )
    try:
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    except:
        detected_meta.update(
                        {'compress':'lzw',
                            'nodata': 9999,
                            'driver': 'GTiff',
                        }
                    )
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    return

def rgb2gray_convert(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
