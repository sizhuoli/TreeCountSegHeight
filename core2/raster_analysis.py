#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:05:50 2021

@author: sizhuo
"""

import os
os.environ['PROJ_LIB'] = '/usr/share/proj'
import os
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps
import csv
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape

import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
import time

from itertools import product
from tensorflow.keras.models import load_model
import cv2
from skimage.transform import resize
from sklearn.metrics import mean_absolute_error, median_absolute_error

# import sys

# from core.UNet_multires import UNet
# from core.UNet_subatte_resi_3conv import UNet
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info_multires import image_normalize
# from core.dataset_generator_multires import DataGenerator
from core.dataset_generator import DataGenerator
# from core.split_frames import split_dataset
from core.visualize import display_images


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
import glob
from collections import defaultdict

from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import tensorflow as tf
print(tf.__version__)
from scipy.optimize import curve_fit

from rasterstats import zonal_stats
from matplotlib import colors
import csv
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress

class segcount_analyze:
    def __init__(self, config):
        self.config = config
        OPTIMIZER = adam #
        if self.config.tasks == 1:
            self.model = load_model(self.config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            self.model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
        
        elif self.config.tasks == 2:
            self.model = []
            for mod in self.config.trained_model_path:
                modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                modeli.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_count':'mse'},
                               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                                   'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
                self.model.append(modeli)

        self.all_files = load_files(self.config)

    def preprocess(self):
        # # compute gb stats
        self.q1s, self.q3s, self.min, self.max = compute_stat_gb(self.config, self.all_files)
        # selected gb stats
        # self.q1s, self.q3s = compute_stat(self.config, self.all_files)
        return self.q1s, self.q3s
    
    def prediction_segcount(self):
        # for fi
        pred_segcount(self.config, self.all_files, self.model)
        return 

    def chmF(self):
        for fullPath, filename in self.all_files:
            #print(filename)
            #print(fullPath)
            outputFile = os.path.join(self.config.output_dir, filename.replace(self.config.input_image_pref, self.config.output_prefix).replace(self.config.input_image_type, self.config.output_image_type))
            print(outputFile)
            chmF = '/mnt/ssdc/Finland/color2CHM/training/test_pred/2019_test_south/0929_1717_direct_apply_DKscale_upsample_rescale_inf_q20max/M3312A_det_CHM.tif'
            with rasterio.open(outputFile) as segf:
                seg = np.squeeze(segf.read())
                with rasterio.open(chmF) as chmf:
                    chmpred= np.squeeze(chmf.read())
                    chmgtF = '/mnt/ssdc/Finland/CHM/2019/CHM_M3312A_2019.tif'
                    with rasterio.open(chmgtF) as chmgtf:
                        chmgt = np.squeeze(chmgtf.read())
                        chm_seg(self.config, seg, chmpred, chmgt, chmf.meta, chmF.replace('M3312A', 'M3312A_Height_kernel20_s2'), chmF.replace('M3312A', 'M3312A_Height_kernel10_s2_diff'), th = 0.5)
                    
    def pred_segcount_dk_rawtifs(self):
        pred_segcount_dk(self.config, self.all_files, self.model)
        return 
        
def load_files(config):
    
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    print(all_files)
    return all_files


def compute_stat_gb(config, files, water_th = 100):
    # gb for all images to test
    merg1 = []
    merg2 = []
    merg3 = []

    for i in files:
        curf = i[0]
        # print(curf)
        with rasterio.open(curf) as src:
            
            gb_im = np.transpose(src.read(), axes=(1,2,0)) # channel last
            # gb_im = gb_im[:,:,config.channels] # swap channels, inf last
            # print(gb_im.shape)
            merg1.append(gb_im[:, :, 0]) # inf
            merg2.append(gb_im[:,:,1]) # green
            merg3.append(gb_im[:,:,2]) # blue
    # print(np.array(merg1).shape)
    
    q1s, q3s = [], []
    li = [merg2, merg3]
    mins = []
    maxs = []
    # for infrared, exclude values below 50
    merg1 = np.array(merg1).flatten()

    masked_b1 = merg1[merg1>50]
    # print(masked_b3.shape)
    q1 = np.quantile(masked_b1, 0.25)
    q3 = np.quantile(masked_b1, 0.75)
    q1s.append(q1)
    q3s.append(q3)
    mins.append(merg1.min())
    maxs.append(merg1.max())
    
    
    # print(np.array(merg1).shape)
    for i in range(len(li)):
        
        q1 = np.quantile(np.array(li[i]), 0.25, axis = (0,1,2))
        q3 = np.quantile(np.array(li[i]), 0.75, axis = (0,1,2))
        q1s.append(q1)
        q3s.append(q3)
        mins.append(np.array(li[i]).min())
        maxs.append(np.array(li[i]).max())
    
    # print(q1s, q3s)
    
    return np.array(q1s), np.array(q3s), np.array(mins), np.array(maxs)


        
# # Methods to add results of a patch to the total results of a larger area. The operator could be min (useful if there are too many false positives), max (useful for tackle false negatives)
# def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
#     currValue = res[row:row+he, col:col+wi]
#     # print('curcva', currValue.shape)
#     # print('pred', prediction.shape)
#     # print(he, wi)
#     newPredictions = prediction[:he, :wi]
#     # print('newshape', newPredictions.shape)
#     # print('prediction shape', prediction.shape)
#     # print('cur shape', currValue.shape)
#     # print('new pred shape', newPredictions.shape)
# # IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
#     if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
#         currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
#         resultant = np.minimum(currValue, newPredictions) 
#     elif operator == 'MAX':
#         resultant = np.maximum(currValue, newPredictions)
#     else: #operator == 'REPLACE':
#         resultant = newPredictions    
#     res[row:row+he, col:col+wi] =  resultant
#     return (res)

def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
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
        currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
        mm2 = (currValue==0)
        currValue[mm2] = newPredictions[mm2]
        resultant = currValue
    else: #operator == 'REPLACE':
        resultant = newPredictions    
    res[row:row+he, col:col+wi] =  resultant
    return (res)


# save to tif
def writeMaskToDisk_old(detected_mask, detected_meta, wp, image_type, output_shapefile_type, write_as_type = 'uint8', th = 0.5, create_countors = False):
    # Convert to correct required before writing
    if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
        print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
        # detected_mask[detected_mask<th]=0
        # detected_mask[detected_mask>=th]=1
        # detected_mask = detected_mask.astype(write_as_type)
        # detected_meta['dtype'] =  write_as_type
        # detected_meta['count'] = 1
        ##################################################################################################
        ##################################################################################################
        detected_mask[detected_mask<th]=0
        detected_mask[detected_mask>=th]=1
        
        detected_mask = detected_mask.astype(write_as_type)
        detected_meta['dtype'] =  write_as_type
        detected_meta['count'] = 1
        detected_meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 255
                            }
                        )
        # print(np.unique(detected_mask))
        
        
        
        
    with rasterio.open(wp, 'w', **detected_meta) as outds:
        outds.write(detected_mask.astype('uint8'), 1)
    if create_countors:
        wp = wp.replace(image_type, output_shapefile_type)
        create_contours_shapefile(detected_mask, detected_meta, wp)

# for segcount        
def writeMaskToDisk(detected_mask, detected_meta, wp,  image_type, output_shapefile_type, write_as_type = 'float32', th=0.5, create_countors = False, convert = 1):
    # Convert to correct required before writing
    # if 'float' in str(detected_meta['dtype']):
    meta = detected_meta.copy()
    if convert:
        print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
        detected_mask[detected_mask<th]=0
        detected_mask[detected_mask>=th]=1
        # detected_mask = detected_mask.astype(write_as_type)
        # detected_meta['dtype'] =  write_as_type
        # detected_meta['count'] = 1
        
    
    detected_mask = detected_mask.astype(write_as_type)
    # print(detected_mask.shape)
    if detected_mask.ndim != 2:
        detected_mask = detected_mask[0]
    # print(detected_mask.shape)
    meta['dtype'] =  write_as_type
    meta['count'] = 1
    meta.update(
                        {'compress':'lzw',
                          'driver': 'GTiff',
                            'nodata': 255,
                            # 'crs': dst_crs
                        }
                    )
        
    print(detected_mask.shape)
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
    
    return
        



#### try not downsave
def predict_using_model(model, batch, batch_pos, mask, operator, upsample = 1, downsave = 1):
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
        prediction = model.predict(tm1) # tm is a list []
        # print(prediction.shape)
        
        # display_images(np.concatenate((tm1, prediction), axis = -1))
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # print('p shape', p.shape)
        
        
        
        # print('mean before resize', p.mean(), p.std(), p.min(), p.max())
        ###############33#############################################
        if upsample and downsave:   # no preserve range
            # print('**********************UPsample for prediction***************************')
            print('******THIS IS INCORRECT****************Downsample for saving***********************')
            # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
            p = resize(p[:, :], (int(p.shape[0]/2), int(p.shape[1]/2)), preserve_range=True)
            # print('mean after resize', p.mean(), p.std(), p.min(), p.max())
        # print('p shape', p.shape)
        ###############33#############################################
        # print('p shape', p.shape)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        # print('p shape', p.shape)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask

# def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator, upsample = 1, downsave = 1):
#     # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
#     b1 = batch[0]
#     if len(b1) == 2: # 2 inputs
#         # print('2 inputs')
#         tm1 = []
#         tm2 = []
#         for p in batch:
#             tm1.append(p[0]) # (256, 256, 5)
#             tm2.append(p[1]) # (128, 128, 1)
#         tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
#         tm2 = np.stack(tm2, axis = 0)
#         seg, dens = model.predict([tm1, tm2]) # tm is a list []
        
#     else:
#         tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
#         seg, dens = model.predict(tm1) # tm is a list []
        
#     for i in range(len(batch_pos)):
#         (col, row, wi, he) = batch_pos[i]
#         p = np.squeeze(seg[i], axis = -1)
#         c = np.squeeze(dens[i], axis = -1)
        
#         if upsample and downsave:   # no preserve range
#             # print('**********************UPsample for prediction***************************')
#             # print('**********************Downsample for saving***********************')
#             # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
#             p = resize(p[:, :], (int(p.shape[0]/2), int(p.shape[1]/2)), preserve_range=True)
#             c = resize(c[:, :], (int(c.shape[0]/2), int(c.shape[1]/2)), preserve_range=True)
#             # rescale values
#             p = p * (p.shape[0] / float(p.shape[0]/2)) * (p.shape[1] / float(p.shape[1]/2))
#             c = c * (c.shape[0] / float(c.shape[0]/2)) * (c.shape[1] / float(c.shape[1]/2))

#         # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
#         maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
#         maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
#     return maskseg, maskdens

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

def predict_using_model_chm(model, batch, batch_pos, mask, operator, upsample = 1):
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
        prediction = model.predict(tm1) # tm is a list []
        # print(prediction.mean())
    
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask

#### try upsave instead of downsave
def detect_tree(config, model, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1, upsample = 1, downsave = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    
    if not singleRaster and auxData: # multi raster: core img and aux img in a list
    
        core_img = img[0]
        
        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()
        
        # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input
        
    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()
    
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    print('raw shape', meta['width'])
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    if downsave or not upsample:
        mask = np.zeros((nrows, nols), dtype=meta['dtype'])
        
    elif not downsave:
        mask = np.zeros((nrows*2, nols*2), dtype=meta['dtype'])
        meta.update(
                    {'width': int(nols*2),
                     'height': int(nrows*2)
                    }
                    )
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        # temp_im1 = img[0].read(window=window)
        
        if not singleRaster and auxData: 
            # TODO
            patch = np.zeros((height, width, meta['count']))
            
        else:
            if upsample:
                # print('upsample')
                # crop 128 and upsample to 256 for prediction
                patch = np.zeros((height*2, width*2, meta['count'])) #Add zero padding in case of corner images
                # temp_im = img.read(window=window)
                temp_im = img.read(
                                    out_shape=(
                                    img.count,
                                    int(window.height*2), # upsample by 2
                                    int(window.width*2)
                                ),
                                resampling=Resampling.bilinear, window = window)
            else:
                # no upsample
                patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
                # temp_im = img.read(window=window)
                temp_im = img.read(window = window)
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im = np.transpose(temp_im, axes=(1,2,0))
        
        
        # if normalize:
        #     # print('LOCAL NORMALIZE ************************************')
        #     #################################################
        #     ## norm DK
        #     # temp_im = (temp_im-np.array([168, 121,109]))/np.array([31,33,33])
            
        #     ###### norm 
        # r = np.random.random(1)
        # if normalize >= r[0]:
        #     # print('norm*******')
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        # else:
        #     print('no norm****')
        if normalize:
        #     # print('norm*******')
            # print('local normalization')
            temp_im = image_normalize(temp_im, axis=(0,1))
        
        if upsample:
            patch[:int(window.height*2), :int(window.width*2)] = temp_im
        else:
            patch[:int(window.height), :int(window.width)] = temp_im
        
        batch.append(patch)
        if downsave or not upsample:
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        elif not downsave:
            # print('upsave')
            batch_pos.append((window.col_off*2, window.row_off*2, window.width*2, window.height*2))
        
        ########################################################################
        # print((window.col_off, window.row_off, window.width, window.height))
        ########################################################################
        
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            
            mask = predict_using_model(model, batch, batch_pos, mask, 'MAX', upsample = upsample, downsave = downsave)
            
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        mask = predict_using_model(model, batch, batch_pos, mask, 'MAX', upsample = upsample, downsave = downsave)
        batch = []
        batch_pos = []
    return (mask, meta)

def detect_tree_segcount_save(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1, multires = 1, upsample = 1, downsave = 1):
    # for fi 3 bands
    if 'chm' in config.channel_names1:
        CHM = 1
    else:
        CHM = 0
    
    if not singleRaster and auxData: # multi raster: core img and aux img in a list

        core_img = img[0]

        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()

        # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input

    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()
    # img0 = img[0] # channel 0
    
    # read_img0 = img0.read()
    # print(read_img0.shape)
    # nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
    # nrows, nols = 256, 256 # base shape # rasterio read channel first
        
    # meta = img0.meta.copy()
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    
    if downsave or not upsample:
        masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
        maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)

    elif not downsave:
        masksegs = np.zeros((len(models), nrows*2, nols*2), dtype=np.float32)
        maskdenss = np.zeros((len(models), nrows*2, nols*2), dtype=np.float32)
        meta.update(
                    {'width': int(nols*2),
                     'height': int(nrows*2)
                    }
                    )
    
    
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        # temp_im1 = img[0].read(window=window)
        
        if CHM: 
            print('including CHM as input')
            ## TODO
            # if multires: # 2 inputs
            #     print('multires')
                
                # patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # # print('0 shape', temp_im1.shape)
            
                # for ch in range(1, len(img)-1): # except for the last channel 
                    
                #     imgi = rasterio.open(img[ch]) 
                    
                #     sm1 = imgi.read(
                #                     out_shape=(
                #                     1,
                #                     int(window.height),
                #                     int(window.width)
                #                 ),
                #                 resampling=Resampling.bilinear, window = window)
                #     # print('aux shape', aux_sm.shape)
                    
                #     temp_im1 = np.row_stack((temp_im1, sm1))
                
                # # for the 2nd input source
                # patch2 = np.zeros((int(height/2), int(width/2), 1)) # 128, 128, 1
                # temp_img2 = rasterio.open(img[-1]) #chm layer 
                # window2 = windows.Window(window.col_off / 2, window.row_off / 2,
                #                         int(window.width/2), int(window.height/2))
                
                # temp_im2 = temp_img2.read(
                #                     out_shape=(
                #                     temp_img2.count,
                #                     int(window2.height),
                #                     int(window2.width)
                #                 ),
                #                 resampling=Resampling.bilinear, window = window2)
                
                # temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            # elif not multires: # upsample CHM
            #     print('upsampling chm')
            #     patch1 = np.zeros((height, width, len(img))) # except for the last channel
                
            #     # print('0 shape', temp_im1.shape)
            
            #     for ch in range(1, len(img)-1): # except for the last channel 
                    
            #         imgi = rasterio.open(img[ch]) 
                    
            #         sm1 = imgi.read(
            #                         out_shape=(
            #                         1,
            #                         int(window.height),
            #                         int(window.width)
            #                     ),
            #                     resampling=Resampling.bilinear, window = window)
            #         # print('aux shape', aux_sm.shape)
                    
            #         temp_im1 = np.row_stack((temp_im1, sm1))
                
            #     # upsample the CHM channel
            #     chmim = rasterio.open(img[-1])
            #     meta_chm = chmim.meta.copy()
            #     hei_ratio = nrows/meta_chm['height']
            #     wid_ratio = nols/meta_chm['width']
            #     res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
            #                     window.width / wid_ratio, window.height / hei_ratio)
            
            #     chm_sm = chmim.read(
            #                     out_shape=(
            #                     chmim.count,
            #                     int(window.height),
            #                     int(window.width)
            #                 ),
            #                 resampling=Resampling.bilinear, window = res_window)
            #     # print('aux shape', aux_sm.shape)
            #     temp_im1 = np.row_stack((temp_im1, chm_sm))
            #     print(temp_im1.mean())
            
        elif not CHM:
            if upsample:
                # patch1 = np.zeros((height, width, len(img)))
                patch1 = np.zeros((height*2, width*2, meta['count']))
                
                temp_im1 = img.read(
                                    out_shape=(
                                    img.count,
                                    int(window.height*2), # upsample by 2
                                    int(window.width*2)
                                ),
                                resampling=Resampling.bilinear, window = window)
            else:
                # no upsample
                patch1 = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
                # temp_im = img.read(window=window)
                temp_im1 = img.read(window = window)
            
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            # rescale FI data first
            # print('rescaling', temp_im1.mean())
            # temp_im1 = np.array([139, 0, 0]) + ((temp_im1 - mins) / (maxs - mins)) * np.array([255-139, 255, 255])
            # print('rescaling 2', temp_im1.mean())

            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            # print('norm', temp_im1.mean())
            if CHM and multires:
                print('with CHM')
                ##TODO
                # temp_im2 = image_normalize(temp_im2, axis=(0,1))
        if upsample:
            patch1[:int(window.height*2), :int(window.width*2)] = temp_im1
        else:
            patch1[:int(window.height), :int(window.width)] = temp_im1
        
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            print('with CHM')
            ##TODO
            # if multires:
            #     patch2[:window2.height, :window2.width] = temp_im2
            #     batch.append([patch1, patch2])
            # else:
            #     batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
        # print('window colrow wi he', window.col_off, window.row_off, window.width, window.height)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if downsave or not upsample:
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        elif not downsave:
            # print('upsave')
            batch_pos.append((window.col_off*2, window.row_off*2, window.width*2, window.height*2))
        # plt.figure(figsize = (10,10))
        # plt.imshow(temp_im1[:,:,0])
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, 'MAX', upsample = upsample, downsave = downsave)
                
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmaskseg = masksegs[mi, :, :]
            curmaskdens = maskdenss[mi, :, :]
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, 'MAX', upsample = upsample, downsave = downsave)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta
        

def detect_tree_segcount_dk(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    # for dk all bands
    if not singleRaster and auxData: # multi raster: core img and aux img in a list
    
        core_img = img[0]
        
        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()
        
        aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input
        
    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()
        
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    masks_seg = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masks_dens = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        
         
        if not singleRaster and auxData: #multi rasters with possibly different resolutions resampled to the same resolution
            nc1 = meta['count'] + aux_channels1 #4+1 = 5 channels (normal input)
            # print('number of input channels:', nc)
            patch1 = np.zeros((height, width, nc1)) # 256, 256, 5
            temp_im1 = core_img.read(window=window)
            # print('col row', col_off, row_off)
            # print('core shape', core_sm.shape)
            # print('coloff', col_off)
            # print('rowoff', row_off)
            
            for aux in range(aux_channels1): # 0, 1 for two channels; 0 for one channel
                
                # print('Dealing with aux data', aux)
                aux_imgi = rasterio.open(img[aux+1]) #ndvi layer
                meta_auxi = aux_imgi.meta.copy()
                hei_ratio = nrows/meta_auxi['height']
                wid_ratio = nols/meta_auxi['width']
                # print('Handle resolution differences: resampling aux layer with a factor of', (hei_ratio, wid_ratio))
                
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                aux_sm1 = aux_imgi.read(
                                out_shape=(
                                aux_imgi.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, aux_sm1))
            
            # for the 2nd input source
            patch2 = np.zeros((int(height/2), int(width/2), 1)) # 128, 128, 1
            temp_img2 = rasterio.open(img[-1]) #chm layer 
            window2 = windows.Window(window.col_off / 2, window.row_off / 2,
                                    int(window.width/2), int(window.height/2))
            
            temp_im2 = temp_img2.read(
                                out_shape=(
                                temp_img2.count,
                                int(window2.height),
                                int(window2.width)
                            ),
                            resampling=Resampling.bilinear, window = window2)
        # else: #single rater or multi raster without aux # must be multi raster
        #     patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
        #     temp_im = img.read(window=window)
           
        ##########################################################################################
        #suqeeze or not? for one channel should not squeeze
        # temp_im = np.squeeze(temp_im)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        patch2[:window2.height, :window2.width] = temp_im2
        batch.append([patch1, patch2])
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask_seg = masks_seg[mi, :, :]
                curmask_dens = masks_dens[mi, :, :]
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
                
            batch = []
            batch_pos = []
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask_seg = masks_seg[mi, :, :]
            curmask_dens = masks_dens[mi, :, :]
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)
    

# input images are raw tif frames 4 bands with NDVI
def detect_tree_chm(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
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
    mask = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        #single rater or multi raster without aux # must be multi raster
        patch = np.zeros((height, width, len(channels)+1)) #Add zero padding in case of corner images
        read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels]
        # print(temp_im.shape)
        # print('red', temp_im[:, :, -1].max(), temp_im[:, :, -1].min())
        # print('red2', temp_im[:, :, 0].max(), temp_im[:, :, 0].min())
        # print(temp_im[:, :, -1] + temp_im[:, :, 0])
        # print(temp_im[:, :, -1] - temp_im[:, :, 0])
        NDVI = (temp_im[:, :, -1].astype(float) - temp_im[:, :, 0].astype(float)) / (temp_im[:, :, -1].astype(float) + temp_im[:, :, 0].astype(float))
        NDVI = NDVI[..., np.newaxis]
        # print('NDVI', NDVI.max(), NDVI.min())
        # print('bands', temp_im.max(), temp_im.min())
        temp_im = np.append(temp_im, NDVI, axis = -1)
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        temp_im = temp_im / 255
        temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
        
        patch[:window.height, :window.width] = temp_im
        batch.append(patch)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model(model, batch, batch_pos, mask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model(model, batch, batch_pos, mask, config.operator)
        batch = []
        batch_pos = []
        
    if maxnorm:
        mask = mask * 97.19
    return(mask, meta)


def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result


def pred_segcount(config, all_files, model):
    #segcount
    counts = {}
    th = 0.5
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
    
        with rasterio.open(fullPath) as img:
            
            if config.tasks == 2:
                detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_save(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=True, 
                                                                                            auxData = config.aux_data, singleRaster=config.single_raster, multires = config.multires, upsample = config.upsample, downsave = config.downsave)
                writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                
                # density
                writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                counts[filename] = detectedMaskDens.sum()
            elif config.tasks ==1:
                print('didnot check')
                detectedMask, detectedMeta = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 1, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
    
    
    print('Inference for the area has finished, saving counts to csv file')
    w = csv.writer(open(os.path.join(config.output_dir, "counts.csv"), "w"))
    for key, val in counts.items():
        w.writerow([key, val])
      

def pred_segcount_dk(config, all_files, model):
    # for dk all bands
    th = 0.5
    counts = {}
    outputFiles = []
    nochm = []
    waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
        #print(outputFile)
        outputFiles.append(outputFile)
        
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            if not config.single_raster and config.aux_data: # multi raster
                with rasterio.open(fullPath) as core:
                    auxx = []
                    # print(fullPath)
                    # print(fullPath.replace(config.input_image_pref, config.aux_prefs[0]).replace(config.input_tif_dir, config.input_ndvi_dir))
                    # ndvif0 = filename.replace(config.input_image_pref, config.aux_prefs[0])
                    
                    # ndvi_f = glob.glob(f"{config.input_ndvi_dir}/**/{ndvif0}")
                    ndvif0 = fullPath.replace(config.input_image_pref, config.aux_prefs[0]).replace(config.input_image_dir, config.input_ndvi_dir)
                    auxx.append(ndvif0)
                    coor = fullPath[-12:-9]+ fullPath[-8:-5]
                    chmpath = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                    # CHM_640_59_TIF_UTM32-ETRS89
                    aux2 = fullPath.replace(config.input_image_pref, config.aux_prefs[1]).replace(config.input_image_dir, chmpath)
                    auxx.append(aux2)
                    try:    
                        
                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_dk(config, model, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
    
                    except IOError:
                        
                        continue
                        
                        auxx[-1] = waterchm
                        nochm.append(outputFile)
                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_dk(config, model, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
    
                    ###### check threshold!!!!
                    # seg
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    # print(detectedMaskDens.shape)
                    # print(detectedMaskDens.max(), detectedMaskDens.min())
                    # print(detectedMaskDens.sum())
                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()
            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as img:
                    #print(fullPath)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_dk(img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    
        else:
            print('File already analysed!', fullPath)
    
    
    print('Inference for the area has finished, saving counts to csv file')
    
    w = csv.writer(open(os.path.join(config.output_dir, "counts.csv"), "w"))
    for key, val in counts.items():
        w.writerow([key, val])
    
    
    return
        
      
def chm_seg(config, segmask, chmpred, chmgt, meta, outputFile, outputDiff, th = 0.5):
    # mask height with seg masks
    # # return list of dicts
    # stat = zonal_stats("polygons.shp", "elevation.tif",
    #         stats="count min mean max median")
    segmask[segmask<th]=0
    segmask[segmask>=th]=1
    
    # rescale chm values pred
    chmpred = chmpred/100
    
    mask1 = segmask * chmpred
    # print(segmask.shape)
    # mask2 = poolingOverlap(mask1,(3,3),(2,2),method='max',pad=1)
    # mask2 = poolingOverlap(mask1,(5,5),(2,2),method='max',pad=1) # 1m pixel resolution
    # mask2 = poolingOverlap(mask1,(10,10),(4,4),method='max',pad=1) # 1m pixel resolution
    # mask2 = poolingOverlap(mask1,(20,20),(2,2),method='max',pad=1) # 1m pixel resolution
    mask2 = poolingOverlap(mask1,(30,30),(2,2),method='max',pad=1) # 1m pixel resolution
    # print(mask2.shape)
    segmask2 = poolingOverlap(segmask,(2,2),(2,2),method='mean',pad=1)
    # segmask2 = poolingOverlap(segmask,(4,4),(4,4),method='mean',pad=1)
    maskpred_res = mask2*segmask2
    # for computing errors
    maskpred_resP = poolingOverlap(maskpred_res,(2,2),(2,2),method='mean',pad=1)
    # maskpred_resP = poolingOverlap(maskpred_res,(2,2),(2,2),method='max',pad=1)
    # print(maskpred_res.shape)
    
    # print(chmgt.shape)
    maskgt1 = segmask2 * chmgt
    # maskgt2 = poolingOverlap(maskgt1,(10,10),(2,2),method='max',pad=1) # 1m pixel resolution
    maskgt2 = poolingOverlap(maskgt1,(15,15),(2,2),method='max',pad=1) # 1m pixel resolution
    segmask3 = poolingOverlap(segmask2,(2,2),(2,2),method='mean',pad=1)
    maskgt_res = maskgt2*segmask3
    
    h = tf.keras.losses.Huber()
    huberLoss = h(maskgt_res, maskpred_resP).numpy()
    mae = tf.keras.losses.MeanAbsoluteError()
    MaeLoss = mae(maskgt_res, maskpred_resP).numpy()
    mse = tf.keras.losses.MeanSquaredError()
    MseLoss = mse(maskgt_res, maskpred_resP).numpy()
    
    print('*******************Huber loss: {}******************'.format(huberLoss))
    print('*******************MSE loss: {}******************'.format(MseLoss))
    print('*******************MAE loss: {}******************'.format(MaeLoss))
    
    writeMaskToDisk(maskpred_res, meta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
    
    diff = maskgt_res - maskpred_resP # 2m pixel resolution
    writeMaskToDisk(diff, meta, outputDiff, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
    
    # errors
    CHMerror(maskpred_resP, maskgt_res,  nbins = 100)

    
    return mask2
    


def CHMerror(pr, gt_mask_im,  nbins = 100):
    pr = pr.flatten()
    gt_mask_im = gt_mask_im.flatten()
    MAE = mean_absolute_error(gt_mask_im, pr)
    print('MAE for all', MAE)
    MEAE = median_absolute_error(gt_mask_im, pr)
    print('Median absolute error', MEAE)
    # plt.figure()
    # plt.hist2d(pr, gt_mask_im, cmap='Blues', bins=50, density =  1, norm=colors.LogNorm(), vmin = 0.0001)
    # plt.colorbar()
    # plt.xlim(0, 40)
    # plt.ylim(0, 40)
    # plt.plot([0, 40], [0, 40], alpha = 0.7, lw = 0.2, c = 'black', ls = '--')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('CHM ground truth vs estimation per pixel')
    # plt.xlabel('Estimation (m)')
    # plt.ylabel('Ground truth (m)')
    
    gtm = int(np.ceil(gt_mask_im.max()))
    xx = [0, gtm]
    
    inds = []
    intervals = [0, 10, 20, 30, gtm]
    for i in range(4):
        indi = [idx for idx,val in enumerate(gt_mask_im) if intervals[i] <= val < intervals[i+1]]
        inds.append(indi)
    
    
    preds = []
    gtts = []
    for i in range(4):
        predi = pr[inds[i]]
        preds.append(predi)
        gtti = gt_mask_im[inds[i]]
        gtts.append(gtti)

    maes = []
    
    for i in range(4):
        
        maes.append(abs(gtts[i] - preds[i]))

    # fig, ax = plt.subplots(figsize=(9,7))
    # ax.boxplot(maes, showfliers=False, showmeans = True)
    # ax.set_title('CHM error per pixel - interval errors', fontsize = 16)
    # labels = ['0-10m', '10-20m', '20-30m', '>30m']
    # ax.set_xticklabels(labels)
    # ax.set_xlabel('Height', fontsize = 14)
    # ax.set_ylabel('MAE', fontsize = 14)
    
    # ax.grid(True, axis = 'y', alpha = 0.3)
    def func(x, a, b):
        return a * x + b 
        # return a * x + b
    
    popt, pcov = curve_fit(func, gt_mask_im, pr)

    
    fig = plt.figure(figsize = (15, 15))
    gs = GridSpec(6, 6)
    ax_scatter = fig.add_subplot(gs[1:5, 0:5])
    ax_hist_y = fig.add_subplot(gs[0,0:5])
    ax_hist_x = fig.add_subplot(gs[1:5, 5])
    ax_box_x = fig.add_subplot(gs[5, 0:5])
    
    # ax_scatter.scatter(gtt, ppd, s =1, c = 'grey')
    ax_scatter.hist2d(gt_mask_im, pr, cmap='Blues', alpha = 0.8, bins=50, density =  1, norm=colors.LogNorm(), vmin = 0.000005)
    ax_hist_y.hist(gt_mask_im, bins=80, color='navy', alpha=0.3, density = 1)
    ax_hist_y.set_ylim(0, 0.08)
    ax_hist_x.hist(pr, bins=80, color='navy', alpha=0.3, density = 1, orientation = 'horizontal')
    ax_hist_x.set_xlim(0, 0.08)
    ax_scatter.plot(xx, xx, c ='grey', alpha = 0.5)
    slope, intercept, r_value, p_value, std_err = linregress(gt_mask_im, pr)
    ax_scatter.plot(xx, func(np.array(xx), *popt), 'b--', alpha = 0.5, label='f(x) = %5.3f x + %5.3f; r2 = %5.3f ' % (popt[0], popt[1], r_value**2))
    ax_scatter.set_aspect('equal')
    ax_scatter.set_xlim(0, gtm)
    ax_scatter.set_ylim(0, gtm)
    ax_hist_y.set_xlim(0, gtm)
    ax_hist_x.set_ylim(0, gtm)
    # ax_scatter.set_xlabel('Reference height (m)', fontsize = 14)
    # ax_scatter.set_ylabel('Estimated height (m)', fontsize = 14)
    # plt.colorbar(ax = ax_scatter)
    ax_scatter.legend(prop={'size': 12})
    
    labels = ['0-10m', '10-20m', '20-30m', '>30m']
    # boxprops = dict(linestyle='--')
    line_props = dict(linestyle='--', color="grey", alpha=0.9)
    capprops = dict(linewidth = 1, color="grey", alpha = 0.9)
    bp = ax_box_x.boxplot(maes, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    ax_box_x.set_xticklabels(labels)
    ax_box_x.set_xlabel('Height')
    ax_box_x.set_ylabel('MAE (m)')
    
    med = []
    mu = []
    for t in maes:
        med.append(np.median(t))
        mu.append(np.mean(t))
    
    # fig, ax = plt.subplots()
    # bp = ax.boxplot(data3, showmeans=True)
    
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med[i], mu[i])
        ax_box_x.annotate(text, xy=(x, y-1))
    
    ax_box_x.grid(True, axis = 'y', alpha = 0.3)
    
    plt.show()

    
    return 
    
   
    
def pred_normal(config, all_files, model):
    """an old version"""
    # normal prediction 
    # NOTE: check upsample, normalization (local) --- yes local normalize by default
    outputFiles = []
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace('.jp2', '.tif'))
        print(outputFile)
        outputFiles.append(outputFile)
        
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            if not config.single_raster and config.aux_data: # multi raster
                with rasterio.open(fullPath) as core:
                    auxx = []
                    for i in range(2):
                        auxx.append(fullPath.replace(config.input_image_pref, config.aux_prefs[i]))
                            # with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[0])) as aux0:
                            #     with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[1])) as aux1:
                            #     # in this case the resolution of the chm tif is only half of the raw tif, thus upsampling the aux tif
                    print(auxx)
                    detectedMask, detectedMeta = detect_tree(config, model, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    #Write the mask to file
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as img:
                    
                    
                    
                    detectedMask, detectedMeta = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
        else:
            print('File already analysed!', fullPath)
    
    
    mergePredTiff(detectedMeta, outputFiles, config.output_dir, config.output_prefix, config.output_image_type, config.output_shapefile_type, config.output_dtype)
    return 


def pred_DK_norm(config, all_files, model):
    # try global normalize with DK statis 
    # NOTE: (local norm or not)
    
    # use no sea norm or with sea norm        
    sea = 0 #(1 for 84 frames, 0 for 77 frames )
    NIR = 0 # (1 use NIRGB statics, 0 use RGB statics)
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        # outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        # replace jp2 with tif
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace('.jp2', '.tif'))
        # print(outputFile)
        with rasterio.open(fullPath) as img:
            imm = img.read()
            print(imm.shape, imm.mean())
            # imm = np.transpose(imm, axes = (1, 2, 0))
            # print(imm.shape)
            if not sea:
                print('Using norm and std computed without sea areas')
                if NIR: # use NIRGB
                    print('Using NIRGB statics, for FI')
                    dmean = np.array([[[168]], [[121]], [[109]]])
                    dstd = np.array([[[31]], [[33]], [[33]]])
                else:
                    print('Using RGB statics, for FR')
                    dmean = np.array([[[118]], [[121]], [[109]]])
                    dstd = np.array([[[39]], [[33]], [[33]]])
                imm_nor = (imm-dmean)/dstd # statis of DK
                print(imm_nor.mean((1,2)))
            elif sea:
                print('Using norm and std computed with sea areas')
                if NIR:
                    print('Using NIRGB statics, for FI')
                    dmean = np.array([[[155]], [[116]], [[106]]])
                    dstd = np.array([[[30]], [[31]], [[31]]])
                else:
                    print('Using RGB statics, for FR')
                    dmean = np.array([[[111]], [[116]], [[106]]])
                    dstd = np.array([[[37]], [[31]], [[31]]])
                
                imm_nor = (imm-dmean)/dstd # statis of DK, with sea
                
                print(imm_nor.mean((1,2)))
            meta = img.profile.copy()
            meta.update({'dtype':np.float64, 'driver':'GTiff'})
            with MemoryFile() as memfile:
                
                with memfile.open(**meta) as dataset:
                    dataset.write(imm_nor)
                    del imm_nor
                with memfile.open() as dataset:
                    # da = dataset.read()
                    # print(da.shape, da.mean((1,2)))
                    # detectedMask, detectedMeta = detect_tree(config, model, dataset, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 1, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                    detectedMask, detectedMeta = detect_tree(config, model, dataset, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 0, auxData = config.aux_data, singleRaster=config.single_raster, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
    return 


def pred_local_norm(config, all_files, model):
    
    # no global norm, but local norm
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
    
        with rasterio.open(fullPath) as img:
            # imm = img.read()
            # print(imm.shape, imm.mean())
            # # imm = np.transpose(imm, axes = (1, 2, 0))
            # # print(imm.shape)
            # imm_nor = (imm-np.array([[[168]], [[121]], [[109]]]))/np.array([[[31]], [[33]], [[33]]]) # statis of DK
            # print(imm_nor.mean((1,2)))
            # meta = img.profile.copy()
            # meta.update({'dtype':np.float64, 'driver':'GTiff'})
            # with MemoryFile() as memfile:
                
            #     with memfile.open(**meta) as dataset:
            #         dataset.write(imm_nor)
            #         del imm_nor
            #     with memfile.open() as dataset:
                    # da = dataset.read()
                    # print(da.shape, da.mean((1,2)))
            detectedMask, detectedMeta = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 1, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
    
    return


  
def pred_shrink(config, all_files, model):
    notwork = []
    # 128 input shape
    # save as tif with lzw compress
    for fullPath, filename in all_files:
        
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace('.jp2', '.tif'))
        print(outputFile)
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            try:
                if not config.single_raster and config.aux_data: # multi raster
                    with rasterio.open(fullPath) as core:
                        #TODO
                        auxx = []
                        # for i in range(2):
                        #     auxx.append(fullPath.replace(config.input_image_pref, config.aux_prefs[i]))
                        #         # with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[0])) as aux0:
                        #         #     with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[1])) as aux1:
                        #         #     # in this case the resolution of the chm tif is only half of the raw tif, thus upsampling the aux tif
                        print(auxx)
                        # detectedMask, detectedMeta = detect_tree(config, model, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        # #Write the mask to file
                        # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                else: # single raster or multi raster without aux
                    print('Single raster or multi raster without aux')
                    with rasterio.open(fullPath) as img:
                        
                        detectedMask, detectedMeta = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 1,auxData = config.aux_data, singleRaster=config.single_raster, upsample = config.upsample) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
            except:
                notwork.append(outputFile)
        else:
            print('File already analysed!', fullPath)
    
    return
    




def mergePredTiff(meta, outputFiles, output_dir, output_prefix, output_image_type, output_shapefile_type, output_dtype):
    """Merge predictions in one single raster tif
    
    meta: ini meta profile from an individual raster
    outputFiles: list of individual prediction tif files
    
    """
    merged, mergedTrans = merge.merge(outputFiles)
    merged = np.squeeze(merged)
    mergedMeta = meta.copy()
    mergedMeta.update({'width':merged.shape[1], 'height': merged.shape[0], 'transform': mergedTrans})
    # if 'compress' in mergedMeta:
    #     del mergedMeta['compress']
    mergedFn = os.path.join(output_dir, output_prefix+'merged.jp2')
    writeMaskToDisk(merged, mergedMeta, mergedFn, image_type = output_image_type, output_shapefile_type = output_shapefile_type, write_as_type = output_dtype, th = 0.5, create_countors = False)            
    print('All predictions merged to a single raster!')
    return




schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'str', 'canopy': 'float:15.2',},
    }

def drawPolygons(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=255, fill=255)
    mask = np.array(mask)#, dtype=bool)   
    return(mask)

def transformToXY(polygons, transform):
    tp = []
    for polygon in polygons:
        rows, cols = zip(*polygon)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tp.append(list(zip(x,y)))
    return (tp)

def createShapefileObject(polygons, meta, wfile):
    with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
        for idx, mp in tqdm(enumerate(polygons)):
            try:
#                 poly = Polygon(poly)
    #             assert mp.is_valid
    #             assert mp.geom_type == 'Polygon'
                sink.write({
                    'geometry': mapping(mp),
                    'properties': {'id': str(idx), 'canopy': mp.area},
                })
            except:
                print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
#                 print(mp)

# Generate a mask with polygons
def transformContoursToXY(contours, transform = None):
    tp = []
    for cnt in contours:
        pl = cnt[:, 0, :]
        cols, rows = zip(*pl)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tl = [list(i) for i in zip(x, y)]
        tp.append(tl)
    return (tp)


def mask_to_polygons(maskF, transform):
    # first, find contours with cv2: it's much faster than shapely
    th = 0.5
    mask = maskF.copy()
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask = ((mask) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #Convert contours from image coordinate to xy coordinate
    contours = transformContoursToXY(contours, transform)
    if not contours: #TODO: Raise an error maybe
        print('Warning: No contours/polygons detected!!')
        return [Polygon()]
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []

    for idx, cnt in enumerate(contours):
        if idx not in child_contours: #and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
            try:
                poly = Polygon(
                    shell=cnt,
                    holes=[c for c in cnt_children.get(idx, [])])
                           #if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                all_polygons.append(poly)
            except:
                pass
#                 print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
    print(len(all_polygons))
    return(all_polygons)

def create_contours_shapefile(mask, meta, out_fn):
    res = mask_to_polygons(mask, meta['transform'])
#     res = transformToXY(contours, meta['transform'])
    createShapefileObject(res, meta, out_fn)



###### to deal with only weights saved
def expand_model_simple(loaded_model, newInputShape=(128, 128, 3),  oldInputShape = (None, 256, 256, 64), freeze = 1):
    """
    model 0 : old model (256, 256, 3)
    
    new input shape: (128, 128, 3)
    
    no 3 3 conv, only to learn the upsample and downsample weights
    
    freeze all pretrained layers
    
    """
    if freeze:
        loaded_model.trainable = False
    
    loaded_model.layers[0]._batch_input_shape = oldInputShape
    new_model = tf.keras.models.model_from_json(loaded_model.to_json()) 
    # new_model.summary()
    
    model4 = tf.keras.models.Model(new_model.input, new_model.layers[-2].output)
    # model4.summary()
    
    inputs = tf.keras.Input(shape=newInputShape)  # layer 0 is the input layer, which we're replacing
    ly = tf.keras.layers.Conv2D(64, (1, 1), activation='elu', padding='same')(inputs)
    lyc = tf.keras.layers.BatchNormalization()(ly)
    ly = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(lyc)
    
    xx = model4(ly)
    m5 = tf.keras.models.Model(inputs, xx)
    # m5.summary()
    
    
    l = tf.keras.layers.Conv2D(64, (1, 1), activation='elu', padding='same', strides = 2)(m5.output)
    l = tf.keras.layers.BatchNormalization()(l)
    # l = tf.keras.layers.concatenate([lyc, l], axis=3)
    # l = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='same')(l)
    # l = tf.keras.layers.BatchNormalization()(l)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(l)
    model6 = tf.keras.models.Model(m5.input, outputs)
    print(model6.summary())
    return model6



# def predict_using_model(model, batch, batch_pos, mask, operator, upsample = 1):
#     # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    
#     b1 = batch[0]
#     if len(b1) == 2: # 2 inputs
#         print('2 inputs')
#         tm1 = []
#         tm2 = []
#         for p in batch:
#             tm1.append(p[0]) # (256, 256, 5)
#             tm2.append(p[1]) # (128, 128, 1)
#         tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
#         tm2 = np.stack(tm2, axis = 0)
#         prediction = model.predict([tm1, tm2]) # tm is a list []
        
#     else:
#         tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
#         prediction = model.predict(tm1) # tm is a list []
        
#         # display_images(np.concatenate((tm1, prediction), axis = -1))
#     for i in range(len(batch_pos)):
#         (col, row, wi, he) = batch_pos[i]
#         p = np.squeeze(prediction[i], axis = -1)
#         # print('p shape', p.shape)
        
        
        
#         # print('mean before resize', p.mean(), p.std(), p.min(), p.max())
#         ###############33#############################################
#         if upsample:   # no preserve range
#             # print('**********************UPsample for prediction***************************')
#             print('**********************Downsample for saving***********************')
#             # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
#             p = resize(p[:, :], (int(p.shape[0]/2), int(p.shape[1]/2)), preserve_range=True)
#             # print('mean after resize', p.mean(), p.std(), p.min(), p.max())
#         # print('p shape', p.shape)
#         ###############33#############################################
#         # print('p shape', p.shape)
#         # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
#         mask = addTOResult(mask, p, row, col, he, wi, operator)
#     return mask


# works with upsampling inputs
# def detect_tree(config, model, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1, upsample = 1):
#     """img can be one single raster or multi rasters
    
#     img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
#     or
    
#     img = img #single raster
    
    
#     """
    
#     if not singleRaster and auxData: # multi raster: core img and aux img in a list
    
#         core_img = img[0]
        
#         nols, nrows = core_img.meta['width'], core_img.meta['height']
#         meta = core_img.meta.copy()
        
#         # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input
        
#     else: #single rater or multi raster without aux
#         nols, nrows = img.meta['width'], img.meta['height']
#         meta = img.meta.copy()
        
#     if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
#         meta['dtype'] = np.float32
    
#     print('w', meta['width'])
#     offsets = product(range(0, nols, stride), range(0, nrows, stride))
#     # print(offsets)
#     big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
# #     print(nrows, nols)

#     mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    
#     batch = []
#     batch_pos = [ ]
#     for col_off, row_off in tqdm(offsets):
#         window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
#         # temp_im1 = img[0].read(window=window)
        
#         if not singleRaster and auxData: 
#             # TODO
#             patch = np.zeros((height, width, meta['count']))
            
#         else:
#             if upsample:
#                 # print('upsample')
#                 # crop 128 and upsample to 256 for prediction
#                 patch = np.zeros((height*2, width*2, meta['count'])) #Add zero padding in case of corner images
#                 # temp_im = img.read(window=window)
#                 temp_im = img.read(
#                                     out_shape=(
#                                     img.count,
#                                     int(window.height*2), # upsample by 2
#                                     int(window.width*2)
#                                 ),
#                                 resampling=Resampling.bilinear, window = window)
#             else:
#                 # no upsample
#                 patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
#                 # temp_im = img.read(window=window)
#                 temp_im = img.read(window = window)
#         # stack makes channel first
#         # print('rstack tt shape', temp_im1.shape)
#         temp_im = np.transpose(temp_im, axes=(1,2,0))
        
        
#         # if normalize:
#         #     # print('LOCAL NORMALIZE ************************************')
#         #     #################################################
#         #     ## norm DK
#         #     # temp_im = (temp_im-np.array([168, 121,109]))/np.array([31,33,33])
            
#         #     ###### norm 
#         # r = np.random.random(1)
#         # if normalize >= r[0]:
#         #     # print('norm*******')
#         #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
#         # else:
#         #     print('no norm****')
#         if normalize:
#         #     # print('norm*******')
#             # print('local normalization')
#             temp_im = image_normalize(temp_im, axis=(0,1))
        
#         if upsample:
#             patch[:int(window.height*2), :int(window.width*2)] = temp_im
#         else:
#             patch[:int(window.height), :int(window.width)] = temp_im
        
#         batch.append(patch)
            
#         batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        
#         ########################################################################
#         # print((window.col_off, window.row_off, window.width, window.height))
#         ########################################################################
        
#         if (len(batch) == config.BATCH_SIZE):
#             # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#             # batch = []
#             # batch_pos = []
            
#             mask = predict_using_model(model, batch, batch_pos, mask, 'MAX', upsample = upsample)
            
#             batch = []
#             batch_pos = []
            
#     # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
#     if batch:
#         # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#         # batch = []
#         # batch_pos = []
#         mask = predict_using_model(model, batch, batch_pos, mask, 'MAX', upsample = upsample)
#         batch = []
#         batch_pos = []
#     return (mask, meta)




# with resized modelinput
# def detect_tree(config, model, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
#     """img can be one single raster or multi rasters
    
#     img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
#     or
    
#     img = img #single raster
    
    
#     """
    
#     if not singleRaster and auxData: # multi raster: core img and aux img in a list
    
#         core_img = img[0]
        
#         nols, nrows = core_img.meta['width'], core_img.meta['height']
#         meta = core_img.meta.copy()
        
#         # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input
        
#     else: #single rater or multi raster without aux
#         # nols, nrows = img.meta['width'], img.meta['height']
#         # meta = img.meta.copy()
#     ##############################################################################3
#     ##############################################################################3
#         nols, nrows = int(img.meta['width']*2), int(img.meta['height']*2)
#         meta = img.meta.copy()
#         ##############################################################################3
#     ##############################################################################3
#     if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
#         meta['dtype'] = np.float32
        
#     offsets = product(range(0, nols, stride), range(0, nrows, stride))
#     # print(offsets)
#     big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
# #     print(nrows, nols)

#     mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    
#     batch = []
#     batch_pos = [ ]
#     for col_off, row_off in tqdm(offsets):
#         window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
#         # temp_im1 = img[0].read(window=window)
        
#         if not singleRaster and auxData: 
#             # TODO
#             patch = np.zeros((height, width, meta['count']))
            
#         else:
            
#             patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
#             # temp_im = img.read(window=window)
            
#             # temp_im = img.read(
#             #                     out_shape=(
#             #                     img.count,
#             #                     int(window.height),
#             #                     int(window.width)
#             #                 ),
#             #                 resampling=Resampling.bilinear, window = window)
            
#             ##################################################################
#             ##################################################################
#             temp_im = img.read(
#                                 out_shape=(
#                                 img.count,
#                                 int(window.height),
#                                 int(window.width)
#                             ),
#                             resampling=Resampling.bilinear, window = window)
            
            
            
#             ##################################################################
#             ##################################################################
#         # stack makes channel first
#         # print('rstack tt shape', temp_im1.shape)
#         temp_im = np.transpose(temp_im, axes=(1,2,0))
        
#         # print(temp_im1.shape, temp_im2.shape)
        
#         if normalize:
#             temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            
        
#         patch[:int(window.height), :int(window.width)] = temp_im
        
#         batch.append(patch)
            
#         batch_pos.append((window.col_off, window.row_off, window.width, window.height))
#         if (len(batch) == config.BATCH_SIZE):
#             # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#             # batch = []
#             # batch_pos = []
#             mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#             batch = []
#             batch_pos = []
            
#     # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
#     if batch:
#         # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#         # batch = []
#         # batch_pos = []
#         mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#         batch = []
#         batch_pos = []
#     return (mask, meta)


# trying with fine tune model
# def detect_tree(config, model, img, width=256, height=256, stride = 224, normalize=True, auxData = 0, singleRaster = 1):
#     """img can be one single raster or multi rasters
    
#     img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
#     or
    
#     img = img #single raster
    
    
#     """
    
#     if not singleRaster and auxData: # multi raster: core img and aux img in a list
    
#         core_img = img[0]
        
#         nols, nrows = core_img.meta['width'], core_img.meta['height']
#         meta = core_img.meta.copy()
        
#         # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input
        
#     else: #single rater or multi raster without aux
#         # nols, nrows = img.meta['width'], img.meta['height']
#         # meta = img.meta.copy()
#     ##############################################################################3
#     ##############################################################################3
#         nols, nrows = int(img.meta['width']), int(img.meta['height'])
#         print('nols', nols)
#         meta = img.meta.copy()
#         ##############################################################################3
#     ##############################################################################3
#     if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
#         meta['dtype'] = np.float32
        
#     offsets = product(range(0, nols, stride), range(0, nrows, stride))
#     # print(offsets)
#     big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
# #     print(nrows, nols)

#     mask = np.zeros((int(nrows*2), int(nols*2)), dtype=meta['dtype'])
    
#     batch = []
#     batch_pos = [ ]
#     for col_off, row_off in tqdm(offsets):
#         window =windows.Window(col_off=col_off, row_off=row_off, width=128, height=128).intersection(big_window)
        
#         # temp_im1 = img[0].read(window=window)
        
#         if not singleRaster and auxData: 
#             # TODO
#             patch = np.zeros((height, width, meta['count']))
            
#         else:
            
#             patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
#             # print('patch', patch.shape)
#             # temp_im = img.read(window=window)
            
#             # temp_im = img.read(
#             #                     out_shape=(
#             #                     img.count,
#             #                     int(window.height),
#             #                     int(window.width)
#             #                 ),
#             #                 resampling=Resampling.bilinear, window = window)
            
#             ##################################################################
#             ##################################################################
#             temp_im = img.read(
#                                 out_shape=(
#                                 img.count,
#                                 int(window.height*2),
#                                 int(window.width*2)
#                             ),
#                             resampling=Resampling.bilinear, window = window)
            
            
            
#             ##################################################################
#             ##################################################################
#         # stack makes channel first
#         # print('rstack tt shape', temp_im1.shape)
#         temp_im = np.transpose(temp_im, axes=(1,2,0))
        
#         # print(temp_im1.shape, temp_im2.shape)
        
#         if normalize:
#             temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            
#         print('temp', temp_im.shape)
#         patch[:int(window.height*2), :int(window.width*2)] = temp_im
#         print('patch', patch.shape)
#         batch.append(patch)
            
#         batch_pos.append((window.col_off, window.row_off, window.width*2, window.height*2))
#         if (len(batch) == config.BATCH_SIZE):
#             # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#             # batch = []
#             # batch_pos = []
#             mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#             batch = []
#             batch_pos = []
            
#     # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
#     if batch:
#         # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#         # batch = []
#         # batch_pos = []
#         mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#         batch = []
#         batch_pos = []
#     return (mask, meta)


# model = UNet([config.BATCH_SIZE, 256, 256, 3],[3])
# model.summary()


# model = expand_model_simple(model, freeze = 1)
# model.trainable = True
# model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity, miou, weight_miou])
# model.load_weights(config.trained_model_path)


# model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity, miou, weight_miou])


# model.save('./saved_models/UNet/testsave.h5')

# tt = load_model('./saved_models/UNet/testsave.h5', custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
# tt.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])

#############################################
# try changing the input shape directly and then predict
# model.summary()
# model_config = model.get_config()
# # model_config['layers'][0]['config']['batch_input_shape'] = (None, 102, 102, 3)
# # model.layer._batch_input_size      
# model.layers[0]._batch_input_shape = (None, 128, 128, 3)

# new_model = model_from_json(model.to_json()) 
# new_model.summary()

# # copy weights from old model to new one
# for layer in new_model.layers:
#     try:
#         layer.set_weights(model.get_layer(name=layer.name).get_weights())
#     except:
#         print("Could not transfer weights for layer {}".format(layer.name))



# ###### increase resolution

# from rasterio.enums import Resampling

# upscale_factor = 1/2

# with rasterio.open(outputFile) as dataset:
#     print(dataset.height)
#     print(dataset.height * upscale_factor)
#     # resample data to target shape
#     data = dataset.read(
#         out_shape=(
#             dataset.count,
#             int(dataset.height * upscale_factor),
#             int(dataset.width * upscale_factor)
#         ),
#         resampling=Resampling.bilinear
#     )

#     # scale image transform
#     transform = dataset.transform * dataset.transform.scale(
#         (dataset.width / data.shape[-1]),
#         (dataset.height / data.shape[-2])
#         )
    
#     print(data.shape)
#     pf = dataset.profile
#     pf['width'] = int(dataset.width * upscale_factor)
#     pf['height'] = int(dataset.height * upscale_factor) 
#     pf['transform'] = transform
    
#     with rasterio.open('/mnt/ssdc/Finland/prep_test/pred_mara0220L_03080213_complex1_w3_finetuneFIDK_upsave/example.tif', 'w', **pf) as dst:
#         dst.write(data)


# ##########################################################
# # try print out prediction directly
# outputFiles = []
# masks = []
# for fullPath, filename in all_files:
#     #print(filename)
#     #print(fullPath)
#     outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
#     #print(outputFile)
#     outputFiles.append(outputFile)
    
#     if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
        
#         if not config.single_raster and config.aux_data: # multi raster
#             with rasterio.open(fullPath) as core:
#                 auxx = []
#                 for i in range(2):
#                     auxx.append(fullPath.replace(config.input_image_pref, config.aux_prefs[i]))
#                         # with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[0])) as aux0:
#                         #     with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[1])) as aux1:
#                         #     # in this case the resolution of the chm tif is only half of the raw tif, thus upsampling the aux tif
#                 print(auxx)
#                 detectedMask, detectedMeta = detect_tree(config, model, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
#                 #Write the mask to file
#                 masks.append(detectedMask)
#                 # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
#         else: # single raster or multi raster without aux
#             print('Single raster or multi raster without aux')
#             with rasterio.open(fullPath) as img:
                
                
                
#                 detectedMask, detectedMeta = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
#                 masks.append(detectedMask)
#                 # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
#     else:
#         print('File already analysed!', fullPath)




# base = '/mnt/ssdc/Finland/prep_test/extracted/'
# all_files = os.listdir(base)
# # image channel 1
# all_files_c1 = [fn for fn in all_files if fn.startswith('infrared') and fn.endswith('png')]
# print(all_files_c1)


# from core.frame_info import FrameInfo

# extracted_filenames = ['infrared', 'green', 'blue']

# frames = []
# for i, fn in enumerate(all_files_c1):
#     # loop through rectangles
#     comb_img = rasterio.open(os.path.join(base, fn)).read()
#     # print('bef', np.min(comb_img), np.mean(comb_img), np.max(comb_img))
#     # print(comb_img.shape)
#     if config.upsample:
#         print('//////UPsampling images///////')
#         comb_img = resize(comb_img[:, :, :], (1, int(comb_img.shape[1]*2), int(comb_img.shape[2]*2)), preserve_range = 1)
#     # print('aft', np.min(comb_img))
#     # print(comb_img.shape)
    
#         # print('If single raster or multi raster without aux')
#     for c in range(1, 3):
#         #loop through raw channels
#         # comb_img = np.append(comb_img, rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[c+1]))).read(), axis = 0)
#         cur = rasterio.open(os.path.join(base, fn.replace('infrared', extracted_filenames[c]))).read()
#         # print('bef', np.min(cur), np.mean(cur))
#         if config.upsample:
#             cur = resize(cur[:, :, :], (1, int(cur.shape[1]*2), int(cur.shape[2]*2)), preserve_range = 1)
#         # print('aft', np.min(cur))
#         comb_img = np.append(comb_img, cur, axis = 0)
        
    
#         #for aux chm channel, upsample by 2 
#         # 0.4m resolution compared to 0.2m resolution
#         #chm = rasterio.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.extracted_filenames[-1]))).read()
#         #chm = chm.repeat(config.upscale_factor, axis = 1).repeat(config.upscale_factor, axis = 2)
#         # handle resolution diff: resize chm layer to the same shape as core image
#         #chm = resize(chm[:, :, :], (1, comb_img.shape[1], comb_img.shape[2]))
#         #comb_img = np.append(comb_img, chm, axis = 0)

    
#     comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
#     # print('statis', comb_img.min(), comb_img.mean(), comb_img.max())
#     # annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.extracted_filenames[0],config.annotation_fn)))
#     # np.asarray(annotation_im)
#     # annotation = np.array(annotation_im)
#     annotation = rasterio.open(os.path.join(base, fn.replace(extracted_filenames[0],'annotation'))).read()
#     annotation = np.squeeze(annotation)
#     # print('bef', np.unique(annotation))
#     if config.upsample:
#         annotation = resize(annotation[:, :], (int(annotation.shape[0]*2), int(annotation.shape[1]*2)), preserve_range = 1).astype(int)
#     # print('aft', np.unique(annotation))
#     weight = rasterio.open(os.path.join(base, fn.replace(extracted_filenames[0],'boundary'))).read()
#     weight = np.squeeze(weight)
#     if config.upsample:
#     # print('bef', np.unique(weight))
#         weight = resize(weight[:, :], (int(weight.shape[0]*2), int(weight.shape[1]*2)), preserve_range = 1).astype(int)
#     # print('aft', np.unique(weight))
#     f = FrameInfo(comb_img, annotation, weight)
#     frames.append(f)
    
    

# # training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
# training_frames = validation_frames = testing_frames  = list(range(len(frames)))
# annotation_channels = [3] + [4]
# test_generator = DataGenerator([0,1,2], (256, 256, 5), testing_frames, frames, annotation_channels, 3, augmenter= None).random_generator(8, normalize = 1)

# for i in range(1):
#     test_images, real_label = next(test_generator)
#     print(test_images.shape)
#     #5 images per row: pan, ndvi, label, weight, prediction
#     prediction = model.predict(test_images, steps=1)
#     prediction[prediction>0.5]=1
#     prediction[prediction<=0.5]=0
#     display_images(np.concatenate((test_images, real_label, prediction), axis = -1))
           
    


