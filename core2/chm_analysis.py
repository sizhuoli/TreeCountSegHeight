#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:21:52 2021

@author: sizhuo
"""
import glob
import os
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps

from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape

import numpy as np               # numerical array manipulation
# import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
# import time

from itertools import product
from tensorflow.keras.models import load_model
import cv2
import tensorflow.keras.backend as K
import copy 
from skimage.transform import resize
from rasterio.io import MemoryFile
# import sys

# from core.UNet_multires import UNet
from core.UNet_attention_CHM import UNet

from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
# from core.eva_losses import eva_acc, eva_dice, eva_sensitivity, eva_specificity, eva_miou
from core.optimizers import adaDelta, adagrad, adam, nadam
# from core.frame_info_multires import FrameInfo, image_normalize
# # from core.dataset_generator_multires import DataGenerator
# # from core.split_frames import split_dataset
# from core.visualize import display_images


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

from rasterio.enums import Resampling

import tensorflow as tf
print(tf.__version__)
from sklearn.metrics import mean_absolute_error, median_absolute_error

from sklearn.metrics import confusion_matrix
import seaborn as sn
from matplotlib import colors
import csv
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from scipy.optimize import curve_fit
import random




class chm_analyze:
    def __init__(self, config, logging):
        self.config = config
        OPTIMIZER = adam #
        if not self.config.multitask:
            self.model = load_model(self.config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            self.model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
        
        elif config.multitask:
            self.model = load_model(self.config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            self.model.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_chm': tf.keras.losses.Huber()},
                          metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                               'output_chm':[tf.keras.metrics.RootMeanSquaredError()]})
    

        self.all_files = load_files(self.config)
        logging.info('check info')
        logging.debug('check debug')
        logging.warning('check warning')
    def pred_testing(self):
        self.pr, self.ggt = predict_testing(self.config, self.all_files, self.model)
        return
    
    
    def pred_sampling_eva_DK(self, num = 10, sample_fixed = True):
        # randomly sample tifs from all files, pred and report errors
        # exclude training frames
        self.train_frames = load_train_files(self.config)
        # self.use_files = list(set(self.all_files) - set(self.train_frames))
        
        self.use_files = [i for i in self.all_files if i[1] not in self.train_frames]
        print('before exclude', len(self.all_files))
        print('exclude train', len(self.use_files))
        # sample randomly from the files
        if sample_fixed:
            # fix seed
            random.seed(1)
        # self.use_files_sampled = random.sample(self.use_files, num)
        # print('sampled', len(self.use_files_sampled))
        # maes = predict_large(self.config, self.use_files_sampled, self.model)
        # sampled based on forest type
        self.typefile = gps.read_file(self.config.typefile)
        # print(self.typefile.head())
        self.ft_errors = []
        self.avg_hs = []
        for ty in range(3):
            # allheight = []
            print('forest type', ty)
            # for each forest type, sample and compute errors
            self.use_files_ft = [i for i in self.use_files if np.unique(self.typefile[self.typefile['location'] ==i[1]]['_majority'].values) == ty]
            if len(self.use_files_ft) < num:
                print('Samples not enough, no. tifs:', len(self.use_files_ft))
                self.use_files_ft_sampled = self.use_files_ft
                # check forest type
                print('checking forest type of sampled tifs')
                for iid in range(len(self.use_files_ft_sampled)):
                    print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)
                
                maes_ft, avg_h, all_gt_hei = predict_large(self.config, self.use_files_ft_sampled, self.model)
                plt.figure(figsize = (6,6))
                plt.title('height hist - forest type '+str(ty), fontsize = 14)
                plt.hist(all_gt_hei, bins=50, density = 1)
                self.ft_errors.append(maes_ft)
                self.avg_hs.append(avg_h)
            else: 
                maes_ft = []
                while len(maes_ft) != num:
                    self.use_files_ft_sampled = random.sample(self.use_files_ft, num)
                    # check forest type
                    print('checking forest type of sampled tifs')
                    for iid in range(len(self.use_files_ft_sampled)):
                        print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)
                    
                    maes_ft, avg_h, all_gt_hei = predict_large(self.config, self.use_files_ft_sampled, self.model)
                    
                plt.figure(figsize = (6,6))
                plt.title('height hist - forest type '+str(ty), fontsize = 14)
                plt.hist(all_gt_hei, bins=50, density = 1)
                
                self.ft_errors.append(maes_ft)
                self.avg_hs.append(avg_h)
        plot_errors_ft(self.ft_errors, self.avg_hs)
        
        return self.ft_errors
        
    
    
    def plot_testing(self):
        CHMerror(self.pr, self.ggt)
        return 
    
    
    def pred_largescale(self):
        predict_large(self.config, self.all_files, self.model)
        return 
    
    def preprocess(self):
        # # compute gb stats
        self.q1s, self.q3s, self.min, self.max = compute_stat_gb(self.config, self.all_files)
        # selected gb stats
        # self.q1s, self.q3s = compute_stat(self.config, self.all_files)
        return self.q1s, self.q3s
    
    def plot_finland(self):
        
        self.pr, self.ggt, self.maes, self.avg_h = predict_finland(self.config, self.all_files, self.model)
        
        return 


        

def load_files(config):
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    # print(all_files)
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
            gb_im = gb_im[:,:,config.channels] # swap channels, inf last
            # print(gb_im.shape)
            merg1.append(gb_im[:, :, 0]) # green
            merg2.append(gb_im[:,:,1]) # blue
            merg3.append(gb_im[:,:,2]) # inf
    # print(np.array(merg1).shape)
    
    q1s, q3s = [], []
    li = [merg1, merg2]
    mins = []
    maxs = []
    # print(np.array(merg1).shape)
    for i in range(len(li)):
        
        q1 = np.quantile(np.array(li[i]), 0.25, axis = (0,1,2))
        q3 = np.quantile(np.array(li[i]), 0.75, axis = (0,1,2))
        q1s.append(q1)
        q3s.append(q3)
        mins.append(np.array(li[i]).min())
        maxs.append(np.array(li[i]).max())
    # for infrared, exclude values below 50
    merg3 = np.array(merg3).flatten()

    masked_b3 = merg3[merg3>50]
    # print(masked_b3.shape)
    q1 = np.quantile(masked_b3, 0.25)
    q3 = np.quantile(masked_b3, 0.75)
    q1s.append(q1)
    q3s.append(q3)
    mins.append(merg3.min())
    maxs.append(merg3.max())
    # print(q1s, q3s)
    
    return np.array(q1s), np.array(q3s), np.array(mins), np.array(maxs)

def compute_stat_GB(config):
    # real gb (all images in FI)
    
    base_dir = config.input_image_dir

    raster_fps = glob.glob(f"{base_dir}/**/*.jp2")
    # raster_fps = glob.glob(f"{base_dir}/*.jp2")

    print(len(raster_fps))
    
    selec = random.sample(raster_fps, k = 30)
    
    merg1 = []
    merg2 = []
    merg3 = []
    q1s = []
    q3s = []
    for curf in tqdm(selec):
        with rasterio.open(curf) as src:
            
            gb_im = np.transpose(src.read(), axes=(1,2,0)) # channel last
            gb_im = gb_im[:,:,config.channels] # swap channels, inf last
            # print(gb_im.shape)
            merg1.append(gb_im[:, :, 0])
            merg2.append(gb_im[:,:,1])
            merg3.append(gb_im[:,:,2])
    # print(np.array(merg1).shape)
    # q1s = np.quantile(np.array(merg1), 0.25, axis = (0,1,2))
    # q3s = np.quantile(np.array(merg1), 0.75, axis = (0,1,2))
    for j in [merg1, merg2, merg3]: # 3 bands
        mm = np.array(j).flatten()
        mm.sort()
        mq1 = mm[int(len(mm)*0.25)]
        mq3 = mm[int(len(mm)*0.75)]
        q1s.append(mq1)
        q3s.append(mq3)
    return q1s, q3s

def load_train_files(config):
    all_files = []
    for root, dirs, files in os.walk(config.training_frames_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append(file)
    print('Number of raw tif to exclude (training):', len(all_files))
    # print(all_files)
    return all_files


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

def predict_using_model(model, batch, batch_pos, mask, operator, upsample = 1):
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

def predict_using_model_fi(model, batch, batch_pos, mask, operator, upsample = 1, downsave = 1):
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
        # print('pred from model', prediction.shape)
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        
        # if upsample and downsave:   # no preserve range
        #     # print('**********************UPsample for prediction***************************')
        #     # print('**********************Downsample for saving***********************')
        #     # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
        #     p = resize(p[:, :], (int(p.shape[0]/2), int(p.shape[1]/2)), preserve_range=True)
            
        #     # rescale values
        #     p = p * (p.shape[0] / float(p.shape[0]/2)) * (p.shape[1] / float(p.shape[1]/2))
            
        # print('before add tore ', p.shape)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask

#multitask
def predict_using_model_segchm(model, batch, batch_pos_seg, batch_pos_chm, maskseg, maskchm, operator, upsample = 1):
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
        pred_seg, pred_chm = model.predict(tm1) # tm is a list []
        # print(predictions[2].shape)
        # pred_seg = predictions['output_seg']
        # pred_chm = predictions['output_chm']
        # print(prediction.shape)
    
    for i in range(len(batch_pos_seg)):
        (col, row, wi, he) = batch_pos_seg[i]
        pseg = np.squeeze(pred_seg[i], axis = -1)
        maskseg = addTOResult(maskseg, pseg, row, col, he, wi, operator)
        
        (col, row, wi, he) = batch_pos_chm[i]
        pchm = np.squeeze(pred_chm[i], axis = -1)
        maskchm = addTOResult(maskchm, pchm, row, col, he, wi, operator)
    return maskseg, maskchm


# input images are raw tif frames 0 only 4/3 bands
def detect_tree_rawtif(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
    """img can be one single raster or multi rasters
    
    img = [core_img, aux1, aux2, ...]
    
    or
    
    img = img #single raster
    """
        
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
    
    # compute gb FI stats
    # if config.robustscaleFI_gb:
        
    #     gb_im = np.transpose(img.read(), axes=(1,2,0)) # channel last
    #     gb_im = gb_im[:,:,channels] # swap channels, inf last
    #     q1s = np.quantile(gb_im, 0.25, axis = (0,1))
    #     q3s = np.quantile(gb_im, 0.75, axis = (0,1))
    #     assert len(q1s) == 3 and len(q3s) == 3
    #     print('q1', q1s)
    #     print('q3', q3s)
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        
         
        #single rater or multi raster without aux # must be multi raster
        patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels] # swap channels, inf last
        
        if len(config.channels) > 3: # all bands
            if config.gbnorm:
                print('all bands - gb norm')
                temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
                temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])

            elif config.robustscale:
                # todo: robust for all bands
                print('incomplete')
                temp_im = (temp_im - np.array([[73.0, 72.0, 145.0]]))/ np.array([[113.0-73.0, 96.0-72.0, 182.0-145.0]])
                if normalize:
                    temp_im = image_normalize(temp_im, axis=(0,1))
        
        elif len(config.channels) == 3: # 3 bands
            
            if config.robustscale: 
                print('3 bands - robust scale - DK - gb')
                q1s = np.array([[73.0, 72.0, 145.0]])
                q3s = np.array([[113.0, 96.0, 182.0]])
                temp_im = (temp_im - q1s)/ (q3s - q1s) 
        
            if config.robustscaleFI_gb: # 3 bands FI self stats gbly
                print('3 bands - robust scale - FI - gb')
                q1s = np.array([46, 65, 21])
                q3s = np.array([ 99, 106, 130])
                temp_im = (temp_im - q1s)/ (q3s - q1s)
                
            if config.robustscaleFI_local: # 3 bands FI self stats locally
                print('3 bands - robust scale - FI - local')
                q1s = np.quantile(temp_im, 0.25, axis = (0,1))
                q3s = np.quantile(temp_im, 0.75, axis = (0,1))
                assert len(q1s) == 3 and len(q3s) == 3
                temp_im = (temp_im - q1s)/ (q3s - q1s)
            
            
            
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

def detect_tree_rawtif_fi(config, model, img, channels, q1s = None, q3s = None, mins = None, maxs = None, width=256, height=256, stride = 128, normalize=0, maxnorm = 0, upsample = 1, downsave = 0):
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
    
    
    
    if downsave or not upsample:
        mask = np.zeros((int(nrows), int(nols)), dtype=meta['dtype'])

    elif not downsave:
        mask = np.zeros((int(nrows*2), int(nols*2)), dtype=meta['dtype'])
        meta.update(
                    {'width': int(nols*2),
                     'height': int(nrows*2),
                    }
                    )
    
    
    # compute gb FI stats # too slow
    # if config.robustscaleFI_gb:
        
    #     gb_im = np.transpose(img.read(), axes=(1,2,0)) # channel last
    #     gb_im = gb_im[:,:,channels] # swap channels, inf last
    #     q1s = np.quantile(gb_im, 0.25, axis = (0,1))
    #     q3s = np.quantile(gb_im, 0.75, axis = (0,1))
    #     assert len(q1s) == 3 and len(q3s) == 3
    #     print('q1', q1s)
    #     print('q3', q3s)
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        if upsample:
            # patch1 = np.zeros((height, width, len(img)))
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
            patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
            read_im = img.read(window = window)
        # #single rater or multi raster without aux # must be multi raster
        # patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        # read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels] # swap channels, inf last
        # print(temp_im.mean(axis = (0, 1)))
        
        # print('max rescaling', temp_im.max)
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        if len(config.channels) > 3:
            if config.gbnorm: # all bands
                logging.info('all bands - standarization - gb')
                temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
                temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])

            elif config.robustscale: # all bands robust scale
                logging.info('incomplete')
                # todo: robust scale for all bands
                # temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                if normalize:
                    temp_im = image_normalize(temp_im, axis=(0,1))
                    
        elif len(config.channels) == 3: # 3 bands
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
                temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                temp_im = (temp_im - np.array([[0.350, 0.321, 0.560]]))/ np.array([[ 0.895, 0.703, 1.107]])

            
            if config.gbnorm_FI:
                # normalize globally gb inf from FI training data
                logging.info('3 bands - gb norm - FI training data')
                temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.300, 0.338, 0.343]]))/ np.array([[0.168, 0.153, 0.146]]) # 2 big tifs
                # temp_im = (temp_im - np.array([[0.256, 0.306, 0.311]]))/ np.array([[0.125, 0.118, 0.129]])
                # from 3 tif FI
                temp_im = (temp_im - np.array([[0.253, 0.300, 0.321]]))/ np.array([[0.122, 0.118, 0.127]])
            
            
            if config.localtifnorm:
                logging.info('3 bands - local tif norm - DK')
                temp_im = temp_im / 255
                temp_im = image_normalize(temp_im, axis=(0,1))
        # print('read', temp_im.shape)
        if upsample:
            patch[:int(window.height*2), :int(window.width*2)] = temp_im
        else:
            patch[:int(window.height), :int(window.width)] = temp_im
        # patch[:window.height, :window.width] = temp_im
        batch.append(patch)
        
        if downsave or not upsample:
            batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        elif not downsave:
            # print('upsave')
            batch_pos.append((int(window.col_off*2), int(window.row_off*2), int(window.width*2), int(window.height*2)))
        
        # batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model_fi(model, batch, batch_pos, mask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model_fi(model, batch, batch_pos, mask, config.operator)
        batch = []
        batch_pos = []
        
    if maxnorm:
        mask = mask * 97.19
    return(mask, meta)

# input images are raw tif frames 4 bands with NDVI
def detect_tree_rawtif_ndvi(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
    """img can be one single raster or multi rasters
    
    img = [core_img, aux1, aux2, ...]
    
    or
    
    img = img #single raster
    """
        
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
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        
         
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
        # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
        temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])
        
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

# input images are separate band images
def detect_tree_separateBands(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
    """img can be one single raster or multi rasters
    
    img = [core_img, aux1, aux2, ...]
    
    or
    
    img = img #single raster
    """
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()    
    #single rater or multi raster without aux
    nols, nrows = img0.meta['width'], img0.meta['height']
    meta = img0.meta.copy()
        
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
        # print(col_off, row_off)
        # transform = windows.transform(window, core_img.transform)
        
         
        #single rater or multi raster without aux # must be multi raster
        patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        temp_im = img[0].read(window=window)
        for ch in range(1, len(config.channel_names)): # except for the last channel 
                    
            imgi = rasterio.open(img[ch]) 
            
            sm1 = imgi.read(
                            out_shape=(
                            1,
                            int(window.height),
                            int(window.width)
                        ),
                        resampling=Resampling.bilinear, window = window)
            # print('aux shape', aux_sm.shape)
            
            temp_im = np.row_stack((temp_im, sm1))
            
        temp_im = np.transpose(temp_im, axes=(1,2,0))
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        # print('global normalize')
        if len(config.channel_names) > 3 and config.gbnorm: # all bands
            temp_im = temp_im / 255
            # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
            temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])

        elif len(config.channels) > 3 and config.robustscale: # all bands robust scale
            # todo: robust scale for all bands
            # temp_im = temp_im / 255
            # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
            if normalize:
                temp_im = image_normalize(temp_im, axis=(0,1))
        elif len(config.channels) == 3 and config.robustscale: # 3 bands
            temp_im = (temp_im - np.array([[73.0, 72.0, 145.0]]))/ np.array([[113.0-73.0, 96.0-72.0, 182.0-145.0]])
        
        
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


# multitask
def detect_tree_segchm(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
    """img can be one single raster or multi rasters
    
    img = [core_img, aux1, aux2, ...]
    
    or
    
    img = img #single raster
    """
        
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
    maskchm = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])
    maskseg = np.zeros((nrows, nols), dtype=meta['dtype'])
    
    batch = []
    batch_pos_seg = []
    batch_pos_chm = []
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        
         
        #single rater or multi raster without aux # must be multi raster
        patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels]
        
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        
        patch[:window.height, :window.width] = temp_im
        batch.append(patch)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        batch_pos_chm.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        batch_pos_seg.append((window.col_off, window.row_off, window.width, window.height))

        if (len(batch) == config.BATCH_SIZE):
            maskseg, maskchm = predict_using_model_segchm(model, batch, batch_pos_seg, batch_pos_chm, maskseg, maskchm, config.operator)
            batch = []
            batch_pos_seg = []
            batch_pos_chm = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        maskseg, maskchm = predict_using_model_segchm(model, batch, batch_pos_seg, batch_pos_chm, maskseg, maskchm, config.operator)
        batch = []
        batch_pos_seg = []
        batch_pos_chm = []
    
    if maxnorm:
        maskchm = maskchm * 97.19
    return(maskseg, maskchm, meta)
        
        
def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, write_as_type = 'float32', scale = 1):
    # Convert to correct required before writing
    
    # scale by 100 and then store
    if scale:
        detected_mask = detected_mask * 100
    
    # print(detected_mask.max())
      
    # print('mask', detected_mask.shape)
    # print('meta', detected_meta['height'])
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
        

def CHMdiff(detected_mask, gt_mask, filename, nbins = 100):
    gt_mask_im = np.squeeze(gt_mask.read())
    print('gt mask im', gt_mask_im.shape)
    # no data mask out
    detected_mask[gt_mask_im == -9999] = 0
    gt_mask_im[gt_mask_im == -9999] = 0
    
    diff = gt_mask_im - detected_mask
    
    h = tf.keras.losses.Huber()
    huberLoss = h(gt_mask_im, detected_mask).numpy()
    
    mse = tf.keras.losses.MeanSquaredError()
    MseLoss = mse(gt_mask_im, detected_mask).numpy()

    # MseLossf = mse(gt_mask_im.flatten(), detected_mask.flatten()).numpy()    
    print('*******************Huber loss: {}******************'.format(huberLoss))
    print('*******************MSE loss: {}******************'.format(MseLoss))
    # print('*******************MSE lossf: {}******************'.format(MseLossf))
    
    nbins = nbins
    plt.figure()
    plt.hist(diff.flatten(), bins=nbins, density = True)
    plt.title('CHM diff for '+filename)
    
    return diff, gt_mask_im, detected_mask
    



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

def pooling(mat,ksize,method='max',pad=False):
    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def CHMdiff_fi(detected_mask, gt_mask, filename, nbins = 100, scale = 1):
    
    if scale:
        # resample data to target shape
        gt_mask_im = gt_mask.read(
            out_shape=(
                gt_mask.count,
                int(gt_mask.height * 2),
                int(gt_mask.width * 2)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = gt_mask.transform * gt_mask.transform.scale(
            (gt_mask.width / gt_mask_im.shape[-1]),
            (gt_mask.height / gt_mask_im.shape[-2])
        )
    else:
        gt_mask_im = np.squeeze(gt_mask.read())
    
    gt_mask_im = np.squeeze(gt_mask_im)
    detected_mask = np.squeeze(detected_mask)
    print('gt mask im', gt_mask_im.shape)
    # no data mask out
    detected_mask[gt_mask_im == -9999] = 0
    gt_mask_im[gt_mask_im == -9999] = 0
    
    gt_pool = pooling(gt_mask_im, (10,10), method='mean')
    pd_pool = pooling(detected_mask, (10,10), method='mean')
    
    gt_poolo = poolingOverlap(gt_mask_im, (10,10), (5, 5), method='mean')
    pd_poolo = poolingOverlap(detected_mask, (10,10), (5, 5), method='mean')
    
    
    diff = gt_mask_im - detected_mask
    diffP = gt_pool - pd_pool
    diffPo = gt_poolo - pd_poolo
    
    
    h = tf.keras.losses.Huber()
    huberLoss = h(gt_mask_im, detected_mask).numpy()
    
    mse = tf.keras.losses.MeanSquaredError()
    MseLoss = mse(gt_mask_im, detected_mask).numpy()

    # MseLossf = mse(gt_mask_im.flatten(), detected_mask.flatten()).numpy()    
    print('*******************Huber loss: {}******************'.format(huberLoss))
    print('*******************MSE loss: {}******************'.format(MseLoss))
    # print('*******************MSE lossf: {}******************'.format(MseLossf))
    
    nbins = nbins
    plt.figure()
    plt.hist(diff.flatten(), bins=nbins, density = True)
    plt.title('CHM diff for '+filename)


    # pooling to get errors
    
    huberLossP = h(gt_pool, pd_pool).numpy()
    MseLossP = mse(gt_pool, pd_pool).numpy()
    
    nbins = nbins
    plt.figure()
    plt.hist(diffP.flatten(), bins=nbins, density = True)
    plt.title('POOLING CHM diff for '+filename)
    
    print('*******************POOLING Huber loss: {}******************'.format(huberLossP))
    print('*******************POOLING MSE loss: {}******************'.format(MseLossP))
    
    # overlapping pooling to get errors
    
    huberLossPo = h(gt_poolo, pd_poolo).numpy()
    MseLossPo = mse(gt_poolo, pd_poolo).numpy()
    
    nbins = nbins
    plt.figure()
    plt.hist(diffPo.flatten(), bins=nbins, density = True)
    plt.title('Overlapping POOLING CHM diff for '+filename)
    
    print('*******************Overlapping POOLING Huber loss: {}******************'.format(huberLossPo))
    print('*******************Overlapping POOLING MSE loss: {}******************'.format(MseLossPo))
    
    return diff, gt_mask_im, detected_mask
    

def CHMerror0(detected_mask, gt_mask,  nbins = 100):
    gt_mask_im = np.squeeze(gt_mask.read()).flatten()
    pr = detected_mask.flatten()
    # plt.figure()
    # plt.scatter(c , s =1, c = 'grey')
    plt.figure(figsize = (10,8))
    plt.hist2d(pr, gt_mask_im, cmap='Blues')
    plt.colorbar()
    
    gtm = int(np.ceil(gt_mask_im.max()))
    
    def categorical(digit, gtm, bins = 0.5):
        """
        This will give us access to our binned targets (y):
        terrible : 0.0 < y <= 3.0
        okay     : 3.0 < y <= 5.0
        great    : 5.0 < y <= 7.0
        amazing  : 7.0 < y < 10.1
        """
        return np.digitize(digit, list(np.arange(0.0, gtm, bins)))

    # small intervals
    bins = 0.5
    ppdcat = categorical(pr ,gtm, bins)
    gttcat = categorical(gt_mask_im, gtm, bins)
    cf = confusion_matrix(gttcat, ppdcat)
    ticks = np.around(list(np.linspace(0, gtm, gtm)), decimals=2)
    fig = plt.figure(figsize = (10,10))
    ax = sn.heatmap(cf,   annot = False, linewidths=.05,  cmap="YlGnBu", square = 1)
    # plt.locator_params(nbins=42)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(gtm))
    ax.set_xticklabels(ticks)
    ax.yaxis.set_major_locator(plt.MaxNLocator(gtm))
    ax.set_yticklabels(ticks)
    
    plt.xlabel("Prediction (m)") 
    plt.ylabel('Ground truth (m)')
    plt.title('CHM max height perpixel \n Training loss = Huber')


    
    
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

    fig, ax = plt.subplots(figsize=(9,7))
    ax.boxplot(maes, showfliers=False, showmeans = True)
    ax.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    labels = ['0-10m', '10-20m', '20-30m', '>30m']
    ax.set_xticklabels(labels)
    ax.set_xlabel('Height', fontsize = 14)
    ax.set_ylabel('MAE', fontsize = 14)
    
    ax.grid(True, axis = 'y', alpha = 0.3)

    return     
    
    

def mergePredTiff(meta, outputFiles, output_dir, output_prefix, output_image_type, output_dtype):
    """Merge predictions in one single raster tif
    
    meta: ini meta profile from an individual raster
    outputFiles: list of individual prediction tif files
    
    """
    merged, mergedTrans = merge.merge(outputFiles)
    merged = np.squeeze(merged)
    mergedMeta = meta.copy()
    mergedMeta.update({'compress':'lzw', 'width':merged.shape[1], 'height': merged.shape[0], 'transform': mergedTrans})
    mergedFn = os.path.join(output_dir, output_prefix+'merged.tif')
    writeMaskToDisk(merged, mergedMeta, mergedFn, image_type = output_image_type, write_as_type = output_dtype)            
    print('All predictions merged to a single raster!')
    return
  
def predict0(config, all_files, model):
    # for need to search for CHM files version
    outputFiles = []
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        #print(outputFile)
        outputFiles.append(outputFile)
        outputChmDiff = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.chmdiff_prefix))
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            with rasterio.open(fullPath) as img:
                # locate gt chm 
                coor = fullPath[-12:-9]+ fullPath[-8:-5]
                chmdir = os.path.join(config.gt_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                chmPath = fullPath.replace(config.input_image_pref, config.chm_pref).replace(config.input_image_dir, chmdir)
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                
                    
                    detectedMask, detectedMeta = detect_tree_separateBands(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    
                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
                    # compute diff
                    chmdiff = CHMdiff(detectedMask, chm, filename)
                    writeMaskToDisk(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chmdiff_dtype, scale = 0)
    
        else:
            print('File already analysed!', fullPath)
    return  

def predict_finland(config, all_files, model):
    # for need to search for CHM files 
    maes = {}
    avg_h = {}
    outputFiles = []
    pr = np.array([])
    ggt = np.array([])
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        # print(outputFile)
        outputFiles.append(outputFile)
        outputChmDiff = os.path.join(config.output_dir, filename[:-4] + config.chmdiff_suffix + config.output_image_type)
        # print(outputChmDiff)
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            with rasterio.open(fullPath) as img:
                # locate gt chm 
                
                chmPath = config.gt_chm_dir  + config.chm_pref + filename[:-4] + config.chm_sufix
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                    if config.upsample and config.downsave:
                        # detectedMask, detectedMeta = detect_tree_rawtif_fi(config, model, img, config.channels, q1, q3, mins, maxs, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        # print('**********************UPsample for prediction***************************')
                        detectedMask, detectedMeta = detect_tree_rawtif_fi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        
                        # print('**********************Downsample for saving***********************')
                        # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
                        # detectedMask = resize(detectedMask[:, :], (int(detectedMask.shape[0]/2), int(detectedMask.shape[1]/2)), preserve_range=True)
                        # print(detectedMask.mean())
                        # # rescale values
                        # detectedMask = detectedMask * (detectedMask.shape[0] / float(detectedMask.shape[0]/2)) * (detectedMask.shape[1] / float(detectedMask.shape[1]/2))
                          
                    
                    else:
                        detectedMask, detectedMeta = detect_tree_rawtif(config, model, img, config.channels, q1, q3, mins, maxs, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    
                    
                        
                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
                    # compute diff
                    chmdiff, gt_chm_mask, detected_chm_mask = CHMdiff_fi(detectedMask, chm, filename)
                    writeMaskToDisk(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chmdiff_dtype, scale = 0)
                    pr = np.append(pr,  detected_chm_mask.flatten())
                    ggt = np.append(ggt, gt_chm_mask.flatten())
                    maei = mean_absolute_error(ggt, pr)
                    maes[filename] = maei
                    print('mae', maei)
                    avg_h[filename] = np.mean(ggt)
                    print('--------- mean height', np.mean(ggt))
        else:
            print('File already analysed!', fullPath)
    MAE = mean_absolute_error(ggt, pr)
    print('MAE for all', MAE)
    return pr, ggt, maes, avg_h
        



def predict_testing(config, all_files, model):
    # testing areas, separte bands
    outputFiles = []
    pr = np.array([])
    ggt = np.array([])
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        #print(outputFile)
        outputFiles.append(outputFile)
        outputChmDiff = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.chmdiff_prefix))
        
            
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
            
            
            chmPath = fullPath.replace(config.input_image_pref, config.chm_pref).replace('png', 'tif')
            # print(chmPath)
            with rasterio.open(chmPath) as chm:
            
            
                #print(fullPath)
                print('---- separate band images -----')
                detectedMask, detectedMeta = detect_tree_separateBands(config, model, [im0, *chs], config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                
                #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                if config.saveresult:
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
                # compute diff
                # CHMerror(detectedMask, chm, filename)
                # chmdiff, _, _ = CHMdiff(detectedMask, chm, filename)
                # compute diff with tree level height
                chmdiff, gt_chm_mask, detected_chm_mask = CHMdiff_fi(detectedMask, chm, filename, scale = 0)
                if config.saveresult:
                    writeMaskToDisk(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chmdiff_dtype, scale = 0)
                pr = np.append(pr,  detectedMask.flatten())
                ggt = np.append(ggt, np.squeeze(chm.read()).flatten())
    
    MAE = mean_absolute_error(ggt, pr)
    print('MAE for all', MAE)
    return pr, ggt



def CHMerror(pr, gt_mask_im,  nbins = 100):
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
    ax_hist_x.hist(pr, bins=80, color='navy', alpha=0.3, density = 1, orientation = 'horizontal')
    ax_scatter.plot(xx, xx, c ='grey', alpha = 0.5)
    slope, intercept, r_value, p_value, std_err = linregress(gt_mask_im, pr)
    ax_scatter.plot(xx, func(np.array(xx), *popt), 'b--', alpha = 0.5, label='f(x) = %5.3f x + %5.3f; r2 = %5.3f ' % (popt[0], popt[1], r_value**2))
    ax_scatter.set_aspect('equal')
    ax_scatter.set_xlim(0, 45)
    ax_scatter.set_ylim(0, 45)
    ax_hist_y.set_xlim(0, 45)
    ax_hist_x.set_ylim(0, 45)
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

def plot_errors_ft(ft_err, md):
    errs = []
    mds = []
    for i in range(len(ft_err)):
        errs.append(list(ft_err[i].values()))
        mds.append(list(md[i].values()))
        
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    line_props = dict(linestyle='--', color="grey", alpha=0.9)
    capprops = dict(linewidth = 1, color="grey", alpha = 0.9)
    bp = ax.boxplot(errs, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    pp = ax.boxplot(mds, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    labels = ['Non-forest', 'Broadleaved', 'Coniferous']
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize = 14)
    ax.set_xlabel('Forest type', fontsize = 14)
    ax.set_ylabel('MAE (m)', fontsize = 14)
    
    med = []
    mu = []
    for t in errs:
        med.append(np.median(t))
        mu.append(np.mean(t))
        
    med2 = []
    mu2 = []
    for t in mds:
        med2.append(np.median(t))
        mu2.append(np.mean(t))
    
    # fig, ax = plt.subplots()
    # bp = ax.boxplot(data3, showmeans=True)
    
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med[i], mu[i])
        ax.annotate(text, xy=(x, y))
    
    for i, line in enumerate(pp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med2[i], mu2[i])
        ax.annotate(text, xy=(x, y))
    
    ax.grid(True, axis = 'y', alpha = 0.3)
    
    plt.show()
    return 

def plot_errors_fi(ft_err, md):
    errs = []
    mds = []
    
    errs = list(ft_err.values())
    mds=list(md.values())
        
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    line_props = dict(linestyle='--', color="grey", alpha=0.9)
    capprops = dict(linewidth = 1, color="grey", alpha = 0.9)
    bp = ax.boxplot(errs, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    pp = ax.boxplot(mds, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    labels = ['all']
    ax.set_xticks([1])
    ax.set_xticklabels(labels, fontsize = 14)
    ax.set_xlabel('.', fontsize = 14)
    ax.set_ylabel('MAE (m)', fontsize = 14)
    
    med = []
    mu = []
    for t in errs:
        med.append(np.median(t))
        mu.append(np.mean(t))
        
    med2 = []
    mu2 = []
    for t in mds:
        med2.append(np.median(t))
        mu2.append(np.mean(t))
    
    # fig, ax = plt.subplots()
    # bp = ax.boxplot(data3, showmeans=True)
    
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med[i], mu[i])
        ax.annotate(text, xy=(x, y))
    
    for i, line in enumerate(pp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med2[i], mu2[i])
        ax.annotate(text, xy=(x, y))
    
    ax.grid(True, axis = 'y', alpha = 0.3)
    
    plt.show()
    return 


def predict_large(config, all_files, model):
    # eva: whether to compare with gt chm
    if config.eva:
        maes = {}
        nochm = []
        inte_maes_all = []
        heights_gt = np.array([])
    avg_h = {}
    outputFiles = []
    for fullPath, filename in all_files:
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        outputFiles.append(outputFile)
        # outputChmDiff = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.chmdiff_prefix))
        
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            with rasterio.open(fullPath) as img:
                if config.eva:
                    # locate gt chm 
                    coor = fullPath[-12:-9]+ fullPath[-8:-5]
                    chmdir = os.path.join(config.gt_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                    chmPath = fullPath.replace(config.input_image_dir, chmdir).replace('2018_', 'CHM_')
                    if 'a1/' in chmPath:
                        chmPath = chmPath.replace('a1/', '')
                    elif 'a2/' in chmPath:
                        chmPath = chmPath.replace('a2/', '')
                    # print(chmPath)
                    try:   
                        # print(chmPath)
                        with rasterio.open(chmPath) as chm:
                            print('image', filename)
                            print('chm', os.path.basename(chmPath))
                            if chm.read().max() > 50:
                                print('GT chm containing large values > 50m, please redo sampling')
                                # skip error files
                                continue
                            detectedMask, detectedMeta = detect_tree_rawtif_ndvi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                            
                            #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                            if config.saveresult:
                                writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
                            pr = detectedMask.flatten()
                            ggt = np.squeeze(chm.read()).flatten()
                            maei = mean_absolute_error(ggt, pr)
                            maes[filename] = maei
                            print('mae', maei)
                            avg_h[filename] = np.mean(ggt)
                            print('--------- mean height', np.mean(ggt))
                            heights_gt = np.append(heights_gt, ggt)
                        
                    except:
                        nochm.append(filename)
                        detectedMask, detectedMeta = detect_tree_rawtif_ndvi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                else:
                    # only predict, do not evaluate
                    detectedMask, detectedMeta = detect_tree_rawtif_ndvi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                if config.saveresult:
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
                    
                    
        else:
            print('File already analysed!', fullPath)
    if config.eva:
        if config.saveresult:
            w = csv.writer(open(os.path.join(config.output_dir, "maes.csv"), "w"))
            for key, val in maes.items():
                w.writerow([key, val])
    
        return maes, avg_h, heights_gt.flatten()
    else:
        return


def predict_segchm(config, all_files, model):
    # multitask
    outputFiles = []
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFilechm = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        #print(outputFile)
        outputFileseg = os.path.join(config.output_dir, filename.replace(config.input_image_pref, 'det_2018_'))
        outputFiles.append(outputFilechm)
        outputChmDiff = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.chmdiff_prefix))
        if not os.path.isfile(outputFilechm) or config.overwrite_analysed_files: 
            
            with rasterio.open(fullPath) as img:
                # locate gt chm 
                coor = fullPath[-12:-9]+ fullPath[-8:-5]
                chmdir = os.path.join(config.gt_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                chmPath = fullPath.replace(config.input_image_pref, config.chm_pref).replace(config.input_image_dir, chmdir)
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                
                    detectedMaskSeg, detectedMaskChm, detectedMeta = detect_tree_segchm(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    
                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                    writeMaskToDisk(detectedMaskChm, detectedMeta, outputFilechm, image_type = config.output_image_type, write_as_type = config.output_dtype)
                    # compute diff
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFileseg, image_type = config.output_image_type, write_as_type = config.output_dtype)
    
                    chmdiff = CHMdiff(detectedMaskChm, chm, filename)
                    writeMaskToDisk(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chmdiff_dtype, scale = 0)
    
        else:
            print('File already analysed!', fullPath)
    
    return
    
    

def expand_model(loaded_model, newInputShape = (None, 512, 512, 3)):
    """"
    shrink all layer shape by 2
    
    input shape: 128, output shape: 128, bottom shape: 8
    
    """
    
    loaded_model.layers[0]._batch_input_shape = newInputShape
    # loaded_model.summary()
    # model_config = loaded_model.get_config()
    
    new_model = tf.keras.models.model_from_json(loaded_model.to_json()) 
    print(new_model.summary())
    
    return new_model



# def image_normalize(im, axis = (0,1), c = 1e-8):
#     '''
#     Normalize to zero mean and unit standard deviation along the given axis'''
#     return (im - im.mean(axis, keepdims=True)) / (im.std(axis, keepdims=True) + c)

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)



def predict_gb_norm(config, all_files, model):
    # with global norm
    outputFiles = []
    
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        #print(outputFile)
        outputFiles.append(outputFile)
        
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
            
            with rasterio.open(fullPath) as img:
                #print(fullPath)
                imm = img.read()
                # print(imm.shape)
                # print(imm.mean(axis = (1, 2)))
                imm = np.transpose(imm, axes=(1,2,0)) # channel last
                norm_im = image_normalize(imm, axis=(0,1))
                norm_im = np.transpose(norm_im, axes=(2, 0, 1)) # channel first
                # print(norm_im.shape)
                # print(norm_im.mean(axis = (1, 2)))
                meta = img.profile.copy()
                meta.update({'dtype':np.float32})
                with MemoryFile() as memfile:
                
                    with memfile.open(**meta) as dataset:
                        dataset.write(norm_im.astype("float32"))
                        del norm_im
                    with memfile.open() as dataset:
                        detectedMask, detectedMeta = detect_tree_rawtif(config, model, dataset, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                        writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype)
        else:
            print('File already analysed!', fullPath)
    
    mergePredTiff(detectedMeta, outputFiles, config.output_dir, config.output_prefix, config.output_image_type, config.output_dtype)
    return 
    














