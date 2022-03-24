#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:54:26 2021

@author: sizhuo
"""

# from 1ha girds with FTY, randomly sample patches for predicting segcount and chm

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
import scipy
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
# from core.eva_losses import eva_acc, eva_dice, eva_sensitivity, eva_specificity, eva_miou
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info_multires import FrameInfo, image_normalize
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
from shapely.geometry import shape
from rasterio.windows import Window
from rasterio.features import shapes
# from osgeo import ogr, gdal
import multiprocessing
import math

from rasterio.windows import Window

# class sampler:
#     # randomly sample tifs (patches) for both segcount and chm
    

class analyzer:
    def __init__(self, config):
        
        self.config = config
        OPTIMIZER = adam #
        if not self.config.chm_multitask:
            self.chm_model = load_model(self.config.chm_trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            self.chm_model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
        
        elif config.chm_multitask:
            self.chm_model = load_model(self.config.chm_trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            self.chm_model.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_chm': tf.keras.losses.Huber()},
                          metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                               'output_chm':[tf.keras.metrics.RootMeanSquaredError()]})
    
        
        
        self.segcount_models = []
        for mod in config.segcount_trained_model_paths:
            modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            modeli.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            self.segcount_models.append(modeli)


        # self.all_files = load_files(self.config)
        self.typefile = gps.read_file(self.config.typefile)
    
    def sampler(self, k = 10):
        
        # sample patches and prepare sparate bands like for testing patches
        sel_id = random.sample(range(len(self.typefile)), k)
        self.sel_patches = self.typefile.loc[sel_id, :]
        
        self.all_files = []
        
        for i in range(len(self.sel_patches)):
            # cur = sel_patches.loc[i, :]
            # print(str(sel_patches.iloc[i,4])[:4])
            fname_cor = '2018_1km_' + str(self.sel_patches.iloc[i,4])[:4] + '_' + str(self.sel_patches.iloc[i, 1])[:3] + '.tif'
            fpath = glob.glob(self.config.input_image_dir + "/**/" + fname_cor, recursive = True)
            # print(fpath)
            fchmname_cor = 'CHM_1km_' + str(self.sel_patches.iloc[i,4])[:4] + '_' + str(self.sel_patches.iloc[i, 1])[:3] + '.tif'
            chmpath = glob.glob(self.config.gt_chm_dir + "/**/" + fchmname_cor, recursive = True)
            # print(chmpath)
            fndviname_cor = 'compress_ndvi_2018_1km_' + str(self.sel_patches.iloc[i,4])[:4] + '_' + str(self.sel_patches.iloc[i, 1])[:3] + '.tif'
            ndvipath = glob.glob(self.config.ndvi_dir + "/**/" + fndviname_cor, recursive = True)
            # print(ndvipath)
            
            self.all_files.append((fpath, ndvipath, chmpath))
        # main tif
        return
    
    
    def preprocess(self):
        c = 0
        for f in self.all_files:
            # print(f[0])
            try:
                with rasterio.open(f[0][0]) as src:
                    filename = os.path.basename(f[0][0])
                    # The size in pixels of your desired window
                    xsize, ysize = 500, 500
                
                
                    # Create a Window and calculate the transform from the source dataset    
                    # print(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*5)
                    window = Window(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*5, 5000 - int(str(int(self.sel_patches.iloc[c,4]))[-3:])*5 - 500, xsize, ysize)
                    transform = src.window_transform(window)
                
                    # Create a new cropped raster to write to
                    profile = src.profile
                    profile.update({
                        'height': xsize,
                        'width': ysize,
                        'transform': transform})
                    suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'
                    with rasterio.open(self.config.output_extracted_dir+filename.replace('.tif', suf), 'w', **profile) as dst:
                        # Read the data from the window and write it to the output raster
                        dst.write(src.read(window=window))
                
                # ndvi
                with rasterio.open(f[1][0]) as src1:
                    filename = os.path.basename(f[1][0])
                    # The size in pixels of your desired window
                    xsize, ysize = 500, 500
                
                
                    # Create a Window and calculate the transform from the source dataset    
                    # print(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*5)
                    window = Window(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*5, 5000 - int(str(int(self.sel_patches.iloc[c,4]))[-3:])*5 - 500, xsize, ysize)
                    transform = src1.window_transform(window)
                
                    # Create a new cropped raster to write to
                    profile = src1.profile
                    profile.update({
                        'height': xsize,
                        'width': ysize,
                        'transform': transform})
                    suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'
                    with rasterio.open(self.config.output_extracted_dir+filename.replace('.tif', suf), 'w', **profile) as dst:
                        # Read the data from the window and write it to the output raster
                        dst.write(src1.read(window=window))
                
                # chm
                with rasterio.open(f[2][0]) as src2:
                    filename = os.path.basename(f[2][0])
                    # The size in pixels of your desired window
                    xsize, ysize = 250, 250
                
                    # Create a Window and calculate the transform from the source dataset    
                    # print(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*5)
                    window = Window(int(int(str(int(self.sel_patches.iloc[c,1]))[-3:])*2.5), 2500 - int(int(str(int(self.sel_patches.iloc[c,4]))[-3:])*2.5) - 250, xsize, ysize)
                    transform = src2.window_transform(window)
                
                    # Create a new cropped raster to write to
                    profile = src2.profile
                    profile.update({
                        'height': xsize,
                        'width': ysize,
                        'transform': transform})
                    suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'
                    with rasterio.open(self.config.output_extracted_dir+filename.replace('.tif', suf), 'w', **profile) as dst:
                        # Read the data from the window and write it to the output raster
                        dst.write(src2.read(window=window))
                c += 1
            
            except:
                c += 1
                continue
            
        return 
    
    
    def segcount_eva(self):
        self.pred_segs = segcount_pred(self)
        return
    
    def thinning(self):
        self.tree_density, self.crown_area_avgs = self_thinning_curve(self.pred_segs, th = 0.5)
        return 
    
    def chm_eva(self):
        # predict_testing(self.config, self.all_patches, self.chm_model)
        predict_testing(self.config, self.chm_model)
        return 
    
    def polygons(self):
        polygons_dir = os.path.join(self.config.output_pred_dir, "polygons")
        create_polygons(self.config.output_dir, polygons_dir, self.config.output_dir, postproc_gridsize = (1, 1), postproc_workers = 40)

    
    # def pred_sampling_eva_DK(self, num = 10, sample_fixed = True):
    #     # randomly sample tifs from all files, pred and report errors
    #     # exclude training frames
    #     self.train_frames = load_train_files(self.config)
    #     # self.use_files = list(set(self.all_files) - set(self.train_frames))
        
    #     self.use_files = [i for i in self.all_files if i[1] not in self.train_frames]
    #     print('before exclude', len(self.all_files))
    #     print('exclude train', len(self.use_files))
    #     # sample randomly from the files
    #     if sample_fixed:
    #         # fix seed
    #         random.seed(1)
    #     # self.use_files_sampled = random.sample(self.use_files, num)
    #     # print('sampled', len(self.use_files_sampled))
    #     # maes = predict_large(self.config, self.use_files_sampled, self.model)
    #     # sampled based on forest type
    #     self.typefile = gps.read_file(self.config.typefile)
    #     # print(self.typefile.head())
    #     self.ft_errors = []
    #     for ty in range(3):
            
    #         print('forest type', ty)
    #         # for each forest type, sample and compute errors
    #         self.use_files_ft = [i for i in self.use_files if np.unique(self.typefile[self.typefile['location'] ==i[1]]['_majority'].values) == ty]
    #         if len(self.use_files_ft) < num:
    #             print('Samples not enough, no. tifs:', len(self.use_files_ft))
    #             self.use_files_ft_sampled = self.use_files_ft
    #             # check forest type
    #             print('checking forest type of sampled tifs')
    #             for iid in range(len(self.use_files_ft_sampled)):
    #                 print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)
                
    #             maes_ft = predict_large(self.config, self.use_files_ft_sampled, self.model)
    #             self.ft_errors.append(maes_ft)
    #         else: 
    #             maes_ft = []
    #             while len(maes_ft) != num:
    #                 self.use_files_ft_sampled = random.sample(self.use_files_ft, num)
    #                 # check forest type
    #                 print('checking forest type of sampled tifs')
    #                 for iid in range(len(self.use_files_ft_sampled)):
    #                     print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)
                    
    #                 maes_ft = predict_large(self.config, self.use_files_ft_sampled, self.model)
    #             self.ft_errors.append(maes_ft)
            
    #     plot_errors_ft(self.ft_errors)
        
    #     return self.ft_errors
        
    
    
    # def plot_testing(self):
    #     CHMerror(self.pr, self.ggt)
    #     return 
    
    
    # def pred_largescale(self):
    #     maes = predict_large(self.config, self.all_files, self.model)
    #     return maes
    
    # def preprocess(self):
    #     # # compute gb stats
    #     # self.q1s, self.q3s = compute_stat(self.config)
    #     # selected gb stats
    #     # self.q1s, self.q3s = compute_stat(self.config, self.all_files)
    #     return self.q1s, self.q3s
    
    # def plot_finland(self, q1s, q3s):
        
    #     self.pr, self.ggt = predict_finland(self.config, self.all_files, self.model, q1s, q3s)
        
    #     return 


def plot_scatter_nonlinear(x, y, title, ylabel, xlabel, ft = 30, spinexy = True, color = 'red', markersize = 2, alpha = 0.5, log = 1, fitcurve = 1):
    
    def func(x, a, b):
        return a * x + b 
    
    
    # log space
    xlog = np.log10(x)
    ylog = np.log10(y)
    popt, pcov = curve_fit(func, xlog, ylog)
    # r2 = r2_score(np.array(xlog), np.array(ylog))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xlog, ylog)

    x1 = np.array(x).copy()
    x1.sort()
    xx = [np.array(x).min(), np.array(x).max()]
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
    plt.scatter(x, y, s=markersize, c = color)
    # plt.plot(xx, xx, alpha = alpha)
    # plt.xlim(0, limix)
    # plt.ylim(0, limiy)
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.ylabel(ylabel, fontsize = 16)
    plt.xlabel(xlabel, fontsize = 16)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.title(title, fontsize = 16)
    print(xx)
    print(func(np.array(np.log10(xx)), *popt))
    if fitcurve:
        plt.plot(np.array(xx), 10**func(np.array(np.log10(xx)), *popt), 'r--', label='f(x) = %5.3f x + %5.3f\nr2 = %5.3f' % (popt[0], popt[1], r_value**2))
        if log:
            plt.yscale('log')
            plt.xscale('log')
    plt.legend()
    return
    


# Methods to add results of a patch to the total results of a larger area. The operator could be min (useful if there are too many false positives), max (useful for tackle false negatives)
def addTOResult_segcount(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
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
        currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
        mm2 = (currValue==0)
        currValue[mm2] = newPredictions[mm2]
        resultant = currValue
    else: #operator == 'REPLACE':
        resultant = newPredictions    
    res[row:row+he, col:col+wi] =  resultant
    return (res)


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
        maskseg = addTOResult_segcount(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult_segcount(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens


def detect_tree_segcount(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    """img can be one single raster or multi rasters
    
    img = [core_img, aux1, aux2, ...]
    
    or
    
    img = img #single raster
    """
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
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, config.segcount_operator)
                
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
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, config.segcount_operator)
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)
    

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

def writeMaskToDisk_segcount(detected_mask, detected_meta, wp, image_type, write_as_type = 'uint8', th = 0.5, create_countors = False, convert = 1):
    # Convert to correct required before writing
    meta = detected_meta.copy()
    if convert:
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask<th]=0
            detected_mask[detected_mask>=th]=1
        
        
        
    detected_mask = detected_mask.astype(write_as_type)
    detected_mask = detected_mask[0]
    meta['dtype'] =  write_as_type
    meta['count'] = 1
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
        create_contours_shapefile(detected_mask, detected_meta, wp)

def writeMaskToDisk_chm(detected_mask, detected_meta, wp, image_type, write_as_type = 'float32', scale = 1):
    # Convert to correct required before writing
    
    # scale by 100 and then store
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


def segcount_pred(self, th = 0.5):

    # save count per image in a dateframe
    
    counts = {}
    self.all_patches = load_files(self.config)
    outputFiles = []
    pred_segs = []
    # nochm = []
    # waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    for fullPaths in tqdm(self.all_patches):
        #print(filename)
        # print('ssssssssssssssssssssssssssssssss')
        # print(fullPaths[0][0])
        try:
            filename = os.path.basename(fullPaths[0])
        except:
            filename = os.path.basename(fullPaths[0][0])
        outputFile = os.path.join(self.config.output_pred_dir, filename.replace(self.config.input_image_pref, self.config.segcount_output_prefix).replace(self.config.input_image_type, self.config.output_image_type))
        #print(outputFile)
        outputFiles.append(outputFile)
        
        if not os.path.isfile(outputFile) or self.config.overwrite_analysed_files: 
            
            if not self.config.segcount_single_raster and self.config.segcount_aux_data: # multi raster
                try:    
                    with rasterio.open(fullPaths[0]) as core:
                        
                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(self.config, self.segcount_models, [core, fullPaths[1], fullPaths[2]], width = self.config.WIDTH, height = self.config.HEIGHT, stride = self.config.segcount_STRIDE, auxData = self.config.segcount_aux_data, singleRaster=self.config.segcount_single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        
                        
                        pred_segs.append(np.squeeze(detectedMaskSeg))
                        
                        ###### check threshold!!!!
                        # seg
                        writeMaskToDisk_segcount(detectedMaskSeg, detectedMeta, outputFile, image_type = self.config.output_image_type,  write_as_type = self.config.segcount_output_dtype, th = th, create_countors = False)
                        # print(detectedMaskDens.shape)
                        # print(detectedMaskDens.max(), detectedMaskDens.min())
                        # print(detectedMaskDens.sum())
                        # density
                        writeMaskToDisk_segcount(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = self.config.output_image_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                        counts[filename] = detectedMaskDens.sum()
                except:    
                    with rasterio.open(fullPaths[0][0]) as core:
                        
                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(self.config, self.segcount_models, [core, fullPaths[1][0], fullPaths[2][0]], width = self.config.WIDTH, height = self.config.HEIGHT, stride = self.config.segcount_STRIDE, auxData = self.config.segcount_aux_data, singleRaster=self.config.segcount_single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        
                        
                        pred_segs.append(np.squeeze(detectedMaskSeg))
                        
                        ###### check threshold!!!!
                        # seg
                        writeMaskToDisk_segcount(detectedMaskSeg, detectedMeta, outputFile, image_type = self.config.output_image_type,  write_as_type = self.config.segcount_output_dtype, th = th, create_countors = False)
                        # print(detectedMaskDens.shape)
                        # print(detectedMaskDens.max(), detectedMaskDens.min())
                        # print(detectedMaskDens.sum())
                        # density
                        writeMaskToDisk_segcount(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = self.config.output_image_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                        counts[filename] = detectedMaskDens.sum()
            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                
        else:
            print('File already analysed!', fullPaths)
    return pred_segs

def load_files(config):
    all_files = []
    for root, dirs, files in os.walk(config.output_extracted_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                m1 = os.path.join(root, file)
                m2 = m1.replace(config.input_image_pref, config.segcount_aux_prefs[0])
                m3 = m1.replace(config.input_image_pref, config.segcount_aux_prefs[1])
                all_files.append((m1, m2, m3))
    print('Number of raw tif to predict:', len(all_files))
    # print(all_files)
    return all_files

def pred_chm_rawtif_ndvi(config, model, img, channels, width=256, height=256, stride = 128, normalize=1, maxnorm = 0):
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
        temp_im = temp_im / 255
        temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
        
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        
        patch[:window.height, :window.width] = temp_im
        batch.append(patch)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model_chm(model, batch, batch_pos, mask, config.chm_operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model_chm(model, batch, batch_pos, mask, config.chm_operator)
        batch = []
        batch_pos = []
        
    if maxnorm:
        mask = mask * 97.19
    return(mask, meta)

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
        mask = addTOResult_chm(mask, p, row, col, he, wi, operator)
    return mask


def predict_testing(config, model):
    
    outputFiles = []
    pr = np.array([])
    ggt = np.array([])
    all_patches = load_files(config)
    for fullPaths in all_patches:
        # print(fp1, fp2, chmfp)
        #print(filename)
        
        try:
            filename = os.path.basename(fullPaths[0])
        except:
            filename = os.path.basename(fullPaths[0][0])
        outputFile = os.path.join(config.output_pred_dir, filename.replace(config.input_image_pref, config.chm_output_prefix))
        #print(outputFile)
        outputFiles.append(outputFile)
        outputChmDiff = os.path.join(config.output_pred_dir, filename.replace(config.input_image_pref, config.chm_diff_prefix))
        
        try:    
            with rasterio.open(fullPaths[0]) as im:
                
                chmPath = fullPaths[2]
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                
                
                    #print(fullPath)
                    detectedMask, detectedMeta = pred_chm_rawtif_ndvi(config, model, im, config.chm_channels, width = config.WIDTH, height = config.HEIGHT, stride = config.chm_STRIDE, maxnorm = config.chm_maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    
                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                    if config.saveresult:
                        writeMaskToDisk_chm(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.chm_output_dtype)
                    # compute diff
                    # CHMerror(detectedMask, chm, filename)
                    chmdiff, _, _ = CHMdiff(detectedMask, chm, filename)
                    if config.saveresult:
                        writeMaskToDisk_chm(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chm_chmdiff_dtype, scale = 0)
                    pr = np.append(pr,  detectedMask.flatten())
                    ggt = np.append(ggt, np.squeeze(chm.read()).flatten())
        except:
            with rasterio.open(fullPaths[0][0]) as im:
                
                chmPath = fullPaths[2][0]
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                
                
                    #print(fullPath)
                    detectedMask, detectedMeta = pred_chm_rawtif_ndvi(config, model, im, config.chm_channels, width = config.WIDTH, height = config.HEIGHT, stride = config.chm_STRIDE, maxnorm = config.chm_maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    
                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
                    if config.saveresult:
                        writeMaskToDisk_chm(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.chm_output_dtype)
                    # compute diff
                    # CHMerror(detectedMask, chm, filename)
                    chmdiff, _, _ = CHMdiff(detectedMask, chm, filename)
                    if config.saveresult:
                        writeMaskToDisk_chm(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chm_chmdiff_dtype, scale = 0)
                    pr = np.append(pr,  detectedMask.flatten())
                    ggt = np.append(ggt, np.squeeze(chm.read()).flatten())
    
    MAE = mean_absolute_error(ggt, pr)
    print('MAE for all', MAE)
    return pr, ggt


def create_vector_vrt(vrt_out_fp, layer_fps, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file.
    Concatenates several vector files in a single VRT file with OGRVRTUnionLayers.
    Layer file paths are stored as relative paths, to allow copying of the VRT file with all its data layers.
    """
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty VRT file, skipping: {vrt_out_fp}")

    xml = f'<OGRVRTDataSource>\n' \
          f'    <OGRVRTUnionLayer name="{out_layer_name}">\n'
    for layer_fp in tqdm(layer_fps, desc="Creating VRT", disable=not pbar):
        shapefile = ogr.Open(layer_fp)
        layer = shapefile.GetLayer()
        relative_path = layer_fp.replace(f"{os.path.join(os.path.dirname(vrt_out_fp), '')}", "")
        xml += f'        <OGRVRTLayer name="{os.path.basename(layer_fp).split(".")[0]}">\n' \
               f'            <SrcDataSource relativeToVRT="1">{relative_path}</SrcDataSource>\n' \
               f'            <SrcLayer>{layer.GetName()}</SrcLayer>\n' \
               f'            <GeometryType>wkb{ogr.GeometryTypeToName(layer.GetGeomType())}</GeometryType>\n' \
               f'        </OGRVRTLayer>\n'
    xml += '    </OGRVRTUnionLayer>\n' \
           '</OGRVRTDataSource>\n'
    with open(vrt_out_fp, "w") as file:
        file.write(xml)

def create_vector_gpkg(out_fp, layer_fps, crs, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file.
    Concatenates several vector files in a single VRT file with OGRVRTUnionLayers.
    Layer file paths are stored as relative paths, to allow copying of the VRT file with all its data layers.
    """
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty VRT file, skipping: {vrt_out_fp}")

    dfall = gps.GeoDataFrame()
    for layer_fp in tqdm(layer_fps, desc="Creating GPKG", disable=not pbar):
        # shapefile = ogr.Open(layer_fp)
        # layer = shapefile.GetLayer()
        df = gps.read_file(layer_fp, layer='trees')
        dfall = dfall.append(df)
        
    # with open(out_fp, "w") as file:
    #     file.write(dfall)
    dfall.to_file(out_fp, driver="GPKG", crs = crs, layer="trees")
    return
    
def polygonize_chunk(params):
    """Polygonize a single window chunk of the raster image."""

    raster_fp, raster_chm, out_fp, window = params
    polygons = []
    with rasterio.open(raster_fp) as src:
        raster_crs = src.crs
        for feature, _ in shapes(src.read(window=window), src.read(window=window), 4,
                                 src.window_transform(window)):
            polygons.append(shape(feature))
    if len(polygons) > 0:
        df = gps.GeoDataFrame({"geometry": polygons})
        df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="trees")
        
        # # read chm raster window
        # with rasterio.open(raster_chm) as src2:
        #     data = src2.read(window=window)
        #     print(data.shape)
        #     profile = src2.profile
        #     profile.height = data.shape[2]
        #     profile.width = data.shape[1]
        #     profile.transform = src2.window_transform(window)
        #     with rasterio.open("/vsimem/temp.tif", "w", **profile) as dst:
        #         dst.write(data)
        
            
        #     heights = zonal_stats(out_fp, "/vsimem/temp.tif",
        #         stats="max")
        #     df["max_height"] = heights
        #     print(df.head())
        #     df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="heights")
        
        return out_fp
    

def create_polygons(raster_dir, polygons_basedir, postprocessing_dir, postproc_gridsize = (2, 2), postproc_workers = 40):
    """Polygonize the raster to a vector polygons file.
    Because polygonization is slow and scales exponentially with image size, the raster is split into a grid of several
    smaller chunks, which are processed in parallel.
    As vector file merging is also very slow, the chunks are not merged into a single vector file, but instead linked in
    a virtual vector VRT file, which allows the viewing in QGIS as a single layer.
    """

    # Polygonise all raster predictions
    # for DK
    raster_fps = glob.glob(f"{raster_dir}/det_1km*.tif")
    # # for FI
    # raster_fps = glob.glob(f"{raster_dir}/*det_seg.tif")
    raster_chms = [i.replace('det_seg', 'det_CHM') for i in raster_fps]
    print('seg masks for polygonization:', raster_fps)
    print('chm masks:', raster_chms)
    for ind in tqdm(range(len(raster_fps))):

        # Create a folder for the polygons VRT file, and a sub-folder for the actual gpkg data linked in the VRT
        prediction_name = os.path.splitext(os.path.basename(raster_fps[ind]))[0]
        polygons_dir = os.path.join(polygons_basedir, prediction_name)
        if os.path.exists(polygons_dir):
            print(f"Skipping, already processed {polygons_dir}")
            continue
        # os.mkdir(polygons_dir)
        os.makedirs(polygons_dir)
        os.mkdir(os.path.join(polygons_dir, "vrtdata"))

        # Create a list of rasterio windows to split the image into a grid of smaller chunks
        chunk_windows = []
        n_rows, n_cols = postproc_gridsize
        with rasterio.open(raster_fps[ind]) as raster:
            raster_crs = raster.crs
            width, height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                out_fp = os.path.join(polygons_dir, "vrtdata", f"{prediction_name}_{width * j}_{height * i}.gpkg")
                chunk_windows.append([raster_fps[ind], raster_chms[ind], out_fp, Window(width * j, height * i, width, height)])

        # Polygonise image chunks in parallel
        polygon_fps = []
        
        with multiprocessing.Pool(processes=postproc_workers) as pool:
            with tqdm(total=len(chunk_windows), desc="Polygonising raster chunks", position=1, leave=False) as pbar:
                for out_fp in pool.imap_unordered(polygonize_chunk, chunk_windows):
                    if out_fp:
                        polygon_fps.append(out_fp)
                    # dfall = dfall.append(df)
                    pbar.update()
                    
        # Merge all polygon chunks into one polygon VRT
        create_vector_vrt(os.path.join(polygons_dir, f"polygons_{prediction_name}.vrt"), polygon_fps)
        out_dfall = os.path.join(polygons_dir, f"{prediction_name}_all_seg.gpkg")
        # dfall.to_file(out_dfall, driver="GPKG", crs=raster_crs, layer="trees")
        create_vector_gpkg(out_dfall, polygon_fps, raster_crs, out_layer_name="trees", pbar=False)
        
    # Create giant VRT of all polygon VRTs
    merged_vrt_fp = os.path.join(polygons_basedir, f"all_polygons_{os.path.basename(postprocessing_dir)}.vrt")
    create_vector_vrt(merged_vrt_fp, glob.glob(f"{polygons_basedir}/*/*.vrt"))

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
    
def self_thinning_curve(pred_segs, th = 0.5):
    """self thinning: log density vs log avg crown area per plot"""
    
    crown_area_avgs = []
    tree_density = []
    for i in range(len(pred_segs)):
        curmask = pred_segs[i]
        curmask[curmask<th]=0
        curmask[curmask>=th]=1
        # curlb = gts[i][:, :, 0].astype(np.uint8)
        pred_seg = curmask.astype(np.uint8)
        # lbc = curlb.copy()
        contours, hierarchy = cv2.findContours(pred_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        crown_areas = []
        for c in contours:
            # cur = np.zeros(curlb.shape)
            area = cv2.contourArea(c)
            # cv2.drawContours(cur, [c], -1, (255,255,255), -1)
            # pred_con = pred.copy()
            # cur[cur == 255] = 1
            # pred_con = pred_con*cur
            # area_pred = pred_con.sum()
            
            crown_areas.append(area)
            # # plot figures
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(cur)
            # plt.title(str(area_gt))
            # plt.subplot(122)
            # plt.imshow(pred_con)
            # plt.title(str(area_pred))
        assert len(pred_seg.shape) == 2
        tot_area = pred_seg.size
        # conver to trees/hecter
        tot_area = tot_area * 0.04
        tree_density.append((len(contours)/tot_area)*10000)
        avg = np.array(crown_areas).mean()
        crown_area_avgs.append(avg*0.04)
    return tree_density, crown_area_avgs
