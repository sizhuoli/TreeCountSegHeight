#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:31:21 2021

@author: sizhuo
"""



import os
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
import fiona                     # I/O vector data (shape, geojson, ...)
# import geopandas as gps

from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape

import numpy as np               # numerical array manipulation
# import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
# import time
from skimage.transform import resize

from itertools import product
from tensorflow.keras.models import load_model
import cv2
import tensorflow.keras.backend as K
import copy 
# from sklearn.metrics import r2_score
# import sys

# from core.UNet_multires import UNet
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core.eva_losses import eva_acc, eva_dice, eva_sensitivity, eva_specificity, eva_miou
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info_multires import FrameInfo, image_normalize
# from core.dataset_generator_multires import DataGenerator
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

from collections import defaultdict

from rasterio.enums import Resampling
from scipy.optimize import curve_fit
import scipy
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))



class eva_segcount:
    def __init__(self, config):
        self.config = config
        OPTIMIZER = adam #
        self.models = []
        for mod in self.config.trained_model_paths:
            modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            modeli.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_count':'mse'},
                            metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                                'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
            self.models.append(modeli)
            modeli.summary()
            ############################################3
            # # try different input shape
            # modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            
            # modeli.layers[0]._batch_input_shape = (None, 512, 512, 6)
            # # loaded_model.summary()
            # # model_config = loaded_model.get_config()
            
            # new_model = tf.keras.models.model_from_json(modeli.to_json()) 
            # new_model.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_count':'mse'},
            #                metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
            #                    'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
            # print(new_model.summary())
            # self.models.append(new_model)
            ############################################3
            
        
        self.all_files = load_files(self.config)
        
        # if savefig
        # savedir = './savefig/inrgb3/'
        # if not os.path.exists(savedir):
        #     os.makedirs(savedir)
    
    def pred(self, thr = 0.5, save = 0):
        if save == 0:
            self.outputSeg, self.pred_labels, self.outputDens, self.pred_counts, ssss = predict_segcount(self.all_files, self.config, self.models, thr)
        elif save == 1:
            self.outputSeg, self.pred_labels, self.outputDens, self.pred_counts = predict_segcount_save(self.all_files, self.config, self.models, thr)

    def report_seg(self, thres = 2, plot = 0, savefig = 0):
        self.gtseg, self.gtdens = load_truths_segcount(self.all_files, self.config)
        # if plot == 1 and savefig == 1: 
        #     report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = 2, plot = 1, modeln = 'Model', savefig = 1, savename = savedir + 'Model')

        # if not savefig but display images
        if plot == 1:
            report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = thres, plot = 1, modeln = 'Model')
        
        # if no plot
        else:
            c_all, c_nosmall, c_gt, self.clear_ps = report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = thres, plot = 0)
            c_gt_ha = []
            c_nosmall_ha = []
            for i in range(len(self.clear_ps)):
                # compute tree density tree (trees/ha)
                tot_area = self.clear_ps[i].size * 0.04 # in m**2
                c_gt_d = (c_gt[i] / tot_area) * 10000 # no trees/ha
                c_nosmall_d = (c_nosmall[i] / tot_area) * 10000 # no trees/ha
                c_gt_ha.append(c_gt_d)
                c_nosmall_ha.append(c_nosmall_d)
                
            
            return c_all, c_nosmall, c_gt, c_gt_ha, c_nosmall_ha, self.clear_ps
    
    def segcount_save(self):
        outputSeg, pred_labels, outputDens, pred_counts = predict_segcount_save(self.all_files, self.config, self.models)
        return 
    
    def report_count(self):
        # density count (count from density branch)
        for mm in range(len(self.config.trained_model_paths)):
            print('**********model************', mm)
            ttc = 0
            cgt = 0
            
            ttlist = []
            predlist = []
            for i in range(len(self.pred_counts)):
                # print('gt', gtdens[i].sum())
                ttlist.append(self.gtdens[i].sum())
                # print('pred', pred_counts[i][mm])
                predlist.append(self.pred_counts[i][mm])
                ttc+=self.pred_counts[i][mm]
                cgt += self.gtdens[i].sum()
            print('pred count', ttc)
            print('ground truth count', cgt)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(ttlist), np.array(predlist))
            print('r2 of counts', r_value**2)
            # print('r2 score: ', r2_score(np.array(ttlist), np.array(predlist)))
        return ttlist, predlist
    
    def report_count_density(self):
        # (no trees / ha) pred vs gt
        for mm in range(len(self.config.trained_model_paths)):
            ttc = 0
            cgt = 0
            
            ttlist = []
            predlist = []
            for i in range(len(self.pred_counts)):
                print('------------------------------------------')
                # print('gt', gtdens[i].sum())
                curcount_gt = self.gtdens[i].sum()
                print('gt', curcount_gt)
                print(self.gtdens[i].size)
                tot_area = self.gtdens[i].size * 0.04 # in m**2
                print(tot_area)
                # gt_density
                gt_count_density = (curcount_gt / tot_area) * 10000 # no trees/ha
                print(gt_count_density)
                ttlist.append(gt_count_density)
                curcount_pred = self.pred_counts[i][mm]
                print('pred', curcount_pred)
                # pred_density
                pred_count_density = (curcount_pred / tot_area) * 10000 # no trees/ha
                print(pred_count_density)
                predlist.append(pred_count_density)
                ttc+=self.pred_counts[i][mm]
                cgt += self.gtdens[i].sum()
            print('pred count', ttc)
            print('ground truth count', cgt)
            
            
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(ttlist), np.array(predlist))
            # print('r2 score: ', r2_score(np.array(ttlist), np.array(predlist)))
            print('r2 score: ', r_value**2)
        return ttlist, predlist
        
        
    def tree_crown(self):
        crowns = tree_sensitivity(self.clear_ps, self.gtseg)
        return crowns
    
    def self_thin(self):
        tree_density, avg_crown = self_thinning_curve(self.outputDens, self.clear_ps)
        
        return tree_density, avg_crown
class eva_seg:
    def __init__(self, config):
        self.config = config
        OPTIMIZER = adam #
        self.models = []
       
        for mod in self.config.trained_model_paths:
            modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            modeli.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            self.models.append(modeli)
            
        self.all_files = load_files(self.config)
        
        # if savefig
        # savedir = './savefig/inrgb3/'
        # if not os.path.exists(savedir):
        #     os.makedirs(savedir)
    
    def pred(self, save = 0):
        if save == 0:
            self.outputMask, self.pred_labels = predict(self.all_files, self.config, self.models)
        elif save == 1:
            self.outputMask, self.pred_labels = predict_save(self.all_files, self.config, self.models)
    def report_seg(self, thres = 2, plot = 0, savefig = 0):
        self.gts = load_truths(self.all_files, self.config)
        # if plot == 1 and savefig == 1: 
        #     report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = 2, plot = 1, modeln = 'Model', savefig = 1, savename = savedir + 'Model')

        # if not savefig but display images
        # if plot == 1:
        #     report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = thres, plot = 1, modeln = 'Model')
        
        # if no plot
        if plot == 0:
            c_all, c_nosmall, c_gt, self.clear_p = report(self.models, self.pred_labels, self.config, self.all_files, self.gts, thres = thres, plot = 0)
        
        
        c_gt_ha = []
        c_nosmall_ha = []
        for i in range(len(self.clear_p)):
            # compute tree density tree (trees/ha)
            tot_area = self.clear_p[i].size * 0.04 # in m**2
            c_gt_d = (c_gt[i] / tot_area) * 10000 # no trees/ha
            c_nosmall_d = (c_nosmall[i] / tot_area) * 10000 # no trees/ha
            c_gt_ha.append(c_gt_d)
            c_nosmall_ha.append(c_nosmall_d)
        
            
        return c_all, c_nosmall, c_gt, c_gt_ha, c_nosmall_ha, self.clear_p
    
    def tree_crown(self):
        crowns = tree_sensitivity(self.clear_p, self.gts)
        return crowns
    
def plot_scatter(x, y, title, ylabel, xlabel, limi, spinexy = True, markersize = 2, alpha = 0.5):
    
    def func(x, a, b):
        return a * x + b 
    
    popt, pcov = curve_fit(func, x, y)
    # r2 = r2_score(np.array(x), np.array(y))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    x1 = np.array(x).copy()
    x1.sort()
    xx = [0, limi]
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
    plt.scatter(x, y, s=markersize)
    plt.plot(xx, xx, alpha = alpha)
    plt.xlim(0, limi)
    plt.ylim(0, limi)
    plt.ylabel(ylabel, fontsize = 16)
    plt.xlabel(xlabel, fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize = 16)
    plt.plot(np.array(xx), func(np.array(xx), *popt), 'r--', label='f(x) = %5.3f x + %5.3f\nr2 = %5.3f' % (popt[0], popt[1], r_value**2))
    plt.legend()
    return

def plot_scatter_nonlinear(x, y, title, ylabel, xlabel, limi, spinexy = True, markersize = 2, alpha = 0.5, log = 1):
    
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
    plt.scatter(x, y, s=markersize)
    # plt.plot(xx, xx, alpha = alpha)
    # plt.xlim(0, limi)
    # plt.ylim(0, limi)
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.ylabel(ylabel, fontsize = 16)
    plt.xlabel(xlabel, fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize = 16)
    print(xx)
    print(func(np.array(np.log10(xx)), *popt))
    plt.plot(np.array(xx), 10**func(np.array(np.log10(xx)), *popt), 'r--', label='f(x) = %5.3f x + %5.3f\nr2 = %5.3f' % (popt[0], popt[1], r_value**2))
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.legend()
    return
    

# def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
#     currValue = res[row:row+he, col:col+wi]
    
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(currValue)
#     newPredictions = prediction[:he, :wi]
    
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(newPredictions)
    
# # IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
#     if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
#         currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
#         resultant = np.minimum(currValue, newPredictions) 
#     elif operator == 'MAX':
#         resultant = np.maximum(currValue, newPredictions)
#     elif operator == "MIX": # alpha blending # note do not combine with empty regions
#         # print('mix')    
#         mm1 = currValue!=0
#         currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
#         mm2 = (currValue==0)
#         currValue[mm2] = newPredictions[mm2]
#         resultant = currValue
#     else: #operator == 'REPLACE':
#         resultant = newPredictions    
# # Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# # We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# # So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get. 
# # However, in case the values are strecthed before hand this problem will be minimized
#     res[row:row+he, col:col+wi] =  resultant
    
    
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(resultant)
    
#     return (res)

def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX', dens = 0):
    # print(row, col, he, wi)
    # print(res.shape)
    currValue = res[row:row+he, col:col+wi]
    # print(currValue.shape)
    # plt.figure(figsize = (10,10))
    # plt.imshow(currValue)
    newPredictions = prediction[:he, :wi]
    
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

def addTOResult_pad_reflect(res, prediction, row, col, he, wi, operator = 'MAX', dens = 0):
    # print(row, col, he, wi)
    # print(res.shape)
    currValue = res[row:row+he+14, col:col+wi+14]
    # print(currValue.shape)
    # plt.figure(figsize = (10,10))
    # plt.imshow(currValue)
    newPredictions = prediction[:he, :wi]
    # print('np', newPredictions.shape)
    # plt.figure(figsize = (10,10))
    # plt.imshow(newPredictions)
    # print(newPredictions.sum())
    mask = np.pad(newPredictions, (7, 7), 'edge')
    mask[7:mask.shape[0]-7, 7:mask.shape[1]-7] = 1
    if dens:
        mask[mask<1] *= 100
    newPredictions2 = np.pad(newPredictions, (7, 7), 'reflect')
    nn = mask * newPredictions2
    # plt.figure(figsize = (10,10))
    # plt.imshow(newPredictions2)
    # print('n2', newPredictions2.shape)
    # plt.figure(figsize = (10,10))
    # plt.imshow(nn) 
    # print('nn', nn.shape)
    
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, nn) 
    elif operator == 'MAX':
        resultant = np.maximum(currValue, nn)
    elif operator == "MIX": # alpha blending # note do not combine with empty regions
        # print('mix')    
        mm1 = currValue!=0
        currValue[mm1] = currValue[mm1] * 0.5 + nn[mm1] * 0.5
        mm2 = (currValue==0)
        currValue[mm2] = nn[mm2]
        resultant = currValue
    else: #operator == 'REPLACE':
        resultant = nn  
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get. 
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he+14, col:col+wi+14] =  resultant
    
    
    # plt.figure(figsize = (10,10))
    # plt.imshow(resultant)
    
    return (res)

def predict_using_model(model, batch, batch_pos, mask, operator):
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
    
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask



def predict_using_model_chm(model, batch, batch_pos, mask, operator, upsample = 1, downsave = 1):
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
        # no preserve range
        # print('**********************UPsample for prediction***************************')
        # print('**********************Upsample for saving***********************')
        # # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
        # p = resize(p[:, :], (int(p.shape[0]*2), int(p.shape[1]*2)), preserve_range=True)
        # print('mean after resize', p.mean(), p.std(), p.min(), p.max())
        # print('p shape', p.shape)
        ###############33#############################################
        # print('p shape', p.shape)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        # print('p shape', p.shape)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask




# 2 tasks
def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator):
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
        seg, dens = model.predict([tm1, tm2]) # tm is a list []
        
    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        seg, dens = model.predict(tm1) # tm is a list []
        try:
            print(seg.shape)
        except:
            seg = model.predict(tm1)['output_seg']
            dens = model.predict(tm1)['output_dens']
            # print('******************************')
            # print(seg.shape, dens.shape)
            # print('******************************')
        
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator, dens = 0)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator, dens = 1)
    return maskseg, maskdens

#3 tasks
def predict_using_model_segcountchm(model, batch, batch_pos, maskseg, maskdens, maskchm, operator):
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
        seg, dens, chm = model.predict([tm1, tm2]) # tm is a list []
        
    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        seg, dens, chm = model.predict(tm1) # tm is a list []
        
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        ch = np.squeeze(chm[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
        maskchm = addTOResult(maskchm, ch, row, col, he, wi, operator)
    return maskseg, maskdens, maskchm


def predict_using_model_single_input(model, batch, batch_pos, mask, operator):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    # tm1 = []
    # tm2 = []
    # for p in batch:
    #     tm1.append(p[0]) # (256, 256, 5)
        # tm2.append(p[1]) # (128, 128, 1)
    tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
    # tm2 = np.stack(tm2, axis = 0)
    prediction = model.predict(tm1) # tm is a list []
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask


def detect_tree(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masks = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask = masks[mi, :, :]
                
                curmask = predict_using_model(models[mi], batch, batch_pos, curmask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask = masks[mi, :, :]
            curmask = predict_using_model(models[mi], batch, batch_pos, curmask, config.operator)
        batch = []
        batch_pos = []
    return masks

def detect_tree_chm(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masks = np.zeros((len(models), int(nrows/2), int(nols/2)), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask = masks[mi, :, :]
                
                curmask = predict_using_model_chm(models[mi], batch, batch_pos, curmask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask = masks[mi, :, :]
            curmask = predict_using_model_chm(models[mi], batch, batch_pos, curmask, config.operator)
        batch = []
        batch_pos = []
    return masks

def detect_tree_save(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    meta = img0.meta.copy()
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masks = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask = masks[mi, :, :]
                
                curmask = predict_using_model(models[mi], batch, batch_pos, curmask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask = masks[mi, :, :]
            curmask = predict_using_model(models[mi], batch, batch_pos, curmask, config.operator)
        batch = []
        batch_pos = []
    return (masks, meta)

def detect_tree_segcount(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    
    ssss = []
    
    
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
    maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            # print('using normalize training data')
            # temp_im1 = (temp_im1 - np.array([[118.66, 121.69, 107.15, 178.43, 0]]))/ np.array([[29.32, 26.32, 25.42, 24.28, 1]])
        
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
                # temp_im2 = (temp_im2 - np.array([[3.5]])) / np.array([[3]])
                
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        
        
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
                
                if len(ssss) < 5:
                    ssss.append([patch1, patch2])
                
                
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
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
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, ssss



def detect_tree_segcount_pad_reflect(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    
    ssss = []
    
    
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masksegs = np.zeros((len(models), nrows+14, nols+14), dtype=np.float32)
    maskdenss = np.zeros((len(models), nrows+14, nols+14), dtype=np.float32)
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            # print('using normalize training data')
            # temp_im1 = (temp_im1 - np.array([[118.66, 121.69, 107.15, 178.43, 0]]))/ np.array([[29.32, 26.32, 25.42, 24.28, 1]])
        
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
                # temp_im2 = (temp_im2 - np.array([[3.5]])) / np.array([[3]])
                
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        
        
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
                
                if len(ssss) < 5:
                    ssss.append([patch1, patch2])
                
                
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
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
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, ssss


def detect_tree_segcount_save_pad_reflect(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):

    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
    # nrows, nols = 256, 256 # base shape # rasterio read channel first
        
    meta = img0.meta.copy()
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    
    # masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
    # maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masksegs = np.zeros((len(models), nrows+14, nols+14), dtype=np.float32)
    maskdenss = np.zeros((len(models), nrows+14, nols+14), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        # print('coloff', col_off)
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM or det chm
                if not config.detchm: # reference chm, lower resolution
                    patch1 = np.zeros((height, width, len(img))) # except for the last channel
                
                    
                    # print('0 shape', temp_im1.shape)
                
                    for ch in range(1, len(img)-1): # except for the last channel 
                        
                        imgi = rasterio.open(img[ch]) 
                        
                        sm1 = imgi.read(
                                        out_shape=(
                                        1,
                                        int(window.height),
                                        int(window.width)
                                    ),
                                    resampling=Resampling.bilinear, window = window)
                        # print('aux shape', aux_sm.shape)
                        
                        temp_im1 = np.row_stack((temp_im1, sm1))
                    
                    # upsample the CHM channel
                    chmim = rasterio.open(img[-1])
                    meta_chm = chmim.meta.copy()
                    hei_ratio = nrows/meta_chm['height']
                    wid_ratio = nols/meta_chm['width']
                    res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                    window.width / wid_ratio, window.height / hei_ratio)
                
                    chm_sm = chmim.read(
                                    out_shape=(
                                    chmim.count,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = res_window)
                    # print('aux shape', aux_sm.shape)
                    temp_im1 = np.row_stack((temp_im1, chm_sm))
                    print(temp_im1.mean())
                else:
                    print('using det chm')
                    patch1 = np.zeros((height, width, len(img))) 
                
                    for ch in range(1, len(img)): 
                        
                        imgi = rasterio.open(img[ch]) 
                        
                        sm1 = imgi.read(
                                        out_shape=(
                                        1,
                                        int(window.height),
                                        int(window.width)
                                    ),
                                    resampling=Resampling.bilinear, window = window)
                        # print('aux shape', aux_sm.shape)
                        
                        temp_im1 = np.row_stack((temp_im1, sm1))
                    
                    
                        
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            if CHM:
                if multires:
                    print('multires chm, normalize chm layer')
                    temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    temp_im2 = image_normalize(temp_im2, axis=(0,1))
                elif not multires:
                    # temp_im1 = image_normalize(temp_im1, axis=(0,1))
                    print('det chm, norm chm layer')
                    # # normalize all color bands
                    # temp_im1[..., :-1] = image_normalize(temp_im1[..., :-1], axis=(0,1))
                    # # maxmin norm height layer
                    # temp_im1[..., -1] = temp_im1[..., -1]/30
                    # temp_im1[..., -1][temp_im1[..., -1]>2]=0
                    # temp_im1[..., -1][temp_im1[..., -1]<0.03]=0
                    # print('processed bands',  temp_im1.min(axis=(0,1)),temp_im1.max(axis=(0,1)), temp_im1.mean(axis=(0,1)))
                    
                    # normalize all bands
                    temp_im1 = image_normalize(temp_im1, axis=(0,1))
                    
                    
                    
            elif not CHM:
                temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    
                    
                    
                    
                    
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
        # print('window colrow wi he', window.col_off, window.row_off, window.width, window.height)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        # plt.figure(figsize = (10,10))
        # plt.imshow(temp_im1[:,:,0])
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
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
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta
        

def detect_tree_segcount_save(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):

    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
    # nrows, nols = 256, 256 # base shape # rasterio read channel first
        
    meta = img0.meta.copy()
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    
    # masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
    # maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
    maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        # print('coloff', col_off)
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM or det chm
                if not config.detchm: # reference chm, lower resolution
                    patch1 = np.zeros((height, width, len(img))) # except for the last channel
                
                    
                    # print('0 shape', temp_im1.shape)
                
                    for ch in range(1, len(img)-1): # except for the last channel 
                        
                        imgi = rasterio.open(img[ch]) 
                        
                        sm1 = imgi.read(
                                        out_shape=(
                                        1,
                                        int(window.height),
                                        int(window.width)
                                    ),
                                    resampling=Resampling.bilinear, window = window)
                        # print('aux shape', aux_sm.shape)
                        
                        temp_im1 = np.row_stack((temp_im1, sm1))
                    
                    # upsample the CHM channel
                    chmim = rasterio.open(img[-1])
                    meta_chm = chmim.meta.copy()
                    hei_ratio = nrows/meta_chm['height']
                    wid_ratio = nols/meta_chm['width']
                    res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                    window.width / wid_ratio, window.height / hei_ratio)
                
                    chm_sm = chmim.read(
                                    out_shape=(
                                    chmim.count,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = res_window)
                    # print('aux shape', aux_sm.shape)
                    temp_im1 = np.row_stack((temp_im1, chm_sm))
                    print(temp_im1.mean())
                else:
                    print('using det chm')
                    patch1 = np.zeros((height, width, len(img))) 
                
                    for ch in range(1, len(img)): 
                        
                        imgi = rasterio.open(img[ch]) 
                        
                        sm1 = imgi.read(
                                        out_shape=(
                                        1,
                                        int(window.height),
                                        int(window.width)
                                    ),
                                    resampling=Resampling.bilinear, window = window)
                        # print('aux shape', aux_sm.shape)
                        
                        temp_im1 = np.row_stack((temp_im1, sm1))
                    
                    
                        
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            if CHM:
                if multires:
                    print('multires chm, normalize chm layer')
                    temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    temp_im2 = image_normalize(temp_im2, axis=(0,1))
                elif not multires:
                    # temp_im1 = image_normalize(temp_im1, axis=(0,1))
                    print('det chm, norm chm layer')
                    # # normalize all color bands
                    # temp_im1[..., :-1] = image_normalize(temp_im1[..., :-1], axis=(0,1))
                    # # maxmin norm height layer
                    # temp_im1[..., -1] = temp_im1[..., -1]/30
                    # temp_im1[..., -1][temp_im1[..., -1]>2]=0
                    # temp_im1[..., -1][temp_im1[..., -1]<0.03]=0
                    # print('processed bands',  temp_im1.min(axis=(0,1)),temp_im1.max(axis=(0,1)), temp_im1.mean(axis=(0,1)))
                    
                    # normalize all bands
                    temp_im1 = image_normalize(temp_im1, axis=(0,1))
                    
                    
                    
            elif not CHM:
                temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    
                    
                    
                    
                    
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
        # print('window colrow wi he', window.col_off, window.row_off, window.width, window.height)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        # plt.figure(figsize = (10,10))
        # plt.imshow(temp_im1[:,:,0])
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
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
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta



def detect_tree_segcountchm(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    
    
    """
    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
    maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)
    maskchms = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        temp_im1 = img[0].read(window=window)
        
        if CHM: 
            if multires: # 2 inputs
                print('multires')
                patch1 = np.zeros((height, width, len(img)-1)) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
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
                
                temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
            elif not multires: # upsample CHM
                print('upsampling')
                patch1 = np.zeros((height, width, len(img))) # except for the last channel
            
                
                # print('0 shape', temp_im1.shape)
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    # print('aux shape', aux_sm.shape)
                    
                    temp_im1 = np.row_stack((temp_im1, sm1))
                
                # upsample the CHM channel
                chmim = rasterio.open(img[-1])
                meta_chm = chmim.meta.copy()
                hei_ratio = nrows/meta_chm['height']
                wid_ratio = nols/meta_chm['width']
                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)
            
                chm_sm = chmim.read(
                                out_shape=(
                                chmim.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, chm_sm))
                print(temp_im1.mean())
            
        elif not CHM:
            patch1 = np.zeros((height, width, len(img)))
            # print('PATCH shape', patch1.shape)
        
            
            # print('0 shape', temp_im1.shape)
        
            for ch in range(1, len(img)): 
                
                imgi = rasterio.open(img[ch]) 
                
                sm1 = imgi.read(
                                out_shape=(
                                1,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = window)
                # print('aux shape', aux_sm.shape)
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            if CHM and multires:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            if multires:
                patch2[:window2.height, :window2.width] = temp_im2
                batch.append([patch1, patch2])
            else:
                batch.append(patch1)
            
        elif not CHM:
            batch.append(patch1)
            
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                curmaskchm = maskchms[mi, :, :]
                
                curmaskseg, curmaskdens, curmaskchm = predict_using_model_segcountchm(models[mi], batch, batch_pos, curmaskseg, curmaskdens, curmaskchm, config.operator)
                
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
            curmaskchm = maskchms[mi, :, :]
                
            curmaskseg, curmaskdens, curmaskchm = predict_using_model_segcountchm(models[mi], batch, batch_pos, curmaskseg, curmaskdens, curmaskchm, config.operator)
                
               
        batch = []
        batch_pos = []
    return masksegs, maskdenss, maskchms

def detect_tree_single_input(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    """img can be one single raster or multi rasters
    
    img = [c1, c2, c3, c4, ..] (channels can have different dimensions)
    
    or
    
    img = img #single raster
    """
    
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    # print(read_img0.shape)
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
        
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    # mask = np.zeros((nrows, nols), dtype=np.float32)
    masks = np.zeros((len(models), nrows, nols), dtype=np.float32)
    
    
    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        
        
        patch1 = np.zeros((height, width, len(img))) # except for the last channel
        
        temp_im1 = img[0].read(window=window)
        # print('0 shape', temp_im1.shape)
        for ch in range(1, len(img)): # except for the last channel 
            
            imgi = rasterio.open(img[ch]) 
            
            sm1 = imgi.read(
                            out_shape=(
                            1,
                            int(window.height),
                            int(window.width)
                        ),
                        resampling=Resampling.bilinear, window = window)
            # print('aux shape', aux_sm.shape)
            
            temp_im1 = np.row_stack((temp_im1, sm1))
            
        # for the 2nd input source
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
        
        # # stack makes channel first
        # # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        # temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
        
        # print(temp_im1.shape, temp_im2.shape)
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            # temp_im2 = image_normalize(temp_im2, axis=(0,1))
        
        patch1[:window.height, :window.width] = temp_im1
        # patch2[:window2.height, :window2.width] = temp_im2
        batch.append(patch1)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask = masks[mi, :, :]
                
                curmask = predict_using_model_single_input(models[mi], batch, batch_pos, curmask, config.operator)
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask = masks[mi, :, :]
            curmask = predict_using_model_single_input(models[mi], batch, batch_pos, curmask, config.operator)
        batch = []
        batch_pos = []
    return masks
    
  
def gen_label(masks, thr = 0.5):
    mm = copy.deepcopy(masks)
    print('threshold', thr)
    for m in mm:
        m[m<thr]=0
        m[m>=thr]=1
        # m = m.astype(np.int8)
        # print(m.dtype)\
    
    return mm

def load_files(config):
    exclude = set(['density', 'size', 'species', 'figs', 'test', 'val'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.channel_names[0]):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw image to predict:', len(all_files))
    print(all_files)
    return all_files




def predict(all_files, config, model):
    
    outputMask = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        
        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedMask = detect_tree(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputMask)
    
    return outputMask, pred_labels

def predict_chm(all_files, config, model):
    
    outputMask = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        
        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedMask = detect_tree_chm(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputMask)
    
    return outputMask, pred_labels


def predict_save(all_files, config, model):
    
    outputMask = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        outputFile = os.path.join(config.output_dir, filename.replace(config.channel_names[0], config.output_prefix))
        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedMask, detectedMeta = detect_tree_save(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type,  write_as_type = config.output_dtype)
            
            ##### saving thresholded masks
            dlabel = np.squeeze(gen_label(detectedMask))
            # print(type(dlabel), dlabel.shape, dlabel[0][1])
            mask2 = mask_thresd(dlabel, thres = 2)
            outputFile2 = os.path.join(config.output_dir, filename.replace(config.channel_names[0], 'det_thres2_'))

            writeMaskToDisk(mask2, detectedMeta, outputFile2, image_type = config.output_image_type,  write_as_type = config.output_dtype)
            
            
            outputMask.append(detectedMask)
            # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputMask)
    
    return outputMask, pred_labels

def predict_segcount_save(all_files, config, model, thr, multithre = 0):
    
    outputSeg = []
    outputDens = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        outputFile = os.path.join(config.output_dir, filename.replace(config.channel_names[0], config.output_prefix).replace('png', 'tif'))

        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedSeg, detectedDens, detectedMeta = detect_tree_segcount_save(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            writeMaskToDisk(detectedSeg, detectedMeta, outputFile, image_type = config.output_image_type,  write_as_type = 'float32', convert = 1)
            
            if multithre: # with label smoothing
                names = ['_thre2.tif', '_thre3.tif', '_thre4.tif', '_thre5.tif', '_thre6.tif', '_thre7.tif', '_thre8.tif', '_thre9.tif', '_thre95.tif']
                thres = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                for nn in range(len(thres)):
                    outputFilenn = outputFile.replace('.tif', names[nn])
                    writeMaskToDisk(detectedSeg, detectedMeta, outputFilenn, image_type = config.output_image_type,  write_as_type = 'float32', th=thres[nn], convert = 1)
                
            
            writeMaskToDisk(detectedDens, detectedMeta, outputFile.replace('det', 'pdens'), image_type = config.output_image_type,  write_as_type = config.output_dtype, convert = 0)

            outputSeg.append(detectedSeg)
            outputDens.append(detectedDens)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputSeg, thr)
    
    # get predicton total count
    pred_counts = integrate_count(outputDens, model)
    
    return outputSeg, pred_labels, outputDens, pred_counts
    
def predict_segcount_save_pad_reflect(all_files, config, model, thr, multithre = 0):
    # pad borders to keep all individuals in complete shape
    outputSeg = []
    outputDens = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        outputFile = os.path.join(config.output_dir, filename.replace(config.channel_names[0], config.output_prefix).replace('png', 'tif'))

        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedSeg, detectedDens, detectedMeta = detect_tree_segcount_save_pad_reflect(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            # detectedSeg = detectedSeg[..., np.newaxis]
            # detectedDens = detectedDens[..., np.newaxis]
            print('seg shape', detectedSeg.shape)
            print('den shape', detectedDens.shape)
            detectedSeg = detectedSeg[:, 7:detectedSeg.shape[1]-7, 7:detectedSeg.shape[2]-7]
            print('clipping resultts!!!!!!!!!!!!!!!')
            writeMaskToDisk(detectedSeg, detectedMeta, outputFile, image_type = config.output_image_type,  write_as_type = 'float32', convert = 1)
            
            if multithre: # with label smoothing
                names = ['_thre2.tif', '_thre3.tif', '_thre4.tif', '_thre5.tif', '_thre6.tif', '_thre7.tif', '_thre8.tif', '_thre9.tif', '_thre95.tif']
                thres = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                for nn in range(len(thres)):
                    outputFilenn = outputFile.replace('.tif', names[nn])
                    writeMaskToDisk(detectedSeg, detectedMeta, outputFilenn, image_type = config.output_image_type,  write_as_type = 'float32', th=thres[nn], convert = 1)
                
            # outputFile2 = outputFile.replace('.tif', '_thre2.tif')
            # writeMaskToDisk(detectedSeg, detectedMeta, outputFile2, image_type = config.output_image_type,  write_as_type = 'float32', th=0.2, convert = 1)
            # outputFile3 = outputFile.replace('.tif', '_thre8.tif')
            # writeMaskToDisk(detectedSeg, detectedMeta, outputFile3, image_type = config.output_image_type,  write_as_type = 'float32', th=0.8, convert = 1)
            # print(mask.shape)
            detectedDens = detectedDens[:, 7:detectedDens.shape[1]-7, 7:detectedDens.shape[2]-7]
            print('clipping resultts!!!!!!!!!!!!!!!')
            # print(mask.shape)
            writeMaskToDisk(detectedDens, detectedMeta, outputFile.replace('det', 'pdens'), image_type = config.output_image_type,  write_as_type = config.output_dtype, convert = 0)

            outputSeg.append(detectedSeg)
            outputDens.append(detectedDens)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputSeg, thr)
    
    # get predicton total count
    pred_counts = integrate_count(outputDens, model)
    
    return outputSeg, pred_labels, outputDens, pred_counts


def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, write_as_type = 'float32', th=0.5, convert = 1):
    # Convert to correct required before writing
    # if 'float' in str(detected_meta['dtype']):
    meta = detected_meta.copy()
    mask = detected_mask.copy()
    
    
    if convert:
        print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
        mask[mask<th]=0
        mask[mask>=th]=1
        # detected_mask = detected_mask.astype(write_as_type)
        # detected_meta['dtype'] =  write_as_type
        # detected_meta['count'] = 1
        
        # # dilate with rate 1.27 for wei 5 0512 model
        # dilation = cv2.dilate(img,kernel,iterations = 1)
        
    
    mask = mask.astype(write_as_type)
    # print(detected_mask.shape)
    if mask.ndim != 2:
        mask = mask[0]
    # print(detected_mask.shape)
    meta['dtype'] =  write_as_type
    meta['count'] = 1
    meta.update(
                        {'compress':'lzw',
                          'driver': 'GTiff',
                            'nodata': 255
                        }
                    )
        
    print(mask.shape)
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(mask, 1)
    
    return
        

def integrate_count(maskdensity, model):
    counts = []
    for i in range(len(maskdensity)):
        maskdensity[i][maskdensity[i]<0]=0
        c = maskdensity[i].sum(axis = (1, 2))
        counts.append(c)
        
    return counts

def predict_segcount(all_files, config, model, thr):
    
    outputSeg = []
    outputDens = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        
        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedSeg, detectedDens, ssss = detect_tree_segcount(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            outputSeg.append(detectedSeg)
            outputDens.append(detectedDens)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputSeg, thr)
    
    # get predicton total count
    # print(outputDens[0].shape)
    pred_counts = integrate_count(outputDens, model)
    
    return outputSeg, pred_labels, outputDens, pred_counts, ssss

def predict_segcountchm(all_files, config, model):
    
    outputSeg = []
    outputDens = []
    outputChms = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        
        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                    # print('stop1')
                
            detectedSeg, detectedDens, detectedChms = detect_tree_segcountchm(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            outputSeg.append(detectedSeg)
            outputDens.append(detectedDens)
            outputChms.append(detectedChms)
                # outputMeta.append(detectedMeta)
                
        # else: # single raster or multi raster without aux
        #     print('Single raster or multi raster without aux')
        #     with rasterio.open(fullPath) as img:
        #         #print(fullPath)
        #         detectedMask = detect_tree(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
        #         outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputSeg)
    
    # get predicton total count
    pred_counts = integrate_count(outputDens)
    
    return outputSeg, pred_labels, outputDens, pred_counts, outputChms


def predict_single_input(all_files, config, model):
    
    outputMask = []
    # outputMeta = []
    
    for fullPath, filename in all_files:
        
        if not config.single_raster and config.aux_data: # multi raster
            with rasterio.open(fullPath) as im0:
                chs = []
                for i in range(1, len(config.channel_names)):
                    chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                # print('stop1')
                detectedMask = detect_tree_single_input(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
        else: # single raster or multi raster without aux
            print('Single raster or multi raster without aux')
            with rasterio.open(fullPath) as img:
                #print(fullPath)
                detectedMask = detect_tree_single_input(config, model, [img], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                outputMask.append(detectedMask)
                # outputMeta.append(detectedMeta)
                
    # generate binary label
    pred_labels = gen_label(outputMask)
    
    return outputMask, pred_labels


def load_truths(all_files, config):
    # load ground truths
    gt = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]
        # TODO: add different weights! 
        # do not use boundary at all
        cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]))
        
        cur = np.transpose(cur, axes=(1,2,0))
        gt.append(cur)
        
    return gt


def load_truths_segcount(all_files, config):
    # load ground truths: seg, boundary, ann_kernel
    gtseg = []
    gtdens = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]
        # TODO: add different weights! 
        cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]))
        cur = np.transpose(cur, axes=(1,2,0))
        gtseg.append(cur)
        
        curdens = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[2])).read()[0, :, :]
        
        gtdens.append(curdens)
        
        
        
    return gtseg, gtdens

def load_truths_segcountchm(all_files, config):
    # load ground truths: seg, boundary, ann_kernel
    gtseg = []
    gtdens = []
    gtchm = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]
        # TODO: add different weights! 
        cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[1])).read()[0, :, :]))
        cur = np.transpose(cur, axes=(1,2,0))
        gtseg.append(cur)
        
        curdens = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[2])).read()[0, :, :]
        
        gtdens.append(curdens)
        
        curchm = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[3])).read()[0, :, :]
        
        gtchm.append(curchm)
        
        
    return gtseg, gtdens, gtchm


def load_truths_regcounting(all_files, config):
    # load ground truths
    gt = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]
        # TODO: add different weights! 
        # cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[1])).read()[0, :, :]))
        # print(cur.shape)
        # cur = np.transpose(cur, axes=(1,2,0))
        gt.append(cur)
        
    return gt


def load_truths_chmdensity(all_files, config):
    # load ground truths
    gt = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[-1])).read()[0, :, :]
        # TODO: add different weights! 
        # cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[1])).read()[0, :, :]))
        # print(cur.shape)
        # cur = np.transpose(cur, axes=(1,2,0))
        gt.append(cur)
        
    return gt

def metrics(pred_labels, all_files, gt, model_id = 0, plot = 1, save_scores = 0, savefig = 0, savename = 0):
    ''''Compute metrics for the testing images, one by one
    
        Compute average scores
    '''
    
    acc_list = []
    loss_list = []
    dice_list = []
    sen_list = []
    spe_list = []
    iou_list = []
    
    for i in range(len(pred_labels)):
        logit = pred_labels[i]
        tr = gt[i]
        ann = tr[:,:,0]
        tver = tversky(tr, logit[..., np.newaxis]).numpy()
        
        loss_list.append(tver)
        lb = pred_labels[i].astype(np.int16)
        
        acc = eva_acc(ann, lb)
        acc_list.append(acc)
        dic = eva_dice(ann, logit)
        dice_list.append(dic)
        sen = eva_sensitivity(ann, logit)
        sen_list.append(sen)
        spe = eva_specificity(ann, logit)
        spe_list.append(spe)
        iou = eva_miou(ann, logit)
        iou_list.append(iou)
        
        if plot:
            # plt.figure()
            # plt.imshow(lb)
            # plt.title(all_files[i][1])
            # plt.figure()
            # plt.imshow(tr[:, :, 0])
            # plt.title(all_files[i][1])
            im = np.squeeze(rasterio.open(all_files[i][0]).read())
            
            
            
            if model_id:
                if savefig and savename:
                
                    display_images(np.stack((im, ann, lb), axis = -1)[np.newaxis, ...], titles = ['red' + model_id, 'ann' + model_id, 'pred' + model_id], savefig = savefig, savename = savename + '_' + str(i))
                else:
                    display_images(np.stack((im, ann, lb), axis = -1)[np.newaxis, ...], titles = ['red' + model_id, 'ann' + model_id, 'pred' + model_id], savefig = savefig, savename = savename)

            else: 
                display_images(np.stack((im, ann, lb), axis = -1)[np.newaxis, ...], titles = ['red', 'ann', 'pred'])
                
    avg_acc = avg_metrics(acc_list)
    avg_loss = avg_metrics(loss_list)
    avg_dice = avg_metrics(dice_list)
    avg_iou = avg_metrics(iou_list)
    avg_sen = avg_metrics(sen_list)
    avg_spe = avg_metrics(spe_list)
    
    print("Acc:{}, Dice:{}, mIoU:{}, loss:{}".format(avg_acc, avg_dice, avg_iou, avg_loss))

    if save_scores:
        return acc_list, loss_list, dice_list, sen_list, spe_list, iou_list
    
    return avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe
    

    
def avg_metrics(lst):
    return np.array(lst).sum()/len(lst)

def remove_small_noarea(preds, gt, thres = 10, plot = 1):
    
    clearPreds = []
    counts_all = []
    counts_nosmall = []
    count = 0
    counts_gt = []
    for p in preds:
        pred = p.astype(np.uint8)
        predc = pred.copy()
        contours, hierarchy = cv2.findContours(predc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        count1 = 0
        count2 = 0
        for c in contours:
            count1 += 1
            count2 += 1
            area = cv2.contourArea(c)
            # print(area)
            if area < thres:
                # remove small
                cv2.drawContours(predc, [c], -1, (0,0,0), -1)
                count2 -= 1
        
        clearPreds.append(predc.astype(np.float32))
        counts_all.append(count1)
        counts_nosmall.append(count2)
        
        # gt
        curlb = gt[count][:, :, 0]
        curbd = gt[count][:, :, 1]
        
        # ground truth count
        curlb_int = curlb.astype(np.uint8)
        gt_contours, gt_hierarchy = cv2.findContours(curlb_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        counts_gt.append(len(gt_contours))
        
        if plot:
            plt.figure(figsize = (10, 10))
            plt.imshow(pred)
            plt.title('Before removing samll')
            plt.figure(figsize = (10, 10))
            plt.imshow(predc)
            plt.title('After removing samll')
            plt.figure(figsize = (10, 10))
            plt.imshow(curlb)
            plt.title('label')
            plt.figure(figsize = (10, 10))
            plt.imshow(curbd)
            plt.title('boundary')
            
        
        count += 1
    return clearPreds, counts_all, counts_nosmall, counts_gt


def remove_small(preds, gt, thres = 10, plot = 1):
    
    clearPreds = []
    counts_all = []
    counts_nosmall = []
    count = 0
    counts_gt = []
    totalareaGT = 0
    totalareaP = 0
    for p in preds:
        pred = p.astype(np.uint8)
        predc = pred.copy()
        contours, hierarchy = cv2.findContours(predc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        count1 = 0
        count2 = 0
        for c in contours:
            count1 += 1
            count2 += 1
            area = cv2.contourArea(c)
            # print(area)
            if area < thres:
                # remove small
                cv2.drawContours(predc, [c], -1, (0,0,0), -1)
                count2 -= 1
        
        clearPreds.append(predc.astype(np.float32))
        counts_all.append(count1)
        counts_nosmall.append(count2)
        
        totalareaP += predc.astype(np.float32).sum()
        
        # gt
        curlb = gt[count][:, :, 0]
        curbd = gt[count][:, :, 1]
        
        totalareaGT += curlb.sum()
        
        # ground truth count
        curlb_int = curlb.astype(np.uint8)
        gt_contours, gt_hierarchy = cv2.findContours(curlb_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        counts_gt.append(len(gt_contours))
        
        if plot:
            plt.figure(figsize = (10, 10))
            plt.imshow(pred)
            plt.title('Before removing samll')
            plt.figure(figsize = (10, 10))
            plt.imshow(predc)
            plt.title('After removing samll')
            plt.figure(figsize = (10, 10))
            plt.imshow(curlb)
            plt.title('label')
            plt.figure(figsize = (10, 10))
            plt.imshow(curbd)
            plt.title('boundary')
            
        
        count += 1
    return clearPreds, counts_all, counts_nosmall, counts_gt, totalareaGT, totalareaP

def tree_sensitivity(clearpreds, gts):
    """tree level sensitivit: use gt ann mask to compute % of crown predicted"""
    crown_areas = []
    
    for i in range(len(gts)):
        curlb = gts[i][:, :, 0].astype(np.uint8)
        pred = clearpreds[i].astype(np.uint8)
        # lbc = curlb.copy()
        contours, hierarchy = cv2.findContours(curlb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # get rid of whole image detected as contour
        # largest_areas = sorted(contours, key=cv2.contourArea)
        # pmax = curlb.shape[0]*curlb.shape[1]-100
        # for c in largest_areas:
        #    cur = np.zeros(curlb.shape)
        #    area_gt = cv2.contourArea(c)
        #    if area_gt > 8000:
        #        print('very large area', area_gt)
        #        cv2.drawContours(cur, [c], -1, (255,255,255), -1)
        #        plt.figure()
        #        plt.subplot(121)
        #        plt.imshow(cur)
        #        plt.subplot(122)
        #        plt.imshow(curlb)
        
        for c in contours:
            cur = np.zeros(curlb.shape)
            area_gt = cv2.contourArea(c)
            cv2.drawContours(cur, [c], -1, (255,255,255), -1)
            pred_con = pred.copy()
            cur[cur == 255] = 1
            pred_con = pred_con*cur
            area_pred = pred_con.sum()
            
            crown_areas.append((area_gt, area_pred))
            # # plot figures
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(cur)
            # plt.title(str(area_gt))
            # plt.subplot(122)
            # plt.imshow(pred_con)
            # plt.title(str(area_pred))
            
    return crown_areas
    

def self_thinning_curve(pred_denss, pred_segs):
    """self thinning: log density vs log avg crown area per plot"""
    crown_area_avgs = []
    tree_density = []
    for i in range(len(pred_segs)):
        # curlb = gts[i][:, :, 0].astype(np.uint8)
        pred_seg = pred_segs[i].astype(np.uint8)
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

def mask_thresd(pred, thres = 2):
    
    pred = pred.astype(np.uint8)
    predc = pred.copy()
    contours, hierarchy = cv2.findContours(predc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
        
        area = cv2.contourArea(c)
        # print(area)
        if area < thres:
            # remove small
            cv2.drawContours(predc, [c], -1, (0,0,0), -1)
            
    clearPreds = predc.astype(np.float32)
    
       
    
    return clearPreds
     


def score_without_small(all_files, preds, gt, thres = 10, plot = 1):
    clear, c_all, c_nosmall, c_gt, areaGT, areaP = remove_small(preds, gt, thres = thres, plot = plot)
    print('ground truth counts:', c_gt)
    print('total count:', np.array(c_gt).sum())
    print('-------------------------------------------')
    print('Before removing small objects:')
    print('Post processing --- tree count:', c_all)
    print('Post processing --- total count:', np.array(c_all).sum())
    # print('r2 of counts', r2_score(np.array(c_gt), np.array(c_all)))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(c_gt), np.array(c_all))
    print('r2 of counts', r_value**2)
    # avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(preds, all_files, plot = plot)
    print('--------')
    print('After removing small objects:')
    print('Post processing --- tree count:', c_nosmall)
    print('Post processing --- total count:', np.array(c_nosmall).sum())
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(c_gt), np.array(c_nosmall))
    print('r2 of counts', r_value**2)
    # print('r2 of counts', r2_score(np.array(c_gt), np.array(c_nosmall)))
    print('---------')
    print('Post processing --- total canopy area', areaP)
    print('Post processing --- total canopy area ground truth', areaGT)
    print('Post processing --- total canopy area / ground truth', areaP/areaGT)
    avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(clear, all_files, gt, model_id = 0, plot = 0)
    print('Metrics after removing small objects')
    decrease = (np.array(c_all).sum() - np.array(c_nosmall).sum()) / np.array(c_all).sum()
    print('Tree count decrease after removing small objects:', decrease)
    return clear, c_gt

def score_without_small_scorelist(all_files, preds, gt, thres = 10, plot = 1):
    clear, c_all, c_nosmall, c_gt = remove_small_noarea(preds, gt, thres = thres, plot = plot)
    print('ground truth counts:', c_gt)
    print('total count:', np.array(c_gt).sum())
    print('-------------------------------------------')
    print('Before removing small objects:')
    print('Post processing --- tree count:', c_all)
    print('Post processing --- total count:', np.array(c_all).sum())
    # avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(preds, all_files, plot = plot)
    print('--------')
    print('After removing small objects:')
    print('Post processing --- tree count:', c_nosmall)
    print('Post processing --- total count:', np.array(c_nosmall).sum())
    avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(clear, all_files, gt, model_id = 0, plot = 0, save_scores = 1)
    print('Metrics after removing small objects')
    decrease = (np.array(c_all).sum() - np.array(c_nosmall).sum()) / np.array(c_all).sum()
    print('Tree count decrease after removing small objects:', decrease)
    return avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe

def report(models, pred_labels, config, all_files, gt, thres = 4, plot = 0, modeln = 0, savefig = 0, savename = 0):
    for mi in range(len(models)):
        curpred = []
        for i in range(len(pred_labels)):
            curpred.append(pred_labels[i][mi])
        print('---------------------------------------------------------------------------------------------------')
        print('Metrics for model:', config.trained_model_paths[mi])
        if modeln:
            if savename:
                avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln + str(mi), plot =plot, savefig = savefig, savename = savename + str(mi))
            else:
                avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln + str(mi), plot =plot, savefig = savefig, savename = savename)

        else:
            avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln, plot =plot, savefig = savefig, savename = savename)

        # remove small objects
        print('Removing small object with a threshold of:', thres)
        clearPreds, c_all, c_nosmall, c_gt, areaGT, areaP = remove_small(curpred, gt, thres = thres, plot = 0)
        
        clear_p, c_gt = score_without_small(all_files, curpred, gt, thres = thres, plot = 0)
    return c_all, c_nosmall, c_gt, clear_p

def report_scorelist(models, pred_labels, config, all_files, gt, thres = 4, plot = 0, modeln = 0, savefig = 0, savename = 0):
    for mi in range(len(models)):
        curpred = []
        for i in range(len(pred_labels)):
            curpred.append(pred_labels[i][mi])
        print('---------------------------------------------------------------------------------------------------')
        print('Metrics for model:', config.trained_model_paths[mi])
        if modeln:
            if savename:
                avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln + str(mi), plot =plot, savefig = savefig, savename = savename + str(mi))
            else:
                avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln + str(mi), plot =plot, savefig = savefig, savename = savename)

        else:
            avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(curpred, all_files, gt, model_id = modeln, plot =plot, savefig = savefig, savename = savename)

        # remove small objects
        print('Removing small object with a threshold of:', thres)
        clearPreds, c_all, c_nosmall, c_gt = remove_small_noarea(curpred, gt, thres = thres, plot = 0)
        
        avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = score_without_small_scorelist(all_files, curpred, gt, thres = thres, plot = 0)
    return avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe




