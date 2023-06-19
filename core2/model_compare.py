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

from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape

import numpy as np              
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
from skimage.transform import resize

from itertools import product
from tensorflow.keras.models import load_model
import cv2
import tensorflow.keras.backend as K
import copy 


from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.eva_losses import eva_acc, eva_dice, eva_sensitivity, eva_specificity, eva_miou 
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.frame_info_multires_segcount import FrameInfo, image_normalize
from core2.visualize import display_images


import matplotlib.pyplot as plt  
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  
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
            # modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
            # deal with keras version mismatch
            modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity, 'K': K}, compile=False)
            
            modeli.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_count':'mse'},
                            metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                                'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
            self.models.append(modeli)
            modeli.summary()
            
        self.all_files = load_files(self.config)
        
    
    def pred(self, thr = 0.5, save = 0):
        
        self.outputSeg, self.pred_labels, self.outputDens, self.pred_counts = predict_segcount_save(self.all_files, self.config, self.models, thr)

    def report_seg(self, thres = 2, plot = 0, savefig = 0):
        self.gtseg, self.gtdens = load_truths_segcount(self.all_files, self.config)
        if plot:
            report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = thres, plot = 1, modeln = 'Model')
        
        else:
            c_all, c_nosmall, c_gt, self.clear_ps, self.gts = report(self.models, self.pred_labels, self.config, self.all_files, self.gtseg, thres = thres, plot = 0)
            c_gt_ha = []
            c_nosmall_ha = []
            for i in range(len(self.clear_ps)):
                # compute tree density tree (trees/ha)
                tot_area = self.clear_ps[i].size * 0.04 # in m**2
                c_gt_d = (c_gt[i] / tot_area) * 10000 # no trees/ha
                c_nosmall_d = (c_nosmall[i] / tot_area) * 10000 # no trees/ha
                c_gt_ha.append(c_gt_d)
                c_nosmall_ha.append(c_nosmall_d)
                
            
            return c_all, c_nosmall, c_gt, c_gt_ha, c_nosmall_ha, self.clear_ps, self.gts
    
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
                ttlist.append(self.gtdens[i].sum())
                predlist.append(self.pred_counts[i][mm])
                ttc+=self.pred_counts[i][mm]
                cgt += self.gtdens[i].sum()
            print('pred count', ttc)
            print('reference count', cgt)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(ttlist), np.array(predlist))
            
        return ttlist, predlist
    
    def report_count_density(self):
        for mm in range(len(self.config.trained_model_paths)):
            ttc = 0
            cgt = 0
            szlist = []
            ttlist = []
            predlist = []
            for i in range(len(self.pred_counts)):
                print('------------------------------------------')
                # print('gt', gtdens[i].sum())
                curcount_gt = self.gtdens[i].sum()
                print('reference count', curcount_gt)
                print(self.gtdens[i].size)
                tot_area = self.gtdens[i].size * 0.04 # in m**2
                print(tot_area)
                szlist.append(tot_area)
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
            print('reference count', cgt)
            
            
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(ttlist), np.array(predlist))
            
        return ttlist, predlist, szlist
        
        
    
    
    

def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX', dens = 0):
    
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
    
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions) 
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    elif operator == "MIX": # alpha blending # note do not combine with empty regions
        mm1 = currValue!=0
        currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
        mm2 = (currValue==0)
        currValue[mm2] = newPredictions[mm2]
        resultant = currValue
    else: 
        resultant = newPredictions
    
    res[row:row+he, col:col+wi] =  resultant
    return (res)



# 2 tasks
def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator):
    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
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
            
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator, dens = 0)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator, dens = 1)
    return maskseg, maskdens


       
def detect_tree_segcount_save(config, models, img, width=256, height=256, stride = 128, normalize=True, singleRaster = 1, multires = 1):

    if 'chm' in config.channel_names:
        CHM = 1
    else:
        CHM = 0
        
    img0 = img[0] # channel 0
    
    read_img0 = img0.read()
    nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first        
    meta = img0.meta.copy()
    
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

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
            
                for ch in range(1, len(img)-1): # except for the last channel 
                    
                    imgi = rasterio.open(img[ch]) 
                    
                    sm1 = imgi.read(
                                    out_shape=(
                                    1,
                                    int(window.height),
                                    int(window.width)
                                ),
                                resampling=Resampling.bilinear, window = window)
                    
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
                
                    for ch in range(1, len(img)-1): # except for the last channel 
                        
                        imgi = rasterio.open(img[ch]) 
                        
                        sm1 = imgi.read(
                                        out_shape=(
                                        1,
                                        int(window.height),
                                        int(window.width)
                                    ),
                                    resampling=Resampling.bilinear, window = window)
                        
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
                        
                        temp_im1 = np.row_stack((temp_im1, sm1))
                    
        elif not CHM:
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
                
                temp_im1 = np.row_stack((temp_im1, sm1))
        
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        
        if normalize:
            if CHM:
                if multires:
                    print('multires chm, normalize chm layer')
                    temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    temp_im2 = image_normalize(temp_im2, axis=(0,1))
                elif not multires:
                    print('det chm, norm chm layer')
                    
                    # normalize all bands
                    temp_im1 = image_normalize(temp_im1, axis=(0,1))
                    
                    
                    
            elif not CHM:
                temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                    
                    
                    
                    
                    
        patch1[:window.height, :window.width] = temp_im1
        
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
            
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]
                
                curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
            batch = []
            batch_pos = []
            
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        
        for mi in range(len(models)):
            curmaskseg = masksegs[mi, :, :]
            curmaskdens = maskdenss[mi, :, :]
            
            curmaskseg, curmaskdens = predict_using_model_segcount(models[mi], batch, batch_pos, curmaskseg, curmaskdens, config.operator)
                
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta




def gen_label(masks, thr = 0.5):
    mm = copy.deepcopy(masks)
    print('threshold', thr)
    for m in mm:
        m[m<thr]=0
        m[m>=thr]=1
    
    return mm

def load_files(config):
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.channel_names[0]):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw image to predict:', len(all_files))
    print(all_files)
    return all_files





def predict_segcount_save(all_files, config, model, thr):
    
    outputSeg = []
    outputDens = []
    
    for fullPath, filename in all_files:
        outputFile = os.path.join(config.output_dir, filename.replace(config.channel_names[0], config.outputseg_prefix).replace(config.input_image_type, config.output_image_type))

        # if not config.single_raster and config.aux_data: # multi raster
        with rasterio.open(fullPath) as im0:
            chs = []
            for i in range(1, len(config.channel_names)):
                chs.append(fullPath.replace(config.channel_names[0], config.channel_names[i]))
                
            detectedSeg, detectedDens, detectedMeta = detect_tree_segcount_save(config, model, [im0, *chs], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, singleRaster=config.single_raster, multires= config.multires) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
            writeMaskToDisk(detectedSeg, detectedMeta, outputFile, image_type = config.output_image_type,  write_as_type = config.output_dtype, convert = 1)
            writeMaskToDisk(detectedDens, detectedMeta, outputFile.replace(config.outputseg_prefix, config.outputdens_prefix), image_type = config.output_image_type,  write_as_type = config.output_dtype, convert = 0)

            outputSeg.append(detectedSeg)
            outputDens.append(detectedDens)
                
        
                
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
        
    
    mask = mask.astype(write_as_type)
    if mask.ndim != 2:
        mask = mask[0]
    meta['dtype'] =  write_as_type
    meta['count'] = 1
    meta.update(
                        {'compress':'lzw',
                          'driver': 'GTiff',
                            'nodata': 255
                        }
                    )
        
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



def load_truths_segcount(all_files, config):
    # load ground truths: seg, boundary, ann_kernel
    gtseg = []
    gtdens = []
    for fullPath, filename in all_files:
        cur = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]
        cur = np.stack((cur, rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[0])).read()[0, :, :]))
        cur = np.transpose(cur, axes=(1,2,0))
        gtseg.append(cur)
        curdens = rasterio.open(fullPath.replace(config.channel_names[0], config.label_names[2])).read()[0, :, :]
        gtdens.append(curdens)
    return gtseg, gtdens





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

     


def score_without_small(all_files, preds, gt, thres = 10, plot = 1):
    clear, c_all, c_nosmall, c_gt, areaGT, areaP = remove_small(preds, gt, thres = thres, plot = plot)
    print('ground truth counts:', c_gt)
    print('total count:', np.array(c_gt).sum())
    print('-------------------------------------------')
    print('Before removing small objects:')
    print('Post processing --- tree count:', c_all)
    print('Post processing --- total count:', np.array(c_all).sum())
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(c_gt), np.array(c_all))
    print('--------')
    print('After removing small objects:')
    print('Post processing --- tree count:', c_nosmall)
    print('Post processing --- total count:', np.array(c_nosmall).sum())
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(c_gt), np.array(c_nosmall))
    print('---------')
    print('Post processing --- total canopy area', areaP)
    print('Post processing --- total canopy area ground truth', areaGT)
    print('Post processing --- total canopy area / ground truth', areaP/areaGT)
    avg_acc, avg_loss, avg_dice, avg_iou, avg_sen, avg_spe = metrics(clear, all_files, gt, model_id = 0, plot = 0)
    print('Metrics after removing small objects')
    decrease = (np.array(c_all).sum() - np.array(c_nosmall).sum()) / np.array(c_all).sum()
    print('Tree count decrease after removing small objects:', decrease)
    print('metrics after removing small noise: dice, iou, sens, spec', avg_dice, avg_iou, avg_sen, avg_spe)
    return clear, c_gt, gt


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
        
        clear_p, c_gt, gt = score_without_small(all_files, curpred, gt, thres = thres, plot = 0)
    return c_all, c_nosmall, c_gt, clear_p, gt




