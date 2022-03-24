#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:34:10 2021

@author: sizhuo
"""
# for DK and FI


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import numpy as np
# from core2.raster_analysis import segcount_analyze
from core2.raster_ana_segcount import anaer, CHMerror
from config import RasterAnalysis_multires
import logging


config = RasterAnalysis_multires.Configuration()


# for FI
# predictor = segcount_analyze(config)
# # rescale input 
# predictor.preprocess()
# pred
# for FI
# predictor.prediction_segcount()
# predictor.chmF()


# for DK
#### polygonization need to run in tfgdal env!!!!!!!!!!!!!!!!!
predictor = anaer(config)
predictor.pred_segcount()
predictor.pred_chm()
predictor.pred_3tasks()
predictor.pred_sampling_eva_DK(num = 10, sample_fixed = 0)
predictor.polyonize_sampleDK()
len(predictor.h_prs)


# DK large scale
predictor = anaer(config, largescale = 1)
predictor.pred_3tasks_detchm()



# f1 = [i for i in range(len(predictor.heights_gt)) if predictor.heights_gt[i] >=2]
# pr_f1 = predictor.heights_pr[f1]
# gt_f1 = predictor.heights_gt[f1]
CHMerror(predictor.heights_pr, predictor.heights_gt, nbins = 100)


# fi large tifs
predictor = anaer(config, DK = 0, pred = 1)
# predictor.pred_segcount()
# predictor.pred_chm()
predictor.pred_3tasks_fi(DK = 0)
predictor.polyonize_FI()

# crop into smaller patches firest
predictor = anaer(config, DK = 0, pred = 1, patches = 1)
predictor.preprocess_cropping_FI()
predictor.pred_3tasks_fi_patches()
predictor.polyonize_FI()


# copy DK sampled gt chms
import glob
import shutil
import os
base = '/mnt/ssdc/Denmark/summer_predictions/segcountchm/DK_FTY_TreeHeight_final6_1819_CHM1111_Wmae_filter_chmerror_expand02/forestType_2/'
gtdir = '/mnt/ssda/denmark_CHM/'
gt_dst = base + 'gtCHMs/'
if not os.path.exists(gt_dst):
    os.makedirs(gt_dst)
fps = glob.glob(f"{base}/det_chm_1km*.tif")
len(fps)
for f in fps:
    coor = f[-12:-4]
    chmgt = glob.glob(f"{gtdir}/**/CHM_1km_{coor}.tif")[0]
    # print(f)
    # print(chmgt)
    # print(gt_dst+os.path.basename(chmgt))
    shutil.copy2(chmgt, gt_dst+os.path.basename(chmgt))