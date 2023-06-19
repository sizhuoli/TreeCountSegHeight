#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:41:28 2023

@author: sizhuo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import colors
from scipy.stats import linregress
import re
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import Model_compare_multires
from core2.model_compare import eva_segcount
import tensorflow as tf

config = Model_compare_multires.Configuration()

# segcount
eva = eva_segcount(config)
eva.pred(thr = 0.5)
c_all, c_nosmall, c_gt, c_gt_ha, c_nosmall_ha, clear_p, gts = eva.report_seg(thres = 2)     
# tree density (/ha)
gt_dens_counts, pred_dens_counts, image_area_list = eva.report_count_density()


print('mae', mean_absolute_error(gt_dens_counts, pred_dens_counts))
print('rmae', mean_absolute_error(gt_dens_counts, pred_dens_counts)/np.array(gt_dens_counts).mean())
# relaive total error
ate = np.abs((np.array(gt_dens_counts)-np.array(pred_dens_counts)).sum())
rte = ate/np.array(gt_dens_counts).sum()
print('relatiev total error %', rte*100)
