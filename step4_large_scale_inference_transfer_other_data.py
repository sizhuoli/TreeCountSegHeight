#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:43:25 2021

@author: sizhuo
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import numpy as np
from core2.raster_ana_segcount import anaer, CHMerror
from config import RasterAnalysis_multires
import logging


config = RasterAnalysis_multires.Configuration()


predictor = anaer(config, DK = 0)

predictor.segcount_CN(inf = 1, th = 0.5) # set inf = 1 to keep all bands if rgb only





