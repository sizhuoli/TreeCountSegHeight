#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:43:25 2021

@author: sizhuo
"""


import os

import ipdb
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import numpy as np
from core2.raster_ana_segcount import anaer
from config import RasterAnalysis
import logging


config = RasterAnalysis.Configuration()


predictor = anaer(config)
predictor.load_model()
predictor.segcount_RUN()





