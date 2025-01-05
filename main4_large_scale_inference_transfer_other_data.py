#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:43:25 2021

@author: sizhuo
"""

import tensorflow as tf
from core2.raster_ana_segcount import Anaer
from config import RasterAnalysis


config = RasterAnalysis.Configuration()

predictor = Anaer(config)
predictor.load_model()
predictor.segcount_RUN()
