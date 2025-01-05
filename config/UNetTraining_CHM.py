#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration of the parameters for the 2-UNetTraining.ipynb notebook

Created on Wed Mar  3 00:37:50 2021
@author: rscph
"""

import os


class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the channels and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        
        # dir where extrected images are
        self.base_dir = '/mnt/ssdc/Denmark/color2CHM/training_data201819/extracted/training/train/'
        self.base_dir_val = '/mnt/ssdc/Denmark/color2CHM/training_data201819/extracted/training/val/'
        self.image_type = '.png'
#         self.ndvi_fn = 'ndvi'
#         self.pan_fn = 'pan'
        self.extracted_filenames = ['red', 'green', 'blue']
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.chm_fn = 'chm'
        self.boundary_weights = 5
        self.single_raster = False
        self.aux_data = False # here ndvi are not considered aux
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        # (Input + Output) channels
        self.image_channels = len(self.extracted_filenames)
        self.all_channels = self.image_channels + 1
        self.patch_size = (256,256,self.image_channels) # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = -1
        # maxmin normalize chm
        self.maxmin_norm = 0
        # gb norm DK stat
        self.gb_norm = 1
        self.gb_norm_FI = 0 # FI gb norm
        # robust scaling DK stat
        self.robust_scale = 0
        
        # upscale for aux chm layer
        self.upscale_factor = 2
        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        self.patch_dir = self.base_dir + 'patches{}'.format(self.patch_size[0])
        self.frames_json = os.path.join(self.patch_dir,'frames_list.json')

        # Shape of the input data, height*width*channel
        self.input_shape = (256,256,self.image_channels)
        # self.input_shape = (512,512,self.image_channels)
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        # self.label_weight_channel = [self.image_channels, self.image_channels+1]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 1000

        # number of validation images to use
        self.VALID_IMG_COUNT = 100 #200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500 #1000
        self.model_name = 'unet_attention'
        self.model_path = './saved_models/color2CHM/UNet/'
