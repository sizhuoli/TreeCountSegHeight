#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:10:30 2021

@author: sizhuo
"""

class Configuration:
    def __init__(self):

        self.base_dir = 'path_to_local_data'
        self.image_type = '.png'
        
        ##### 
        # add danish data
        self.base_dir2 = 'path_to_base_data'

        self.extracted_filenames = ['infrared', 'green', 'blue']
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.density_fn = 'ann_kernel'
        self.boundary_weights = 3
        self.single_raster = False
        self.aux_data = False
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        # (Input + Output) channels
        self.image_channels = len(self.extracted_filenames)
        self.all_channels = self.image_channels + 3
        self.patch_size = (256,256,self.all_channels) # Height * Width * (Input + Output) channels
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio.
        self.test_ratio = 0
        self.val_ratio = 0.1
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 1
        
        # upscale for aux chm layer
        self.upscale_factor = 2

        self.upsample = 1 # whether to upsample the input images
        # Shape of the input data, height*width*channel
        self.input_shape = (256,256,self.image_channels)
        # self.label_weight_channel = [self.image_channels, self.image_channels+1]
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 600

        # number of validation images to use
        self.VALID_IMG_COUNT = 100 #200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500 #1000
        
        
        # saving 
        self.pretrained_name = 'complex5'
        self.model_path = '/home/sizhuo/Downloads/saved_models/trees_20210620-0205_Adam_e4_infraredgreenblue_256_84_frames_weightmapTversky_MSE100_5weight_attUNet.h5'
        self.new_model_path = '/home/sizhuo/Downloads/saved_models/finetune/'
        self.log_dir = '/home/sizhuo/Downloads/logs/'


