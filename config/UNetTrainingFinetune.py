#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:10:30 2021

@author: sizhuo
"""

class Configuration:
    def __init__(self):

        # small local dataset for fine-tuning, here data from Finland as used in the paper, downloadable with the google drive link
        self.base_dir = '/home/sizhuo/Downloads/extracted_centroids_kernel5/' # 'path_to_local_data'
        self.image_type = '.png'
        self.oversample_times = 4 # oversample the (small) local data to match with the (larger) pretraining dataset if needed. Eg, the pretraining denmark dataset contains around 80 image patches, while the local dataset (here finland) contains only 18 patches. To balance the amount of data seen by the model during finetuning, the local data is oversampled 4 times.
        
        ##### 
        # danish data used for pretraining, downloadable with the google drive link
        self.base_dir2 = '/home/sizhuo/Downloads/extracted_data_train_patch_normalized/' # 'path_to_base_data'

        self.extracted_filenames = ['infrared', 'green', 'blue'] # color bands in the same order as used for pretraining (self.model_path)
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.density_fn = 'ann_kernel'
        self.boundary_weights = 3  # 3 or 5 is recommended
        self.single_raster = False
        self.aux_data = self.multires = False
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        # (Input + Output) channels
        self.image_channels = len(self.extracted_filenames)
        self.all_channels = self.image_channels + 3
        self.patch_size = (256,256,self.all_channels) # Height * Width * (Input + Output) channels
        self.test_ratio = 0
        self.val_ratio = 0.1
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 1

        self.upsample = 1 # whether to upsample the local input images to match with the denmark data at 20cm, set to 1 if the local data is of lower resolution, eg. 50 cm
        self.upscale_factor = 2 # if self.upsample=1, the upscale factor. e.g., to roughly align 50cm local to 20cm pretraining data, this was set to 2 for the paper. A multiple of 2 is recommended for better numerical stability.

        # Shape of the input data, height*width*channel
        self.input_shape = (256,256,self.image_channels)
        # self.label_weight_channel = [self.image_channels, self.image_channels+1]
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]
        self.inputBN = False
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.pretrain_NBepochs = 1400 # number of epochs for pretraining
        self.NB_EPOCHS = 600 # number of epochs for fine-tuning

        # number of validation images to use
        self.VALID_IMG_COUNT = 100 #200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500 #1000
        
        
        # saving 
        self.pretrained_name = 'complex5'
        # model (pre-)trained with the danish data, downloadable with the google drive link
        self.model_path = '/home/sizhuo/Downloads/saved_models/trees_20210620-0205_Adam_e4_infraredgreenblue_256_84_frames_weightmapTversky_MSE100_5weight_attUNet.h5'
        self.new_model_path = '/home/sizhuo/Downloads/saved_models/finetune/'
        self.log_dir = '/home/sizhuo/Downloads/logs/'


