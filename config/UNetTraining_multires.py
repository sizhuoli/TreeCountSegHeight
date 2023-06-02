# multi tasks (segmentation +  count)
import os
from functools import reduce

import os

class Configuration:
    def __init__(self):

        self.ntasks = 2
        self.multires = 1
        self.base_dir = '/home/sizhuo/Desktop/code_repository/tree_crown_mapping_cleaned-main/example_extracted_data/' # dir to extrected images
        self.image_type = '.png'
        self.grayscale = 0
        self.extracted_filenames = ['red', 'green', 'blue', 'infrared', 'ndvi', 'chm'] # the list of all channels used
        self.channel_names = ['red', 'green', 'blue', 'infrared', 'ndvi'] # list of bands of the same spatial resolution
        self.channel_names2 = ['chm'] # input with different resolution
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.density_fn = 'ann_kernel'
        self.boundary_weights = 5 # weight for separation of individual tree crowns
        self.single_raster = False # whether one band or multiple bands
        self.aux_data = True # using supplementary data: height maps
        self.patch_generation_stratergy = 'random'
        self.image_channels = len(self.channel_names)
        self.image_channels2 = len(self.channel_names2)
        self.all_channels = self.image_channels + 1 + 2 # input bands + chm + label + weight
        self.patch_size = (256,256,self.all_channels) # Height * Width * (Input + Output) channels

        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0.4 # during training, seen as data augmentation

        # upscale for aux chm layer (lower resolution)
        self.upscale_factor = 2

        # Shape of the input data, height*width*channel
        self.input_shape = (256,256,self.image_channels) # input to the model
        self.input_image_channel = list(range(self.image_channels))
        self.input_image_channel2 = list(range(self.image_channels2))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 1500

        self.inputBN = 1 # if input image patches are not normalized, use batch norm for normalization
        self.task_ratio = [100, 1000, 10000] # list of 3 indicating the ratio for density branch at epochs 1, 100, 500

        # number of validation images to use
        self.VALID_IMG_COUNT = 100 #200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500 #1000

        # saving
        self.OPTIMIZER_NAME = 'Adam_e4'
        self.LOSS_NAME = 'WTversky'
        self.LOSS2 = 'Mse'#'Mse100'
        self.model_name = 'unet_attention'
        self.sufix = ''
        self.callbackImSave = './ImageCallbacks/'
        self.chs = reduce(lambda a,b: a+str(b), self.extracted_filenames, '')
        self.log_img = 0 # log images during training
        self.model_path = './saved_models/segcountdensity/'
