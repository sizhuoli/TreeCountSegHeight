


# multi tasks (segmentation +  count)
import os
from functools import reduce

class Configuration:
    def __init__(self):
        self.ntasks = 2
        self.multires = 0 # see multi-resolution input in config/UNetTraining_multires
        self.base_dir = '/home/sizhuo/Desktop/code_repository/tree_crown_mapping_cleaned-main/example_extracted_data/' # extracted image and label patches
        self.image_type = '.png'
        self.grayscale = 0 # convert to grayscale (need to set channel_names = ['red', 'green', 'blue'] for conversion)
         # which channels to use: a list of channel names which correspond to the naming in the base_dir
        self.channel_names = ['red', 'green', 'blue']
        self.annotation_fn = 'annotation' # segmentation mask
        self.weight_fn = 'boundary' # boundary mask
        self.density_fn = 'ann_kernel' # gussian kernel mask
        self.boundary_weights = 3
        self.single_raster = False # set to 1 if only one channel
        self.aux_data = 0 # only no additional input of different resolution
        self.patch_generation_stratergy = 'random'
        if self.grayscale:
            self.image_channels = 1
        else:
            self.image_channels = len(self.channel_names)
        self.all_channels = self.image_channels + 3 # number of all label channls = 3
        self.patch_size = (256,256,self.all_channels) # Height * Width * (Input + Output) channels

        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0.4 # set to 0 if using input batch normalization

        # upscale for addon chm layer
        self.upscale_factor = 2
        self.input_shape = (256,256,self.image_channels)
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]

        self.inputBN = 1 # input batch norm: set to True if input data is not normalized
        self.task_ratio = [100, 1000, 10000] # list of 3 indicating the loss weighting ratio for density branch at epochs 1, 100, 500

        self.BATCH_SIZE = 16
        self.NB_EPOCHS = 1000 # number of epochs
        self.ifBN = True # use batch norm

        # number of validation images to use
        self.VALID_IMG_COUNT = 100
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500

        # saving
        self.OPTIMIZER_NAME = 'Adam_e4'
        self.LOSS_NAME = 'WTversky'
        self.LOSS2 = 'Mse'
        self.model_name = 'unet_attention'
        self.sufix = ''
        self.chs = reduce(lambda a,b: a+str(b), self.channel_names, '')
        self.model_path = './saved_models/segcountdensity/'
