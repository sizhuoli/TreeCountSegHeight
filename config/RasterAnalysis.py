"""a standard approach to predict tree count and segmentation for an example 1km tile with rgb bands and similar spatial resolution as the training data (20cm)"""
import os


class Configuration:   
    def __init__(self):
        self.input_image_dir = './example_1km_tile_tif/'
        self.input_image_type = '.tif' 
        self.input_image_pref = '2018' # prefix of image file names, can be used to filter out images
        self.channel_names1 = ['red', 'green', 'blue'] # if four color bands, set to ['red', 'green', 'blue', 'infrared']
        self.channels = [0, 1, 2] # to take color bands in the correct order (match with the model)
        self.rgb2gray = False # set to True if using only grayscale image (convert rgb band to grayscale)
        self.band_switch = False # set to True if using only subset of bands or change the order of bands
        self.addndvi = False
        # model downloadable with the google drive link
        self.trained_model_path = './models/trees_20210620-0202_Adam_e4_redgreenblue_256_84_frames_weightmapTversky_MSE100_5weight_attUNet.h5' 
        self.fillmiss = False # only fill in missing preds
        self.segcountpred = True # whether predict for segcount
        self.chmpred = False # whether predict for height
        self.normalize = True # patch norm
        self.segcount_tilenorm = False
        self.maxnorm = False
        self.gbnorm = True # DK gb norm from training data height model
        self.gbnorm_FI = False # FI gb norm from training data
        self.robustscale = False # using DK stats
        self.robustscaleFI_local = False
        self.robustscaleFI_gb = False
        self.localtifnorm = False
        self.multires = False
        self.downsave = False # same as upsample
        self.upsample = False # whether the output were upsampled or not
        self.upscale = 0
        self.rescale_values = False
        self.saveresult = 1
        self.tasks = 2
        self.change_input_size = False
        self.input_size = 256 # model input size
        self.input_shape = (self.input_size, self.input_size, len(self.channels))
        self.input_label_channel = [self.channels]
        self.inputBN = False
        self.output_dir = './predictions/'
        self.output_suffix = '_seg' # for segmentation
        self.chmdiff_prefix = 'diff_CHM_'
        self.output_image_type = '.tif'
        self.output_prefix = 'pred_'#+self.input_image_pref
        self.output_prefix_chm = 'pred_chm_'#+self.input_image_pref
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'
        self.output_dtype_chm='int16'
        self.single_raster = False
        self.aux_data = False
        self.operator = "MIX"  # for chm
        self.threshold = 0.00002 # for segmentation
        self.BATCH_SIZE = 256 # Depends upon GPU memory and WIDTH and HEIGHT
        self.WIDTH=256 # crop size
        self.HEIGHT=256
        self.STRIDE=32 #224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            