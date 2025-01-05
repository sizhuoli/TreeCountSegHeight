import os
class Configuration:
    
    def __init__(self):
        # a standard approach to predict tree count and segmentation for an example 1km tile with rgb bands and similar spatial resolution as the training data (20cm)
        
        self.input_image_dir = '/home/sizhuo/Downloads/'#'/home/sizhuo/Desktop/code_repository/TreeCountSegHeight-main/example_1km_tile_tif/'
        self.input_image_type = '.tif' #'.tif'#'.jp2'
        self.input_image_pref = '202307' # prefix of image file names, can be used to filter out images
        self.channel_names1 = ['red', 'green', 'blue'] # if four color bands, set to ['red', 'green', 'blue', 'infrared']
        self.channels = [0, 1, 2] # to take color bands in the correct order (match with the model)
        self.rgb2gray = 0 # set to 1 if using only grayscale image (convert rgb band to grayscale)
        self.band_switch = 0 # set to 1 if using only subset of bands or change the order of bands
        self.addndvi = 0
        # model downloadable with the google drive link
        self.trained_model_path = '/home/sizhuo/Downloads/saved_models/trees_20210620-0202_Adam_e4_redgreenblue_256_84_frames_weightmapTversky_MSE100_5weight_attUNet.h5'
        self.fillmiss = 0 # only fill in missing preds
        self.segcountpred = 1 # whether predict for segcount
        self.chmpred = 0 # whether predict for height
        self.normalize = 1 # patch norm
        self.segcount_tilenorm = 0
        self.maxnorm = 0
        self.gbnorm = 1 # DK gb norm from training data height model
        self.gbnorm_FI = 0 # FI gb norm from training data
        self.robustscale = 0 # using DK stats
        self.robustscaleFI_local = 0
        self.robustscaleFI_gb = 0
        self.localtifnorm = 0
        self.multires = 0
        self.downsave = 0 # same as upsample
        self.upsample = 0 # whether the output were upsampled or not
        self.upscale = 0
        self.rescale_values = 0
        self.saveresult = 1
        self.tasks = 2
        self.change_input_size = 0
        self.input_size = 256 # model input size
        self.input_shape = (self.input_size, self.input_size, len(self.channels))
        self.input_label_channel = [self.channels]
        self.inputBN = False
        self.output_dir = '/home/sizhuo/Downloads/skysat_test/'#'/home/sizhuo/Desktop/code_repository/TreeCountSegHeight-main/example_1km_tile_tif/predictions/'
        self.output_suffix = '_seg' # for segmentation
        self.chmdiff_prefix = 'diff_CHM_'
        self.output_image_type = '.tif'
        self.output_prefix = 'pred_'#+self.input_image_pref
        self.output_prefix_chm = 'pred_chm_'#+self.input_image_pref
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'
        self.output_dtype_chm='int16'
        self.single_raster = 0
        self.aux_data = False
        self.operator = "MIX"  # for chm
        self.threshold = 0.00002 # for segmentation
        self.BATCH_SIZE = 256 # Depends upon GPU memory and WIDTH and HEIGHT
        self.WIDTH=256 # crop size
        self.HEIGHT=256#
        self.STRIDE=32 #224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            