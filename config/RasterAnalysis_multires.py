


# large scale prediction
import os
class Configuration:
    '''
    Configuration for prediction
    '''
    def __init__(self):

        # Input related variables
        self.input_image_dir_base = '/media/RS_storage/Aerial/Denmark/summer2020/Final_Tiffjpeg/'
        self.input_chm_dir = '/mnt/ssda/denmark_CHM/'
        self.input_image_type = '.tif'
        self.input_image_pref = '2020_'
        self.channel_names1 = ['red', 'green', 'blue', 'infrared', 'ndvi']
        self.channels = [0, 1, 2, 3] # read all bands
        self.aux_prefs = ['CHM_'] # prefix for the chm files
        self.trained_model_path = ['./saved_models/segcountdensity/trees_segcountdensity_20210512-1800_Adam_weightmap_tversky_redgreenblueinfraredndvichm_256_84_frames_5weight_tversky_mse100_complex5.h5',
                                   ]
        self.output_dir_base = '/mnt/ssdc/Denmark/summer_predictions2020/'
        self.change_input_size = 0 # whether change model input size, set to False by default
        self.output_image_type = '.tif'
        self.output_prefix = 'seg_'
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'
        self.single_raster = False
        self.aux_data = True
        self.BATCH_SIZE = 8 # Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=256 # Should be same as the WIDTH used for training the model
        self.HEIGHT=256 # Should be same as the HEIGHT used for training the model
        self.STRIDE=196 #224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
        if not os.path.exists(self.output_dir_base):
            os.makedirs(self.output_dir_base)
