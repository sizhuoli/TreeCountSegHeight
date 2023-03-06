# multi tasks (seg count)
import os
from functools import reduce

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the channels and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        self.ntasks = 2
        self.multires = 0 # det CHM has same res as color bands
        # dir where extrected images are
        # self.base_dir = './extracted_build_v01/'
        # self.base_dir = './extracted_data_2aux_v4_cleaned_centroids_detCHM_cleanedCHM_norm_cleaned_nolinear/'
        self.base_dir = './extracted_data_2aux_v4_cleaned_centroid_raw/'
        self.image_type = '.png'
        self.grayscale = 0
#         self.ndvi_fn = 'ndvi'
#         self.pan_fn = 'pan'
        # self.extracted_filenames = ['red', 'green', 'blue', 'infrared', 'ndvi', 'chm']
        self.extracted_filenames = ['red', 'green', 'blue']
        # self.channel_names1 = ['red', 'green', 'blue', 'infrared', 'ndvi']
        self.channel_names1 = ['red', 'green', 'blue']
        # self.channel_names2 = ['chm']
        self.image_channels1 = len(self.channel_names1)
        # self.image_channels2 = len(self.channel_names2)
        # self.extracted_filenames = ['ch']
        # self.extracted_filenames = ['red', 'green', 'blue', 'infrared', 'ndvi']
        # self.extracted_filenames = ['infrared', 'green', 'blue']
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.density_fn = 'ann_kernel'
        self.boundary_weights = 3
        self.single_raster = False
        self.aux_data = 0
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        # (Input + Output) channels
        self.image_channels = len(self.channel_names1)
        self.all_channels = self.image_channels + 3
        # self.patch_size = (256,256,self.all_channels) # Height * Width * (Input + Output) channels
        self.patch_size = (512,512,self.all_channels) # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0
        self.val_ratio = 0.1
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0 #0.4
        
        # upscale for aux chm layer
        self.upscale_factor = 2
        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        self.patch_dir = './extracted_data_2aux_v4/patches{}'.format(self.patch_size[0])
        self.frames_json = os.path.join(self.patch_dir,'frames_list.json')


        # Shape of the input data, height*width*channel
        # self.input_shape = (256,256,self.image_channels)
        self.input_shape = (512,512,self.image_channels)
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]
        # self.label_weight_channel = [self.image_channels, self.image_channels+1]
        self.ifBN = 1
        self.inputBN = 1
        self.task_ratio = [100, 1000, 10000] # list of 3 indicating the ratio for density branch at epochs 1, 100, 500  
        
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 4
        self.NB_EPOCHS = 1000

        # number of validation images to use
        self.VALID_IMG_COUNT = 100 #200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 500 #1000
        
        
        
        # saving 
        self.OPTIMIZER_NAME = 'Adam_e4'
        self.LOSS_NAME = 'WTversky'
        self.LOSS2 = 'Mse'#'Mse100'
        self.model_name = 'complex5_inputBN_rawinput'
        
        # self.model_name = 'efficientnet_B2'
        # self.backbone = "efficientnetb2"
        # # efficientnet code
        # self.model_name = 'unet'
        # self.backbone = "standard_2inputs"
        self.sufix = ''
        self.callbackImSave = './ImageCallbacks/'
        self.chs = reduce(lambda a,b: a+str(b), self.extracted_filenames, '')
        self.log_img = 1 # log images on the way
        self.model_path = './saved_models/segcountdensity/'


