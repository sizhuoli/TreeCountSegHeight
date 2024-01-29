
import os


# Configuration of the parameters for segmentation and counting preprocessing
class Configuration:
    
    def __init__(self):
    	# dir containing annotating areas and tree polygons in shp files
        self.training_base_dir = ''
        self.training_area_fn = 'rectangles.shp' # annotating areas where tree crowns are delineated inclusively
        self.training_polygon_fn = 'polygons.shp' # tree crowns


        # For reading multichannel images
        self.raw_image_base_dir = ''  # dir containing optical images (color bands)
        self.raw_image_file_type = '.tif' # image format
        self.raw_image_prefix = '' # image prefix if any
        self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # channel name for extra bands
        self.prediction_pre = 'det' # prediction suffix
        # Channel names in order for the raw tif image
        self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']
        # channel names in order for the auxiliary tif image
        self.aux_channel_prefixs = [['chm'], ['ndvi']]
        # All channel in one single raster file or not
        self.single_raster = False
        self.normalize = False # keep raw image pixel values
        self.bands = list(range(len(self.raw_channel_prefixs)))
        self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

        # For writing the extracted images and their corresponding annotations and boundary file
        self.path_to_write = './extracted_data_test_raw/'
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
        self.extracted_filenames = ['red', 'green', 'blue', 'infrared']
        self.extracted_annotation_filename = 'annotation'
        self.extracted_boundary_filename = 'boundary'
        self.kernel_size_svls = 15 
        self.kernel_sigma_svls = 5

        # Path to write should be a valid directory
        # assert os.path.exists(self.path_to_write)
        if not os.path.exists(self.path_to_write):
            os.makedirs(self.path_to_write)

        if not len(os.listdir(self.path_to_write)) == 0:
            print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')

