
import os


# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons

        self.training_base_dir = '/home/sizhuo/Desktop/denmark10cm/poly_rect_utm_v4_cleaned/'
        self.training_area_fn = 'rectangles_utm.shp'
        self.training_polygon_fn = 'polygons_utm.shp'
        #################################################################################
        #################################################################################
        # self.training_base_dir = '/home/sizhuo/Desktop/denmark10cm/poly_rect_utm_test_v4/'
        # self.training_area_fn = 'rectangles_utm.shp'
        # self.training_polygon_fn = 'polygons_utm.shp'


        # For reading multichannel images
        # self.raw_image_base_dir = './raw_tif_utm_nocompress_v2/'
        self.raw_image_base_dir = './raw_tif_utm_v4/' # training data has both compressed and noncompressed
        self.raw_image_file_type = '.tif'
        self.raw_image_prefix = 'reset_2018_' # in v4 actually some are not reset
        # self.raw_image_prefix2 = '2018_'
        # auxiliary tif (chm etc.)
        # self.raw_aux_prefix = 'CHMNoBuild_'
        # self.raw_aux_prefix = 'CHM_'
        self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # must be a list
        self.prediction_pre = 'det'
        # Channel names in order for the raw tif image
        self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']
        # channel names in order for the auxiliary tif image
        # self.aux_channel_prefixs = ['chm']
        self.aux_channel_prefixs = [['chm'], ['ndvi']] # must be a nested list
        # All channel in one single raster file or not
        self.single_raster = False
        self.bands = list(range(len(self.raw_channel_prefixs)))
        # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
        self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

        # For writing the extracted images and their corresponding annotations and boundary file
        # self.path_to_write = './extracted_data_2aux_test_centroid_normTrain/' # 18k polygons
        self.path_to_write = '/home/sizhuo/Desktop/denmark10cm/extracted_data_testtest/'
        # self.path_to_write = './extracted_data_2aux_test/' # 3k polygons
        self.show_boundaries_during_processing = True #False
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


# #
# import os


# # Configuration of the parameters for the 1-Preprocessing.ipynb notebook
# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         self.training_base_dir = './poly_rect_utm_test_v4/'
#         self.training_area_fn = 'rectangles_all_classes_2.shp'
#         self.training_polygon_fn = 'polygons_all_classes_2.shp'


#         # For reading multichannel images
#         self.raw_image_base_dir = './raw_tif_utm_v4/' # training data has both compressed and noncompressed
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = 'reset_2018_' # in v4 actually some are not reset
#         self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # must be a list
#         self.detchm = 0
#         self.prediction_pre = 'det' # to ignore predictions
#         # Channel names in order for the raw tif image
#         self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = [['chm'], ['ndvi']] # must be a nested list
#         # All channel in one single raster file or not
#         self.single_raster = False
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

#         # For writing the extracted images and their corresponding annotations and boundary file
#         # self.path_to_write = './extracted_data_2aux_v4_cleaned_centroids_detCHM_cleanedCHM_norm2/' # 18k polygons
#         self.path_to_write = './extracted_data_2aux_test_v4_centroids_val/' # 3k polygons
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['red', 'green', 'blue', 'infrared']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')




# # test with Copernicus map
# import os


# # Configuration of the parameters for the 1-Preprocessing.ipynb notebook
# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         # For reading the training areas and polygons

#         self.training_base_dir = './poly_rect_utm_v4_cleaned/'
#         self.training_area_fn = 'rectangles_utm.shp'
#         self.training_polygon_fn = 'polygons_utm.shp'
#         #################################################################################
#         #################################################################################
#         # self.training_base_dir = './poly_rect_utm_test/'
#         # self.training_area_fn = 'rectangles.shp'
#         # self.training_polygon_fn = 'polygons.shp'


#         # For reading multichannel images
#         # self.raw_image_base_dir = './raw_tif_utm_nocompress_v2/'
#         # self.raw_image_base_dir = './raw_tif_utm_v4/' # training data has both compressed and noncompressed
#         self.raw_image_base_dir = '/mnt/ssdc/Denmark/post_analysis/opernicus_2018dk/TCD_2018_010m_dk_03035_v020/DATA/'
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = 'TCD_' # in v4 actually some are not reset
#         # self.raw_image_prefix2 = '2018_'
#         # auxiliary tif (chm etc.)
#         # self.raw_aux_prefix = 'CHMNoBuild_'
#         # self.raw_aux_prefix = 'CHM_'
#         self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # must be a list
#         self.prediction_pre = 'det'
#         # Channel names in order for the raw tif image
#         self.raw_channel_prefixs = ['cover']
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = [['chm'], ['ndvi']] # must be a nested list
#         # All channel in one single raster file or not
#         self.single_raster = True
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

#         # For writing the extracted images and their corresponding annotations and boundary file
#         self.path_to_write = './extracted_copernicus_test/' # 18k polygons
#         # self.path_to_write = './extracted_data_2aux_test/' # 3k polygons
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['cover']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')





# # for creating data using detchm
# import os


# # Configuration of the parameters for the 1-Preprocessing.ipynb notebook
# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         # For reading the training areas and polygons

#         # # for training areas
#         # self.training_base_dir = './poly_rect_utm_v4_cleaned/'
#         # self.training_area_fn = 'rectangles_utm_linear.shp'
#         # self.training_polygon_fn = 'polygons_utm_linear.shp'#'centroids.shp'
#         #################################################################################
#         #################################################################################
#         # # for testing areas
#         # self.training_base_dir = './poly_rect_utm_test_v4/'
#         # self.training_area_fn = 'rectangles_linear.shp'
#         # self.training_polygon_fn = 'polygons_linear.shp'
#         # # for testing areas
#         self.training_base_dir = './poly_rect_utm_test_v4/'
#         self.training_area_fn = 'rectangles_all_classes.shp'
#         self.training_polygon_fn = 'polygons_all_classes.shp'


#         # For reading multichannel images
#         # self.raw_image_base_dir = './raw_tif_utm_nocompress_v2/'
#         self.raw_image_base_dir = './raw_tif_utm_v4/' # training data has both compressed and noncompressed
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = 'reset_2018_' # in v4 actually some are not reset
#         # self.raw_image_prefix2 = '2018_'
#         # auxiliary tif (chm etc.)
#         # self.raw_aux_prefix = 'CHMNoBuild_'
#         # self.raw_aux_prefix = 'CHM_'
#         # self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # must be a list
#         self.raw_aux_prefix = ['det_CHM__', 'ndvi_2018_'] # must be a list
#         self.detchm = 1
#         self.prediction_pre = 'det' # to ignore predictions
#         # Channel names in order for the raw tif image
#         self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = [['chm'], ['ndvi']] # must be a nested list
#         # All channel in one single raster file or not
#         self.single_raster = False
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

#         # For writing the extracted images and their corresponding annotations and boundary file
#         # self.path_to_write = './extracted_data_2aux_v4_cleaned_centroids_detCHM_cleanedCHM_norm2/' # 18k polygons
#         self.path_to_write = './extracted_data_2aux_test_v4_centroids_all_classes_final_detchm_tttt/' # 3k polygons
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['red', 'green', 'blue', 'infrared']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')

# # # for generating density map for chm
# import os


# # Configuration of the parameters for the 1-Preprocessing.ipynb notebook
# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         # For reading the training areas and polygons

#         self.training_base_dir = './poly_rect_utm_test_v2_addmore/' #'./poly_rect_utm_v4_cleaned/' #'./poly_rect_utm_test/' #
#         self.training_area_fn = 'rectangles.shp'
#         self.training_polygon_fn = 'polygons_utm.shp' #'testpolymax.shp' #'polygons_chmmax.shp'#'testpolymax.shp'# 'polygons_utm.shp'#'centroids.shp'
#         # #################################################################################
#         # #################################################################################
#         # self.training_base_dir = './poly_rect_utm_test/'
#         # self.training_area_fn = 'rectangles.shp'
#         # self.training_polygon_fn = 'polygons.shp'

#         # self.kernelsize = 9 # odd number
#         # For reading multichannel images
#         # self.raw_image_base_dir = './raw_tif_utm_nocompress_v2/'
#         self.raw_image_base_dir = './raw_tif_utm_v4/' # training data has both compressed and noncompressed
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = 'reset_2018_' # in v4 actually some are not reset
#         # self.raw_image_prefix2 = '2018_'
#         # auxiliary tif (chm etc.)
#         # self.raw_aux_prefix = 'CHMNoBuild_'
#         # self.raw_aux_prefix = 'CHM_'
#         self.raw_aux_prefix = ['CHM_', 'ndvi_2018_'] # must be a list
#         self.prediction_pre = 'det'
#         # Channel names in order for the raw tif image
#         self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = [['chm'], ['ndvi']] # must be a nested list
#         # All channel in one single raster file or not
#         self.single_raster = False
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = list(list(range(len(c))) for c in self.aux_channel_prefixs) # for multi auxs

#         # For writing the extracted images and their corresponding annotations and boundary file
#         self.path_to_write = './extracted_data_2aux_test_v2_centroids2/' # 18k polygons
#         # self.path_to_write = './extracted_data_2aux_test/' # 3k polygons
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['red', 'green', 'blue', 'infrared']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')





# # for finland data
# import os

# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         # For reading the training areas and polygons

#         self.training_base_dir = '/mnt/ssdc/Finland/'
#         self.training_area_fn = 'rectangles.shp'
#         self.training_polygon_fn = 'polygons_3067.shp'



#         # For reading multichannel images
#         # self.raw_image_base_dir = './raw_tif_utm_nocompress_v2/'
#         self.raw_image_base_dir = '/mnt/ssdc/Finland/segcount_test/' # training data has both compressed and noncompressed
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = ['L', 'M', 'N', 'P', 'Q', 'T'] # in v4 actually some are not reset
#         # self.raw_image_prefix2 = '2018_'
#         # auxiliary tif (chm etc.)
#         # self.raw_aux_prefix = 'CHMNoBuild_'
#         # self.raw_aux_prefix = 'CHM_'
#         self.raw_aux_prefix = None
#         self.prediction_pre = 'det'
#         # Channel names in order for the raw tif image
#         self.raw_channel_prefixs = ['infrared', 'green', 'blue']
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = None
#         # All channel in one single raster file or not
#         self.single_raster = False
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = None

#         # For writing the extracted images and their corresponding annotations and boundary file
#         # self.path_to_write = './extracted_data_2aux_v4/' # 18k polygons
#         self.path_to_write = '/mnt/ssdc/Finland/segcount_test/extracted_centroids_3/' # 18k polygons
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['infrared', 'green', 'blue']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')



# # test segmentation of buildings
# class Configuration:
#     '''
#     Configuration for the first notebook.
#     Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
#     '''
#     def __init__(self):
#         # For reading the training areas and polygons
#         # self.training_base_dir = './poly_rect_utm_v1/'
#         # self.training_area_fn = 'rectangles_utm2.shp'
#         # self.training_polygon_fn = 'polygons_utm2.shp'

#         self.training_base_dir = './buildings/'
#         self.training_area_fn = 'build_rectangles_utm.shp'
#         self.training_polygon_fn = 'build_polygons_utm.shp'

#         # For reading multichannel images
#         self.raw_image_base_dir = './raw_tif_utm_nocompress/'
#         self.raw_image_file_type = '.tif'
#         self.raw_image_prefix = 'CHM_'
#         # auxiliary tif (chm etc.)
#         # self.raw_aux_prefix = 'CHM_'
#         self.raw_aux_prefix = None
#         self.prediction_pre = 'det'
#         # Channel names in order for the raw tif image
#         # self.raw_channel_prefixs = ['red', 'green', 'blue', 'infrared']

#         self.raw_channel_prefixs = ['ch'] #canopy height
#         # channel names in order for the auxiliary tif image
#         # self.aux_channel_prefixs = ['chm']
#         self.aux_channel_prefixs = None
#         # All channel in one single raster file or not
#         self.single_raster = True
#         self.bands = list(range(len(self.raw_channel_prefixs)))
#         # self.aux_bands = list(range(len(self.aux_channel_prefixs)))
#         self.aux_bands = None

#         # For writing the extracted images and their corresponding annotations and boundary file
#         self.path_to_write = './extracted_build_v01/'
#         self.show_boundaries_during_processing = True #False
#         self.extracted_file_type = '.png'
#         self.extracted_filenames = ['ch']
#         self.extracted_annotation_filename = 'annotation'
#         self.extracted_boundary_filename = 'boundary'


#         # Path to write should be a valid directory
#         # assert os.path.exists(self.path_to_write)
#         if not os.path.exists(self.path_to_write):
#             os.makedirs(self.path_to_write)

#         if not len(os.listdir(self.path_to_write)) == 0:
#             print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')
