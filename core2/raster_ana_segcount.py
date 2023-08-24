#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 01:33:25 2021

@author: sizhuo
"""

import os
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
# import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps
import glob
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape
from skimage.transform import resize

import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
import time

from itertools import product
import cv2
from sklearn.metrics import mean_absolute_error, median_absolute_error

import sys
import math
# from core.UNet_multires import UNet

# from rasterstats import zonal_stats
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from collections import defaultdict
import random
from rasterio.enums import Resampling
from osgeo import ogr, gdal

from scipy.optimize import curve_fit
from matplotlib import colors
import glob
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
import csv
from shapely.geometry import shape
from rasterio.windows import Window
from rasterio.features import shapes
import multiprocessing
from itertools import product
from pathlib import Path
import ipdb
from core.frame_info_multires import FrameInfo, image_normalize
class anaer:
    def __init__(self, config, DK = 1, largescale = 0, pred = 1, patches = 0, polygonize = 0):

        self.config = config
        if not polygonize:
            import tensorflow as tf
            print(tf.__version__)

            print(tf.config.list_physical_devices('GPU'))
            from core.UNet_attention_segcount import UNet
            from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
            from core.optimizers import adaDelta, adagrad, adam, nadam
            from core.frame_info_multires import FrameInfo, image_normalize
            from core.dataset_generator_multires import DataGenerator
            from core.split_frames import split_dataset
            from core.visualize import display_images
            from tensorflow.keras.models import load_model

            if not self.config.change_input_size:
                if DK:
                    if largescale:
                        self.all_files = load_files(config)
                    else:
                        self.all_files = load_files_2018(config)
                        self.all_files1819 = load_files_201819(config)
                else:
                    if not patches: # still large images, need to crop into patches
                        self.all_files = load_files(config)
                        # ipdb.set_trace()
                    elif patches:
                        self.all_files = None
                if pred: # make predicitons
                    OPTIMIZER = adam
                    # import ipdb
                    # ipdb.set_trace()
                    self.models = []
                    for mod in config.trained_model_path:
                        modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                        modeli.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
                        self.models.append(modeli)


                    self.model_chm = load_model(self.config.trained_model_path_chm, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                    self.model_chm.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_chm': tf.keras.losses.Huber()},
                                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                                        'output_chm':[tf.keras.metrics.RootMeanSquaredError()]})
                    
                else:
                    print('only polygonization')

    # =============================================================================
            elif self.config.change_input_size:
                OPTIMIZER = adam
                self.all_files = load_files(config)
                self.models = []
                for mod in config.trained_model_path:
                    modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                    modeli.summary()
                    self.weiwei = modeli.get_weights()
                    # print(modeli.input[0])
                    print(modeli.layers[0]._batch_input_shape)
                    if self.config.rgb2gray:
                        modeli.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, 1)
                    else:
                        modeli.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channel_names1))

                    print(modeli.layers[0]._batch_input_shape)
                    # modeli.layers[0]._batch_input_shape = oldInputShape
                    new_model = tf.keras.models.model_from_json(modeli.to_json())


                    # copy weights from old model to new one
                    for layer in new_model.layers:
                        try:
                            layer.set_weights(modeli.get_layer(name=layer.name).get_weights())
                        except:
                            print("Could not transfer weights for layer {}".format(layer.name))



                    self.weiwei2 = new_model.get_weights()
                    print(new_model.summary())

                    new_model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
                    self.models.append(new_model)
                    # self.models.append(modeli)
                # change input size for chm model as well
                self.model_chm = load_model(self.config.trained_model_path_chm, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                # self.model_chm.compile(optimizer=OPTIMIZER, loss={'output_seg':tversky, 'output_chm': tf.keras.losses.Huber()},
                #               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
                #                     'output_chm':[tf.keras.metrics.RootMeanSquaredError()]})

                if self.config.addndvi:
                    self.model_chm.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channels)+1)

                elif not self.config.addndvi:
                    self.model_chm.layers[0]._batch_input_shape = (None, self.config.input_size, self.config.input_size, len(self.config.channels))

                self.new_model_chm = tf.keras.models.model_from_json(self.model_chm.to_json())


                # copy weights from old model to new one
                for layer in self.new_model_chm.layers:
                    try:
                        layer.set_weights(self.model_chm.get_layer(name=layer.name).get_weights())
                    except:
                        print("Could not transfer weights for layer {}".format(layer.name))
                print(self.new_model_chm.summary())
                self.model_chm = self.new_model_chm
                self.new_model_chm.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])




    def pred_segcount(self):
        pred_segcountdk(self.all_files, self.models, self.config)

    def pred_chm(self):
        predict_chm(self.config, self.all_files, self.model_chm)

    def pred_3tasks(self):
        self.heights_pr, self.heights_gt, _ = predict_segcount_chm(self.config, self.all_files, self.models, self.model_chm, self.config.output_dir)

    def pred_3tasks_detchm(self):
        c = predict_segcount_chm_dk_detchm(self.config, self.all_files, self.models, self.model_chm, self.config.output_dir)


    def pred_sampling_eva_DK(self, num = 10, sample_fixed = True):
        # randomly sample tifs from all files, pred and report errors
        # exclude training frames
        self.train_frames = load_train_files(self.config)
        # self.use_files = list(set(self.all_files) - set(self.train_frames))

        self.use_files = [i for i in self.all_files if i[1] not in self.train_frames]
        self.use_files1819 = [i for i in self.all_files1819 if i[1] not in self.train_frames]
        print('before exclude', len(self.all_files))
        print('exclude train', len(self.use_files))
        print('exclude train 1819', len(self.use_files1819))
        # sample randomly from the files
        if sample_fixed:
            # fix seed
            random.seed(1)
        # self.use_files_sampled = random.sample(self.use_files, num)
        # print('sampled', len(self.use_files_sampled))
        # maes = predict_large(self.config, self.use_files_sampled, self.model)
        # sampled based on forest type
        self.typefile = gps.read_file(self.config.typefile)
        # print(self.typefile.head())
        self.h_prs = []
        self.h_gts = []
        for ty in range(3):
            # allheight = []
            print('forest type', ty)
            local_output_dir = self.config.output_dir + 'forestType_' + str(ty) + '/'
            if not os.path.exists(local_output_dir):
                os.makedirs(local_output_dir)
            path_, dirs_, files_ = next(os.walk(local_output_dir))
            file_count = len(files_)
            if file_count < int(5*num):
                self.use_files_ft = []
                if ty == 0 or ty == 1: # sample from 2018 only for type 0,1
                    # for each forest type, sample and compute errors
                    # self.use_files_ft = [i for i in self.use_files if np.unique(self.typefile[self.typefile['location'] ==i[1]]['_majority'].values) == ty]
                    for j in range(len(self.typefile)):
                        try:
                            if np.unique(self.typefile[self.typefile['location'] ==self.use_files[j][1]]['_majority'].values) ==ty:
                                self.use_files_ft.append(self.use_files[j])
                        except: continue
                elif ty == 2: # sample from 2018 and 2019 for type 2
                    for j in range(len(self.typefile)):
                        try:
                            if np.unique(self.typefile[self.typefile['location'] ==self.use_files1819[j][1]]['_majority'].values) ==ty:
                                self.use_files_ft.append(self.use_files1819[j])
                        except: continue
                print('forest type filtered ')
                if len(self.use_files_ft) < num:
                    print('Samples not enough, no. tifs:', len(self.use_files_ft))
                    # self.use_files_ft_sampled = self.use_files_ft
                    # # check forest type
                    # print('checking forest type of sampled tifs')
                    # for iid in range(len(self.use_files_ft_sampled)):
                    #     print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)

                    # # maes_ft, avg_h, all_gt_hei = predict_large(self.config, self.use_files_ft_sampled, self.model)
                    # local_output_dir = self.config.output_dir + 'forestType_' + str(ty) + '/'
                    # if not os.path.exists(local_output_dir):
                    #     os.makedirs(local_output_dir)
                    # self.heights_pr, self.heights_gt, _ = predict_segcount_chm(self.config, self.use_files_ft_sampled, self.models, self.model_chm, local_output_dir)
                    # self.h_prs.append(self.heights_pr)
                    # self.h_gts.append(self.heights_gt)
                    # # plt.figure(figsize = (6,6))
                    # # plt.title('height hist - forest type '+str(ty), fontsize = 14)
                    # # plt.hist(all_gt_hei, bins=50, density = 1)
                    # # self.ft_errors.append(maes_ft)
                    # # self.avg_hs.append(avg_h)
                    print('CHECK*******************************************')
                else:
                    counter = 0
                    while counter < num:
                        self.use_files_ft_sampled = random.sample(self.use_files_ft, num)
                        # check forest type
                        print('checking forest type of sampled tifs')
                        for iid in range(len(self.use_files_ft_sampled)):
                            print(self.typefile[self.typefile['location'] ==self.use_files_ft_sampled[iid][1]]['_majority'].values)
                        # check chm max height

                        for fullPath, filename in tqdm(self.use_files_ft_sampled):
                            coor = fullPath[-12:-9]+ fullPath[-8:-5]
                            chmpath = os.path.join(self.config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                            # CHM_640_59_TIF_UTM32-ETRS89
                            # aux2 = fullPath.replace(self.config.input_image_pref, self.config.aux_prefs[1]).replace(self.config.input_image_dir, chmpath)
                            aux2 = os.path.join(chmpath, filename.replace(self.config.input_image_pref, self.config.aux_prefs[1]))
                            try:
                                with rasterio.open(aux2) as chm:
                                    # print('image', filename)
                                    # print('chm', os.path.basename(aux2))
                                    if ty == 0:
                                        height_lim = 45
                                    elif ty == 1 or ty == 2:
                                        height_lim = 45
                                    if chm.read().max() < height_lim:
                                        print('ok chm max', counter)
                                        ndvif0 = filename.replace(self.config.input_image_pref, self.config.aux_prefs[0])

                                        ndvi_f = glob.glob(f"{self.config.input_ndvi_dir}/**/{ndvif0}")
                                        if len(ndvi_f) != 0:
                                            counter += 1
                                            print('ok ndvi file', len(ndvi_f))

                            except:
                                print(aux2)
                                print('NO CHM file, water etc.')
                                continue
                    print('all samples do not contain reference height errors')
                    # maes_ft, avg_h, all_gt_hei = predict_large(self.config, self.use_files_ft_sampled, self.model)

                    self.heights_pr, self.heights_gt, _ = predict_segcount_chm(self.config, self.use_files_ft_sampled, self.models, self.model_chm, local_output_dir)
                    self.h_prs.append(self.heights_pr)
                    self.h_gts.append(self.heights_gt)
            else:
                print('forest type analyzed!')
        #         plt.figure(figsize = (6,6))
        #         plt.title('height hist - forest type '+str(ty), fontsize = 14)
        #         plt.hist(all_gt_hei, bins=50, density = 1)

        #         self.ft_errors.append(maes_ft)
        #         self.avg_hs.append(avg_h)
        # plot_errors_ft(self.ft_errors, self.avg_hs)

        return #


    def polyonize_sampleDK(self, postproc_gridsize = (2, 2)):

        from osgeo import ogr, gdal

        # all_dirs = glob.glob(f"{self.config.output_dir}/*/")
        # # ipdb.set_trace()
        # for di in all_dirs:
        #
        #     polygons_dir = os.path.join(di, "polygons")
        #
        #     if not os.path.exists(polygons_dir):
        #
        #         create_polygons(di, polygons_dir, di, postproc_gridsize, postproc_workers = 40)
        all_dirs = [self.config.output_dir]
        # ipdb.set_trace()
        for di in all_dirs:

            polygons_dir = os.path.join(di, "polygons")

            # if not os.path.exists(polygons_dir):

            create_polygons(di, polygons_dir, di, postproc_gridsize, postproc_workers = 40)


    def pred_3tasks_fi(self):
        predict_segcount_chm_fi(self.config, self.all_files, self.models, self.model_chm, self.config.output_dir)

    def pred_3tasks_fi_patches(self):
        self.files = load_files_FI_patches(self.config)
        local_op_dir = self.config.output_dir + 'extracted_pred/'
        if not os.path.exists(local_op_dir):
            os.makedirs(local_op_dir)
        predict_segcount_chm_fi(self.config, self.files, self.models, self.model_chm, local_op_dir)
        return

    def polyonize_FI(self, DK = 0):
        from osgeo import ogr, gdal

        local_op_dir = self.config.output_dir + 'extracted_pred/'
        polygons_dir = self.config.output_dir + 'extracted_pred/polygons/'
        create_polygons(local_op_dir, polygons_dir, local_op_dir, postproc_gridsize = (2, 2), postproc_workers = 40, DK = DK)

    def preprocess_cropping_FI(self):


        opdir = self.config.input_image_dir + 'extracted/'
        # crop big FI tifs to samller patches for fast postprocessing
        if not os.path.exists(opdir):
            os.makedirs(opdir)
        gg = list(range(0, 12000, 2000))
        grids = list(product(gg, gg))
        for f in self.all_files:
            curf = f[0]

            c = 1
            with rasterio.open(curf) as src:

                profile = src.profile
                filename = os.path.basename(f[1])
                xsize, ysize = 2000, 2000

                for rr, cc in grids:
                    window = Window(rr, cc , xsize, ysize)
                    transform = src.window_transform(window)

                    profile = src.profile
                    profile.update({
                        'height': xsize,
                        'width': ysize,
                        'driver': 'GTiff',
                        'transform': transform
                          })
                    opf = opdir +filename.replace('.jp2', '_' +str(c)+ '.tif')
                    # suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'

                    # print(opf)
                    with rasterio.open(opf, 'w', **profile) as dst:
                        # Read the data from the window and write it to the output raster
                        rrd= src.read(window=window)
                        # print(rrd.shape)
                        # print(profile)
                        dst.write(rrd)
                        dst = None
                        c += 1
            # print(curf)

            curchm = os.path.join(self.config.gt_chm_dir, 'CHM_' + Path(curf).stem + '_2019.tif')
            # print(curchm)
            c = 1
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            try:
                with rasterio.open(curchm) as src2:

                    xsize, ysize = 1000, 1000
                    filename = os.path.basename(f[1])

                    for rr, cc in grids:
                        window = Window(int(rr/2), int(cc/2) , xsize, ysize)
                        transform = src2.window_transform(window)
                        # np.array(transform)[2] =
                        # print(transform)
                        # Create a new cropped raster to write to
                        profile = src2.profile
                        profile.update({
                            'height': xsize,
                            'width': ysize,
                            'driver': 'GTiff',
                            'transform': transform
                              })
                        opf = opdir +filename.replace('.jp2', '_' +str(c)+ '_chm.tif')
                        # suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'

                        # print(opf)
                        with rasterio.open(opf, 'w', **profile) as dst:
                            # Read the data from the window and write it to the output raster
                            rrd= src2.read(window=window)
                            # print(rrd.shape)
                            # print(profile)
                            dst.write(rrd)
                            dst = None

                            c += 1
            except:
                print('**************************************************CHM not from the same year!! using uusin CHM/')
                curchm = curchm.replace('2019', 'uusin')
                print(curchm)
                c = 1

                with rasterio.open(curchm) as src2:

                    xsize, ysize = 1000, 1000
                    filename = os.path.basename(f[1])

                    for rr, cc in grids:
                        window = Window(int(rr/2), int(cc/2) , xsize, ysize)
                        transform = src2.window_transform(window)
                        # np.array(transform)[2] =
                        # print(transform)
                        # Create a new cropped raster to write to
                        profile = src2.profile
                        profile.update({
                            'height': xsize,
                            'width': ysize,
                            'driver': 'GTiff',
                            'transform': transform
                              })
                        opf = opdir +filename.replace('.jp2', '_' +str(c)+ '_chm.tif')
                        # suf = '_' + str(self.sel_patches.iloc[c,4])[:-2] + '_' + str(self.sel_patches.iloc[c, 1])[:-2] + '.tif'

                        # print(opf)
                        with rasterio.open(opf, 'w', **profile) as dst:
                            # Read the data from the window and write it to the output raster
                            rrd= src2.read(window=window)
                            # print(rrd.shape)
                            # print(profile)
                            dst.write(rrd)
                            dst = None

                            c += 1


        return


    def segcount_CN(self,  inf = 1, th = 0.5):
        predict_segcount_cn(self.config, self.all_files, self.models, self.model_chm, self.config.output_dir, eva = 0, inf = inf, th = th, rgb2gray = self.config.rgb2gray)
        return

def load_train_files(config):
    all_files = []
    for root, dirs, files in os.walk(config.training_frames_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append(file)
    print('Number of raw tif to exclude (training):', len(all_files))
    # print(all_files)
    return all_files

def load_files(config):
    exclude = set(['water_new', 'md5', 'pred', 'test_kay'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    if config.fillmiss: # only fill the missing predictions (smk) # while this include north
        doneff = gps.read_file(config.grids)
        donef2 = list(doneff['filepath'])
        done_names= set([os.path.basename(f)[:6] for f in donef2])
        all_files = [f for f in all_files if os.path.splitext(f[1])[0] not in done_names]
    # print(all_files)
    print('Number of missing tif to predict:', len(all_files))

    return all_files

def load_files_FI_patches(config):
    exclude = set(['water_new', 'md5'])
    all_files = []
    # for root, dirs, files in os.walk(config.output_dir+'extracted/'):
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith('tif') and 'chm' not in file:
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw patches to predict:', len(all_files))
    # print(all_files)
    return all_files

def load_files_2018(config):
    # 2018 ims in 35-39, 51-53
    all_files = []
    for fo in ['35', '36', '37', '38', '39']:
        fod = os.path.join(config.input_image_dir, 'SOP2018-83_'+fo)
        fs = glob.glob(f"{fod}/**/2018_1km*.tif")
        if len(fs) == 0:
            fs = glob.glob(f"{fod}/2018_1km*.tif")

        aa = [(fn, os.path.basename(fn)) for fn in fs]
        all_files.extend(aa)
        # glob.glob(f"{raster_dir}/det_1km*.tif")
    for fo in ['51', '52', '53']:
        fod = os.path.join(config.input_image_dir, 'SOP2018-85_'+fo)
        fs = glob.glob(f"{fod}/**/2018_1km*.tif")
        if len(fs) == 0:
            fs = glob.glob(f"{fod}/2018_1km*.tif")

        aa = [(fn, os.path.basename(fn)) for fn in fs]
        all_files.extend(aa)

    return all_files

def load_files_201819(config):  # type 2 2018 not enough
    # 2018 ims in 35-39, 51-53
    # 2019 ims in 40-50
    all_files = []
    for fo in ['35', '36', '37', '38', '39']:
        fod = os.path.join(config.input_image_dir, 'SOP2018-83_'+fo)
        fs = glob.glob(f"{fod}/**/2018_1km*.tif")
        if len(fs) == 0:
            fs = glob.glob(f"{fod}/2018_1km*.tif")

        aa = [(fn, os.path.basename(fn)) for fn in fs]
        all_files.extend(aa)
        # glob.glob(f"{raster_dir}/det_1km*.tif")
    for fo in ['40', '41', '42', '43']:
        fod = os.path.join(config.input_image_dir, 'SOP2018-84_'+fo)
        fs = glob.glob(f"{fod}/**/2018_1km*.tif")
        if len(fs) == 0:
            fs = glob.glob(f"{fod}/2018_1km*.tif")

        aa = [(fn, os.path.basename(fn)) for fn in fs]
        all_files.extend(aa)
    for fo in ['44', '45', '46', '47', '48', '49', '50', '51', '52', '53']:
        fod = os.path.join(config.input_image_dir, 'SOP2018-85_'+fo)
        fs = glob.glob(f"{fod}/**/2018_1km*.tif")
        if len(fs) == 0:
            fs = glob.glob(f"{fod}/2018_1km*.tif")

        aa = [(fn, os.path.basename(fn)) for fn in fs]
        all_files.extend(aa)

    return all_files

def pred_segcountdk(all_files, models, config):
    th = 0.5

    # save count per image in a dateframe

    counts = {}

    outputFiles = []
    nochm = []
    waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    for fullPath, filename in tqdm(all_files):
        #print(filename)
        #print(fullPath)
        # for 1km tif prediction:
        if '1km' in fullPath:
            outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
        else:
            outputFile = os.path.join(config.output_dir, filename.split('.')[0] + config.output_suffix + config.output_image_type)
            
        print(outputFile)
        outputFiles.append(outputFile)
        
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:

            if not config.single_raster and config.aux_data: # multi raster
                with rasterio.open(fullPath) as core:
                    auxx = []
                    if config.input_ndvi_separate:
                        ndvif0 = filename.replace(config.input_image_pref, config.aux_prefs[0])

                        ndvi_f = glob.glob(f"{config.input_ndvi_dir}/**/{ndvif0}")
                        # print(fullPath)
                        # print(fullPath.replace(config.input_image_pref, config.aux_prefs[0]).replace(config.input_tif_dir, config.input_ndvi_dir))
                        auxx.append(ndvi_f[0])

                    coor = fullPath[-12:-9]+ fullPath[-8:-5]
                    chmpath = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                    
                    # CHM_640_59_TIF_UTM32-ETRS89
                    aux2 = fullPath.replace(config.input_image_pref, config.aux_prefs[-1]).replace(config.input_image_dir, chmpath)
                    auxx.append(aux2)
                    try:
                        if config.input_ndvi_separate:
                            # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                            detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        else:
                            detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_rawtif_ndvi(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    except IOError:
                        try:
                            # continue
                            auxx[-1] = waterchm
                            nochm.append(outputFile)
                            if config.input_ndvi_separate:
                                # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                                detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                            else:
                                detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_rawtif_ndvi(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        except:
                            # predict for plot crops
                            try:
                                chmpath = os.path.join(config.input_chm_dir, os.path.basename(fullPath))
                                # print(chmpath, fullPath)
                                if config.input_ndvi_separate:
                                    # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, models, [core, chmpath], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                                else:
                                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_rawtif_ndvi(config, models, [core, chmpath], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                            except:
                                continue
                            
                            
                            # continue
                    ###### check threshold!!!!
                    # seg
                    try:
                        writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                        # print(detectedMaskDens.shape)
                        # print(detectedMaskDens.max(), detectedMaskDens.min())
                        # print(detectedMaskDens.sum())
                        # density
                        print(outputFile.replace(config.output_suffix, config.output_suffix_dens))
                        writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace(config.output_suffix, config.output_suffix_dens), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0, rescale = 0)
                        counts[filename] = detectedMaskDens.sum()
                    except:
                        continue
            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as img:
                    #print(fullPath)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, models, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    # print(detectedMaskDens.shape)
                    # print(detectedMaskDens.max(), detectedMaskDens.min())
                    # print(detectedMaskDens.sum())
                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'int16', th = th, create_countors = False, convert = 0, rescale = 1)
                    counts[filename] = detectedMaskDens.sum()
        else:
            print('File already analysed!', fullPath)
    return


def predict_chm(config, all_files, model):
    maes = {}
    outputFiles = []
    nochm = []
    avg_h = {}
    heights_gt = np.array([])
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix))
        #print(outputFile)
        outputFiles.append(outputFile)
        # outputChmDiff = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.chmdiff_prefix))

        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:

            with rasterio.open(fullPath) as img:
                # locate gt chm
                coor = fullPath[-12:-9]+ fullPath[-8:-5]
                chmdir = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                chmPath = fullPath.replace(config.input_image_dir, chmdir).replace('2018_', 'CHM_')
                if 'a1/' in chmPath:
                    chmPath = chmPath.replace('a1/', '')
                elif 'a2/' in chmPath:
                    chmPath = chmPath.replace('a2/', '')
                # print(chmPath)
                try:
                    # print(chmPath)
                    with rasterio.open(chmPath) as chm:
                        print('image', filename)
                        print('chm', os.path.basename(chmPath))
                        if chm.read().max() > 50:
                            print('GT chm containing large values > 50m, please redo sampling')
                            # skip error files
                            continue
                        detectedMask, detectedMeta = detect_tree_rawtif_ndvi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = 0) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                        #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                        if config.saveresult:
                            writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)
                        pr = detectedMask.flatten()
                        ggt = np.squeeze(chm.read()).flatten()
                        # maei = mean_absolute_error(ggt, pr)
                        # maes[filename] = maei
                        # print('mae', maei)
                        # avg_h[filename] = np.mean(ggt)
                        # print('--------- mean height', np.mean(ggt))
                        # heights_gt = np.append(heights_gt, ggt)
                        # # # cate errors
                        # gtm = int(np.ceil(ggt.max()))
                        # inds = []
                        # intervals = [0, 10, 20, 30, gtm]
                        # for i in range(4):
                        #     indi = [idx for idx,val in enumerate(ggt) if intervals[i] <= val < intervals[i+1]]
                        #     inds.append(indi)


                        # preds = []
                        # gtts = []
                        # for i in range(4):
                        #     predi = pr[inds[i]]
                        #     preds.append(predi)
                        #     gtti = ggt[inds[i]]
                        #     gtts.append(gtti)

                        # inte_maes = []

                        # for i in range(4):

                        #     inte_maes.append(abs(gtts[i] - preds[i]))

                        # inte_maes_all.append(inte_maes)

                except:

                    detectedMask, detectedMeta = detect_tree_rawtif_ndvi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = 0) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                    if config.saveresult:
                        writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)

                    nochm.append(filename)
        else:
            print('File already analysed!', fullPath)

    # if config.saveresult:
    #     w = csv.writer(open(os.path.join(config.output_dir, "maes.csv"), "w"))
    #     for key, val in maes.items():
    #         w.writerow([key, val])

    return #maes, avg_h, heights_gt.flatten()


def pred_segcount_fi(config, all_files, model):
    #segcount
    counts = {}
    th = 0.5
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))

        with rasterio.open(fullPath) as img:

            if config.tasks == 2:
                detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=True,
                                                                                            auxData = config.aux_data, singleRaster=config.single_raster, multires = config.multires, upsample = config.upsample, downsave = config.downsave)
                writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                # density
                writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                counts[filename] = detectedMaskDens.sum()
            elif config.tasks ==1:
                print('didnot check')
                detectedMask, detectedMeta = detect_tree_segcount_fi(config, model, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize = 1, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)


    print('Inference for the area has finished, saving counts to csv file')
    w = csv.writer(open(os.path.join(config.output_dir, "counts.csv"), "w"))
    for key, val in counts.items():
        w.writerow([key, val])

def predict_chm_finland(config, all_files, model):
    # for need to search for CHM files
    maes = {}
    avg_h = {}
    outputFiles = []
    pr = np.array([])
    ggt = np.array([])
    for fullPath, filename in all_files:
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(config.output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        # print(outputFile)
        outputFiles.append(outputFile)
        outputChmDiff = os.path.join(config.output_dir, filename[:-4] + config.chmdiff_suffix + config.output_image_type)
        # print(outputChmDiff)
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:

            with rasterio.open(fullPath) as img:
                # locate gt chm

                chmPath = config.gt_chm_dir  + config.chm_pref + filename[:-4] + config.chm_sufix
                # print(chmPath)
                with rasterio.open(chmPath) as chm:
                    if config.upsample and config.downsave:
                        # detectedMask, detectedMeta = detect_tree_rawtif_fi(config, model, img, config.channels, q1, q3, mins, maxs, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        # print('**********************UPsample for prediction***************************')
                        detectedMask, detectedMeta = detect_tree_rawtif_fi(config, model, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                        # print('**********************Downsample for saving***********************')
                        # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
                        # detectedMask = resize(detectedMask[:, :], (int(detectedMask.shape[0]/2), int(detectedMask.shape[1]/2)), preserve_range=True)
                        # print(detectedMask.mean())
                        # # rescale values
                        # detectedMask = detectedMask * (detectedMask.shape[0] / float(detectedMask.shape[0]/2)) * (detectedMask.shape[1] / float(detectedMask.shape[1]/2))


                    else:
                        detectedMask, detectedMeta = detect_tree_rawtif_fi(config, model, img, config.channels, q1, q3, mins, maxs, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width



                    #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                    writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)
                    # compute diff
                    chmdiff, gt_chm_mask, detected_chm_mask = CHMdiff_fi(detectedMask, chm, filename)
                    writeMaskToDisk(chmdiff, detectedMeta, outputChmDiff, image_type = config.output_image_type, write_as_type = config.chmdiff_dtype, scale = 0)
                    pr = np.append(pr,  detected_chm_mask.flatten())
                    ggt = np.append(ggt, gt_chm_mask.flatten())
                    maei = mean_absolute_error(ggt, pr)
                    maes[filename] = maei
                    print('mae', maei)
                    avg_h[filename] = np.mean(ggt)
                    print('--------- mean height', np.mean(ggt))
        else:
            print('File already analysed!', fullPath)
    MAE = mean_absolute_error(ggt, pr)
    print('MAE for all', MAE)
    return pr, ggt, maes, avg_h


def predict_segcount_chm(config, all_files, model_segcount, model_chm, output_dir):
    counter = 1
    th = 0.5

    # save count per image in a dateframe

    counts = {}
    maes = {}


    avg_h = {}
    heights_gt = np.array([])
    heights_pr = np.array([])
    outputFiles = []
    nochm = []
    waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    for fullPath, filename in tqdm(all_files):
        #print(filename)
        #print(fullPath)
        outputFile = os.path.join(output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
        #print(outputFile)
        outputFiles.append(outputFile)
        outputFileChm = os.path.join(output_dir, filename.replace(config.input_image_pref, config.output_prefix_chm).replace(config.input_image_type, config.output_image_type))

        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:

            if not config.single_raster and config.aux_data: # multi raster
                with rasterio.open(fullPath) as core:
                    auxx = []
                    ndvif0 = filename.replace(config.input_image_pref, config.aux_prefs[0])

                    ndvi_f = glob.glob(f"{config.input_ndvi_dir}/**/{ndvif0}")
                    # print(fullPath)
                    # print(fullPath.replace(config.input_image_pref, config.aux_prefs[0]).replace(config.input_tif_dir, config.input_ndvi_dir))
                    # print(ndvi_f)
                    # print(ndvif0)
                    auxx.append(ndvi_f[0])
                    # auxx.append(fullPath.replace(config.input_image_pref, config.aux_prefs[0]).replace(config.input_image_dir, config.input_ndvi_dir))
                            # with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[0])) as aux0:
                            #     with rasterio.open(fullPath.replace(config.input_image_pref, config.input_aux_pref[1])) as aux1:
                            #     # in this case the resolution of the chm tif is only half of the raw tif, thus upsampling the aux tif
                    # auxx.append(fullPath.replace(config.input_image_pref, config.aux_prefs[1]).replace(config.input_tif_dir, config.input_chm_dir))
                    coor = fullPath[-12:-9]+ fullPath[-8:-5]
                    chmpath = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                    # CHM_640_59_TIF_UTM32-ETRS89
                    # aux2 = fullPath.replace(config.input_image_pref, config.aux_prefs[1]).replace(config.input_image_dir, chmpath)
                    aux2 = os.path.join(chmpath, filename.replace(config.input_image_pref, config.aux_prefs[1]))

                    auxx.append(aux2)
                    try:

                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, model_segcount, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    except IOError:
                        # continue
                        auxx[-1] = waterchm
                        nochm.append(outputFile)
                        # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, model_segcount, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width


                    ###### check threshold!!!!
                    # seg
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    # print(detectedMaskDens.shape)
                    # print(detectedMaskDens.max(), detectedMaskDens.min())
                    # print(detectedMaskDens.sum())
                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()

                    # for chm
                    with rasterio.open(aux2) as chm:
                        print('image', filename)
                        print('chm', os.path.basename(aux2))
                        if chm.read().max() > 50:
                            print('GT chm containing large values > 50m, please redo sampling')
                            # skip error files
                            continue
                        detectedChm, detectedMetaChm = detect_tree_rawtif_ndvi(config, model_chm, core, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = 0) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                        #writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)
                        if config.saveresult:
                            writeMaskToDiskChm(detectedChm, detectedMetaChm, outputFileChm, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)
                        pseg = np.squeeze(detectedMaskSeg)
                        ggt = np.squeeze(chm.read())
                        outputTreeH = outputFileChm.replace(config.output_prefix_chm, 'det_treeH_')
                        outputDiff = outputFileChm.replace(config.output_prefix_chm, 'det_treeH_diff_')
                        maskpred_res, maskgt_res = chm_seg(config, pseg, detectedChm, ggt, detectedMeta, outputTreeH, outputDiff, th = 0.5)
                        heights_pr = np.append(heights_pr, maskpred_res)
                        heights_gt = np.append(heights_gt, maskgt_res)
                        counter += 1

            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as core:
                    #print(fullPath)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(config, model_segcount, core, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()

                    # for CHM
                    detectedChm, detectedMetaChm = detect_tree_rawtif_ndvi(config, model_chm, core, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = 0) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    writeMaskToDiskChm(detectedChm, detectedMetaChm, outputFileChm, image_type = config.output_image_type, write_as_type = config.output_dtype_chm, scale = 0)


        else:
            print('File already analysed!', fullPath)
            # load predictions
            coor = fullPath[-12:-9]+ fullPath[-8:-5]
            chmpath = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
            # aux2 = fullPath.replace(config.input_image_pref, config.aux_prefs[1]).replace(config.input_image_dir, chmpath)
            aux2 = os.path.join(chmpath, filename.replace(config.input_image_pref, config.aux_prefs[1]))

            with rasterio.open(outputFile) as core:
                with rasterio.open(aux2) as chmm:
                    with rasterio.open(outputFileChm) as chmmpr:
                        pseg = np.squeeze(core.read())

                        ggt = np.squeeze(chmm.read())
                        pchm = np.squeeze(chmmpr.read())
                        maskpred_res, maskgt_res = chm_seg(config, pseg, pchm, ggt, None, None, None, th = 0.5, save = 0, scale = 1)
                        heights_pr = np.append(heights_pr, maskpred_res)
                        heights_gt = np.append(heights_gt, maskgt_res)

    CHMerror(heights_pr, heights_gt,  nbins = 100)
    return heights_pr, heights_gt, counter


def predict_segcount_chm_dk_detchm(config, all_files, model_segcount, model_chm, output_dir):
    # predict chm first and then use det chm as input to pred segcount
    counter = 1
    th = 0.5
    counts = {}
    outputFiles = []
    for fullPath, filename in tqdm(all_files):

        outputFile = os.path.join(output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
        #print(outputFile)
        outputFiles.append(outputFile)
        outputFileChm = os.path.join(output_dir, filename.replace(config.input_image_pref, config.output_prefix_chm).replace(config.input_image_type, config.output_image_type))

        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:

            if not config.single_raster and config.aux_data: # multi raster
                with rasterio.open(fullPath) as core:
                    auxx = []
                    ndvif0 = filename.replace(config.input_image_pref, config.aux_prefs[0])
                    # print(ndvif0)
                    ndvi_f = glob.glob(f"{config.input_ndvi_dir}/{ndvif0}")
                    # print(ndvi_f)
                    auxx.append(ndvi_f[0])

                    # pred chm first
                    detectedChm, detectedMetaChm = detect_tree_rawtif_ndvi(config, model_chm, core, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, maxnorm = 0) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    detectedChm[detectedChm<1]=0
                    writeMaskToDiskChm(detectedChm, detectedMetaChm, outputFileChm, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)

                    # pred segcount
                    time.sleep(2)
                    auxx.append(outputFileChm)

                    # detectedMask, detectedMeta = detect_tree([core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_detchm(config, model_segcount, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()



            else: # single raster or multi raster without aux
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as img:
                    #print(fullPath)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
                    # writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()
        else:
            # print('File already analysed!', fullPath)
            fd = outputFile.replace('det', 'density')
            with rasterio.open(fd) as dens:
                dd = dens.read().sum()
                counts[filename] = dd



    print('Inference for the area has finished, saving counts to csv file')
    w = csv.writer(open(os.path.join(config.output_dir, "counts.csv"), "w"))
    for key, val in counts.items():
        w.writerow([key, val])
    return counter




def predict_segcount_chm_fi(config, all_files, model_segcount, model_chm, output_dir, eva = 0):
    counter = 1
    th = 0.5
    counts = {}
    if eva:
        heights_gt = np.array([])
        heights_pr = np.array([])
    outputFiles = []
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        if not os.path.exists(outputFile):
            # print(outputFile)
            outputFiles.append(outputFile)
            outputFileChm = os.path.join(output_dir, config.output_prefix_chm + filename.replace(config.input_image_type, config.output_image_type))
            try:
                with rasterio.open(fullPath) as img:
                    # for only south tifs
                    # print(raw.profile['transform'][5])
                    if img.profile['transform'][5] > 7050000:


                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(config, model_segcount, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize,
                                                                                                    auxData = config.aux_data, singleRaster=config.single_raster, multires = config.multires, upsample = config.upsample, downsave = config.downsave)
                        writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                        # density
                        writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('det', 'density'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                        counts[filename] = detectedMaskDens.sum()

                        if config.chmpred:
                            if config.upsample and config.downsave:
                                detectedMaskChm, detectedMetaChm = detect_tree_rawtif_fi(config, model_chm, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                            else:
                                detectedMaskChm, detectedMetaChm = detect_tree_rawtif_fi(config, model_chm, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                            writeMaskToDiskChm(detectedMaskChm, detectedMetaChm, outputFileChm, image_type = config.output_image_type, write_as_type = config.output_dtype_chm)

                            if eva:
                                chmPath = config.gt_chm_dir  + config.chm_pref + filename[:-4] + config.chm_sufix
                                # print(chmPath)
                                with rasterio.open(chmPath) as chm:
                                    pseg = np.squeeze(detectedMaskSeg)
                                    ggt = np.squeeze(chm.read())
                        else:
                            continue

                    else:
                        print('Skipping: NOT in south!', fullPath)

            except:
                print('file invalid')
            counter += 1


        else:
            print('Skipping: File already analysed!', fullPath)
    if eva:
        MAE = mean_absolute_error(heights_gt, heights_pr)
        print('MAE tree level', MAE)
        CHMerror(heights_pr, heights_gt,  nbins = 100)

        return heights_pr, heights_gt, counter
    else:
        return counter



def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get.
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    return (res)

def addTOResult_chm(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    # plt.figure()
    # plt.imshow(np.squeeze(currValue))
    newPredictions = prediction[:he, :wi]
    # plt.figure()
    # plt.imshow(np.squeeze(newPredictions))
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    elif operator == "MIX": # alpha blending # note do not combine with empty regions
        # print('mix')
        mm1 = currValue!=0
        try:
            currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
            mm2 = (currValue==0)
            currValue[mm2] = newPredictions[mm2]
            resultant = currValue
        except:
            resultant = newPredictions[:256, :256]

    else: #operator == 'REPLACE':
        resultant = newPredictions
    res[row:row+he, col:col+wi] =  resultant
    return (res)


# 2 tasks
def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        # print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        seg, dens = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        seg, dens = model.predict(tm1) # tm is a list []

    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens


def predict_using_model_segcount_fi(model, batch, batch_pos, maskseg, maskdens, operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        # print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        seg, dens = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        seg, dens = model.predict(tm1) # tm is a list []

    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)

        if upsample and downsave:   # no preserve range
            # print('**********************UPsample for prediction***************************')
            # print('**********************Downsample for saving***********************')
            # downsample the predictions (upsampled to predict, but should be resized to put in the big window)
            # print(upscale)
            p = resize(p[:, :], (int(p.shape[0]/upscale), int(p.shape[1]/upscale)), preserve_range=True)
            c = resize(c[:, :], (int(c.shape[0]/upscale), int(c.shape[1]/upscale)), preserve_range=True)
            if rescale_values:
                # rescale values
                p = p * (p.shape[0] / float(p.shape[0]/upscale)) * (p.shape[1] / float(p.shape[1]/upscale))
                c = c * (c.shape[0] / float(c.shape[0]/upscale)) * (c.shape[1] / float(c.shape[1]/upscale))

        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens



def predict_using_model_chm(model, batch, batch_pos, mask, operator, upsample = 1):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200

    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        prediction = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        prediction = model.predict(tm1) # tm is a list []
        # print(prediction.mean())

    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        mask = addTOResult_chm(mask, p, row, col, he, wi, operator)
    return mask




def predict_using_model_chm_fi(model, batch, batch_pos, mask, operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200

    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
        print('2 inputs')
        tm1 = []
        tm2 = []
        for p in batch:
            tm1.append(p[0]) # (256, 256, 5)
            tm2.append(p[1]) # (128, 128, 1)
        tm1 = np.stack(tm1, axis = 0) #stack a list of arrays along axis
        tm2 = np.stack(tm2, axis = 0)
        prediction = model.predict([tm1, tm2]) # tm is a list []

    else:
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        # print(tm1.mean(axis = (0, 1, 2)))
        prediction = model.predict(tm1) # tm is a list []
        # print('pred', prediction.min(), prediction.max())
        # print(prediction.mean())
        # print('pred from model', prediction.shape)
        
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # if upsample and downsave:   # no preserve range for chm model, as already a resolution reduce
            
        #     p = resize(p[:, :], (int(p.shape[0]/upscale), int(p.shape[1]/upscale)), preserve_range=True)
        #     if rescale_values:
        #         # rescale values
        #         p = p * (p.shape[0] / float(p.shape[0]/upscale)) * (p.shape[1] / float(p.shape[1]/upscale))
                
        # print('before add tore ', p.shape)
        mask = addTOResult_chm(mask, p, row, col, he, wi, operator)
    
    return mask


# multi-task: segcount
# allow multiple models predicting together
def detect_tree_segcount(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    """img can be one single raster or multi rasters

    img = [core_img, aux1, aux2, ...]

    or

    img = img #single raster
    """
    if not singleRaster and auxData: # multi raster: core img and aux img in a list

        core_img = img[0]

        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()

        aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input

    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    masks_seg = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masks_dens = np.zeros((len(models), nrows, nols), dtype=np.float32)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)


        if not singleRaster and auxData: #multi rasters with possibly different resolutions resampled to the same resolution
            nc1 = meta['count'] + aux_channels1 #4+1 = 5 channels (normal input)
            # print('number of input channels:', nc)
            patch1 = np.zeros((height, width, nc1)) # 256, 256, 5
            temp_im1 = core_img.read(window=window)
            # print('col row', col_off, row_off)
            # print('core shape', core_sm.shape)
            # print('coloff', col_off)
            # print('rowoff', row_off)

            for aux in range(aux_channels1): # 0, 1 for two channels; 0 for one channel

                # print('Dealing with aux data', aux)
                aux_imgi = rasterio.open(img[aux+1]) #ndvi layer
                meta_auxi = aux_imgi.meta.copy()
                hei_ratio = nrows/meta_auxi['height']
                wid_ratio = nols/meta_auxi['width']
                # print('Handle resolution differences: resampling aux layer with a factor of', (hei_ratio, wid_ratio))

                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)

                aux_sm1 = aux_imgi.read(
                                out_shape=(
                                aux_imgi.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                temp_im1 = np.row_stack((temp_im1, aux_sm1))

            # for the 2nd input source
            patch2 = np.zeros((int(height/2), int(width/2), 1)) # 128, 128, 1
            temp_img2 = rasterio.open(img[-1]) #chm layer
            window2 = windows.Window(window.col_off / 2, window.row_off / 2,
                                    int(window.width/2), int(window.height/2))

            temp_im2 = temp_img2.read(
                                out_shape=(
                                temp_img2.count,
                                int(window2.height),
                                int(window2.width)
                            ),
                            resampling=Resampling.bilinear, window = window2)
        else: #single rater or multi raster without aux # must be multi raster
            patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
            temp_im1 = img.read(window=window)


        ##########################################################################################
        #suqeeze or not? for one channel should not squeeze
        # temp_im = np.squeeze(temp_im)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        if not singleRaster and auxData:
            temp_im2 = np.transpose(temp_im2, axes=(1,2,0))

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            if not singleRaster and auxData:
                temp_im2 = image_normalize(temp_im2, axis=(0,1))

        
        if not singleRaster and auxData:
            patch1[:window.height, :window.width] = temp_im1
            patch2[:window2.height, :window2.width] = temp_im2
            batch.append([patch1, patch2])
        else:
            patch[:window.height, :window.width] = temp_im1
            batch.append(patch)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask_seg = masks_seg[mi, :, :]
                curmask_dens = masks_dens[mi, :, :]
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')

            batch = []
            batch_pos = []
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask_seg = masks_seg[mi, :, :]
            curmask_dens = masks_dens[mi, :, :]
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)


def detect_tree_segcount_rawtif_ndvi(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    # updated large scale dk segcount prediction, using 2020 photos
    if not singleRaster and auxData: # multi raster: core img and aux img in a list

        core_img = img[0]

        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()

        aux_channels1 = len(img) - 1 # aux channels with same resolution as color input

    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    masks_seg = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masks_dens = np.zeros((len(models), nrows, nols), dtype=np.float32)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)


        if not singleRaster and auxData: #multi rasters with possibly different resolutions resampled to the same resolution
            nc1 = meta['count'] + aux_channels1 #4+1 = 5 channels (normal input)
            # print('number of input channels:', nc)
            patch1 = np.zeros((height, width, nc1)) # 256, 256, 5
            temp_im1 = core_img.read(window=window)

            # compute ndvi here
            temp_im1 = np.transpose(temp_im1, axes=(1,2,0)) # channel last

            NDVI = (temp_im1[:, :, -1].astype(float) - temp_im1[:, :, 0].astype(float)) / (temp_im1[:, :, -1].astype(float) + temp_im1[:, :, 0].astype(float))
            NDVI = NDVI[..., np.newaxis]
            # print('NDVI', NDVI.max(), NDVI.min())
            # print('bands', temp_im.max(), temp_im.min())
            temp_im1 = np.append(temp_im1, NDVI, axis = -1)

            # temp_im1 = np.row_stack((temp_im1, aux_sm1))
            # print(temp_im1.shape)
            # for the 2nd input source
            patch2 = np.zeros((int(height/2), int(width/2), 1)) # 128, 128, 1
            # print(img[-1])
            temp_img2 = rasterio.open(img[-1]) #chm layer
            # print(temp_img2)
            window2 = windows.Window(window.col_off / 2, window.row_off / 2,
                                    int(window.width/2), int(window.height/2))
            # print(temp_img2.count, int(window2.height), int(window2.width))
            temp_im2 = temp_img2.read(
                                out_shape=(
                                temp_img2.count,
                                int(window2.height),
                                int(window2.width)
                            ),
                            resampling=Resampling.bilinear, window = window2)
        else: #single rater or multi raster without aux # must be multi raster
            patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
            temp_im1 = img.read(window=window)


        ##########################################################################################
        #suqeeze or not? for one channel should not squeeze
        # temp_im = np.squeeze(temp_im)
        if config.input_ndvi_separate:
            temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
        temp_im2 = np.transpose(temp_im2, axes=(1,2,0))
        
        
        
        
        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            temp_im2 = image_normalize(temp_im2, axis=(0,1))

        
        patch1[:window.height, :window.width] = temp_im1
        patch2[:window2.height, :window2.width] = temp_im2
        batch.append([patch1, patch2])
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask_seg = masks_seg[mi, :, :]
                curmask_dens = masks_dens[mi, :, :]
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')

            batch = []
            batch_pos = []
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask_seg = masks_seg[mi, :, :]
            curmask_dens = masks_dens[mi, :, :]
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)

def detect_tree_segcount_detchm(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):

    if not singleRaster and auxData: # multi raster: core img and aux img in a list

        core_img = img[0]

        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()

        aux_channels1 = len(img) - 1 # aux channels with same resolution as color input

    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    masks_seg = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masks_dens = np.zeros((len(models), nrows, nols), dtype=np.float32)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)


        if not singleRaster and auxData: #multi rasters with possibly different resolutions resampled to the same resolution
            nc1 = meta['count'] + aux_channels1 #4+1 = 5 channels (normal input)
            # print('number of input channels:', nc)
            patch1 = np.zeros((height, width, nc1)) # 256, 256, 5
            temp_im1 = core_img.read(window=window)
            # print('col row', col_off, row_off)
            # print('core shape', core_sm.shape)
            # print('coloff', col_off)
            # print('rowoff', row_off)

            for aux in range(aux_channels1): # 0, 1 for two channels; 0 for one channel

                # print('Dealing with aux data', aux)
                aux_imgi = rasterio.open(img[aux+1]) #ndvi layer
                meta_auxi = aux_imgi.meta.copy()
                hei_ratio = nrows/meta_auxi['height']
                wid_ratio = nols/meta_auxi['width']
                # print('Handle resolution differences: resampling aux layer with a factor of', (hei_ratio, wid_ratio))

                res_window = windows.Window(window.col_off / wid_ratio, window.row_off / hei_ratio,
                                window.width / wid_ratio, window.height / hei_ratio)

                aux_sm1 = aux_imgi.read(
                                out_shape=(
                                aux_imgi.count,
                                int(window.height),
                                int(window.width)
                            ),
                            resampling=Resampling.bilinear, window = res_window)
                # print('aux shape', aux_sm.shape)
                if aux == 1:
                    aux_sm1 = aux_sm1/100
                temp_im1 = np.row_stack((temp_im1, aux_sm1))

        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel


        patch1[:window.height, :window.width] = temp_im1

        batch.append(patch1)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmask_seg = masks_seg[mi, :, :]
                curmask_dens = masks_dens[mi, :, :]
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')

            batch = []
            batch_pos = []
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmask_seg = masks_seg[mi, :, :]
            curmask_dens = masks_dens[mi, :, :]
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)


def detect_tree_rawtif_ndvi(config, model, img, channels, width=256, height=256, stride = 128, normalize=0, maxnorm = 0):
    """img can be one single raster or multi rasters

    img = [core_img, aux1, aux2, ...]

    or

    img = img #single raster
    """

    #single rater or multi raster without aux
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])
    mask = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)


        #single rater or multi raster without aux # must be multi raster
        patch = np.zeros((height, width, len(channels)+1)) #Add zero padding in case of corner images
        read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels]
        # print(temp_im.shape)
        # print('red', temp_im[:, :, -1].max(), temp_im[:, :, -1].min())
        # print('red2', temp_im[:, :, 0].max(), temp_im[:, :, 0].min())
        # print(temp_im[:, :, -1] + temp_im[:, :, 0])
        # print(temp_im[:, :, -1] - temp_im[:, :, 0])
        NDVI = (temp_im[:, :, -1].astype(float) - temp_im[:, :, 0].astype(float)) / (temp_im[:, :, -1].astype(float) + temp_im[:, :, 0].astype(float))
        NDVI = NDVI[..., np.newaxis]
        # print('NDVI', NDVI.max(), NDVI.min())
        # print('bands', temp_im.max(), temp_im.min())
        temp_im = np.append(temp_im, NDVI, axis = -1)
        temp_im = temp_im / 255
        # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
        temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])
        # do global normalize to avoid blocky..
        # if normalize:
        # temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel

        patch[:window.height, :window.width] = temp_im
        batch.append(patch)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model_chm(model, batch, batch_pos, mask, 'MIX')
            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model_chm(model, batch, batch_pos, mask, 'MIX')
        batch = []
        batch_pos = []

    if maxnorm:
        mask = mask * 97.19
    return(mask, meta)


def detect_tree_segcount_fi(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1, multires = 1, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1, inf = 1, rgb2gray = 0):
    # for fi 3 bands
    if 'chm' in config.channel_names1:
        CHM = 1
    else:
        CHM = 0

    if not singleRaster and auxData: # multi raster: core img and aux img in a list

        core_img = img[0]

        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()

        # aux_channels1 = len(img) - 1 - 1 # aux channels with same resolution as color input

    else: #single rater or multi raster without aux
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()
    # img0 = img[0] # channel 0

    # read_img0 = img0.read()
    # print(read_img0.shape)
    # nrows, nols = read_img0.shape[1:] # base shape # rasterio read channel first
    # nrows, nols = 256, 256 # base shape # rasterio read channel first

    # meta = img0.meta.copy()

    # tile normalize:
    if config.segcount_tilenorm:
        print('tile norm')
        temp_imm = img.read()
        temp_imm = np.transpose(temp_imm, axes=(1,2,0))
        means = np.mean(temp_imm, axis = (0, 1))
        stds = np.std(temp_imm, axis = (0, 1))


    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    if downsave or not upsample:
        masksegs = np.zeros((len(models), nrows, nols), dtype=np.float32)
        maskdenss = np.zeros((len(models), nrows, nols), dtype=np.float32)

    elif not downsave:
        masksegs = np.zeros((len(models), int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        maskdenss = np.zeros((len(models), int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        meta.update(
                    {'width': int(nols*upscale),
                     'height': int(nrows*upscale)
                    }
                    )

    if rgb2gray:
        meta.update(
                    {'count': 1,
                    }
                    )

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):

        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)

        # temp_im1 = img[0].read(window=window)

        if CHM:
            print('including CHM as input')
            ## TODO


        elif not CHM:
            if upsample:
                # patch1 = np.zeros((height, width, len(img)))
                if inf:
                    patch1 = np.zeros((int(height*upscale), int(width*upscale), meta['count']))
                    if config.band_switch:
                        patch1 = np.zeros((int(height*upscale), int(width*upscale), int(len(config.new_band_order))))

                elif not inf:
                    patch1 = np.zeros((int(height*upscale), int(width*upscale), meta['count']-1))

                temp_im1 = img.read(
                                    out_shape=(
                                    img.count,
                                    int(window.height*upscale), # upsample by 2
                                    int(window.width*upscale)
                                ),
                                resampling=Resampling.bilinear, window = window)
            else:
                # no upsample
                if inf:
                    patch1 = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
                    if config.band_switch:
                        patch1 = np.zeros((height, width, int(len(config.new_band_order))))

                elif not inf:
                    patch1 = np.zeros((height, width, meta['count']-1))

                # temp_im = img.read(window=window)
                temp_im1 = img.read(window = window)

        # stack makes channel first
        # print('rstack tt shape', temp_im1.shape)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))

        if not inf:
            temp_im1 = temp_im1[:,:,:-1]
        # print(temp_im1.shape, temp_im2.shape)

        if config.band_switch: # using only subset of the bands
            temp_im1 = temp_im1[:,:,config.new_band_order]


        if rgb2gray:
            temp_im1 = rgb2gray_convert(temp_im1)[..., np.newaxis]


        if config.segcount_tilenorm:
            temp_im1 = (temp_im1-means)/stds

        if normalize:
            # rescale FI data first
            # print('rescaling', temp_im1.mean())
            # temp_im1 = np.array([139, 0, 0]) + ((temp_im1 - mins) / (maxs - mins)) * np.array([255-139, 255, 255])
            # print('rescaling 2', temp_im1.mean())
            # print('patch norm')
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            # print('norm', temp_im1.mean())
            if CHM and multires:
                print('with CHM')
                ##TODO
                # temp_im2 = image_normalize(temp_im2, axis=(0,1))

        if upsample:
            patch1[:int(window.height*upscale), :int(window.width*upscale)] = temp_im1
        else:

            patch1[:int(window.height), :int(window.width)] = temp_im1

        # if CHM and multires:
        #     patch2[:window2.height, :window2.width] = temp_im2
        #     batch.append([patch1, patch2])
        if CHM:
            print('with CHM')


        elif not CHM:
            batch.append(patch1)
        # print('window colrow wi he', window.col_off, window.row_off, window.width, window.height)
        # batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if downsave or not upsample:
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        elif not downsave:
            # print('upsave')
            batch_pos.append((int(window.col_off*upscale), int(window.row_off*upscale), int(window.width*upscale), int(window.height*upscale)))
        # plt.figure(figsize = (10,10))
        # plt.imshow(temp_im1[:,:,0])
        if (len(batch) == config.BATCH_SIZE):
            # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            # batch = []
            # batch_pos = []
            for mi in range(len(models)):
                curmaskseg = masksegs[mi, :, :]
                curmaskdens = maskdenss[mi, :, :]

                curmaskseg, curmaskdens = predict_using_model_segcount_fi(models[mi], batch, batch_pos, curmaskseg, curmaskdens, 'MAX', upsample = upsample, downsave = downsave, upscale = upscale, rescale_values=rescale_values)

            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        # mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
        # batch = []
        # batch_pos = []
        for mi in range(len(models)):
            curmaskseg = masksegs[mi, :, :]
            curmaskdens = maskdenss[mi, :, :]

            curmaskseg, curmaskdens = predict_using_model_segcount_fi(models[mi], batch, batch_pos, curmaskseg, curmaskdens, 'MAX', upsample = upsample, downsave = downsave, upscale = upscale, rescale_values=rescale_values)

        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta


def detect_tree_rawtif_fi(config, model, img, channels,  width=256, height=256, stride = 128, normalize=0, maxnorm = 0, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1):
    # for FI, upsample input images to deal with resolution diff

    #single rater or multi raster without aux
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    # print(offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    # mask = np.zeros((nrows, nols), dtype=meta['dtype'])

    if downsave and upsample:
        mask = np.zeros((int(nrows), int(nols)), dtype=meta['dtype'])

    # if for denmark-like resolution
    else:
        mask = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])
        # print('HERE!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(mask.shape)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        # transform = windows.transform(window, core_img.transform)
        if upsample:
            # patch1 = np.zeros((height, width, len(img)))
            if config.addndvi:
                patch = np.zeros((height*2, width*2, len(channels)+1))
            else:
                patch = np.zeros((height*2, width*2, len(channels)))

            read_im = img.read(
                                out_shape=(
                                img.count,
                                int(window.height*2), # upsample by 2
                                int(window.width*2)
                            ),
                            resampling=Resampling.bilinear, window = window)
        else:
            # no upsample
            if config.addndvi:
                patch = np.zeros((height, width, len(channels)+1)) #Add zero padding in case of corner images
            else:
                patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images

            read_im = img.read(window = window)
        # #single rater or multi raster without aux # must be multi raster
        # patch = np.zeros((height, width, len(channels))) #Add zero padding in case of corner images
        # read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0)) # channel last
        temp_im = read_im[:,:,channels] # swap channels, inf last
        # print(temp_im.mean(axis = (0,1)))

        # print(temp_im.mean(axis = (0, 1)))
        # print('size', patch.shape)
        if config.addndvi:
            NDVI = (temp_im[:, :, -1].astype(float) - temp_im[:, :, 0].astype(float)) / (temp_im[:, :, -1].astype(float) + temp_im[:, :, 0].astype(float))
            NDVI = NDVI[..., np.newaxis]
            # print('NDVI', NDVI.max(), NDVI.min())
            # print('bands', temp_im.max(), temp_im.min())
            temp_im = np.append(temp_im, NDVI, axis = -1)


        # print('max rescaling', temp_im.max)
        # do global normalize to avoid blocky..
        # if normalize:
        #     temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
        # if len(config.channels) > 3:
        if len(channels) > 3:
            # print('channel > 3')
            if config.gbnorm: # all bands
                logging.info('all bands - standarization - gb')
                temp_im = temp_im / 255
                temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]]))/ np.array([[0.985, 0.895, 0.703, 1.107, 1]])

                # temp_im = (temp_im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
            elif config.robustscale: # all bands robust scale
                logging.info('incomplete')
                # todo: robust scale for all bands
                # temp_im = temp_im / 255
                # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                if normalize:
                    temp_im = image_normalize(temp_im, axis=(0,1))

        # elif len(config.channels) == 3: # 3 bands
        elif len(channels) == 3: # 3 bands, by default this is for FI so only rg+NIR bands
            if channels[0] == 1: # to confirm: this is the case for FI, first band is green
                if config.robustscale: # DK rebust scale
                    logging.info('3 bands - robust scale - DK - gb')
                    # /179 * 255
                    # # only for inf
                    # temp_im = np.array([0, 0, 29]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-29])
                    # q01 and q99 from DK training data
                    # temp_im = np.array([38, 49, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([167-38, 151-49, 212-62])
                    # # q001 q999
                    # temp_im = np.array([29, 41, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([191-29, 175-41, 225-39])
                    # # q001 q999 v2
                    # temp_im = np.array([29, 41, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255-29, 255-41, 255-39])
                    # # only for inf , q01, q99
                    # temp_im = np.array([0, 0, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 212-62])
                    # # only for inf , q01, q99
                    # temp_im = np.array([0, 0, 62]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-62])
                    # # only for inf , q1, max
                    # temp_im = np.array([0, 0, 119]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-119])
                    # # only for inf , q15, max
                    # temp_im = np.array([0, 0, 130]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-130])

                    # # only for inf , q25, max
                    # temp_im = np.array([0, 0, 145]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-145])
                    # only for inf , q20, max
                    temp_im = np.array([0, 0, 139]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-139])

                    # # # q001 q999 inf
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 225-39])
                    # # q001 max inf
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-39])

                    temp_im = (temp_im - np.array([[73.0, 72.0, 145.0]]))/ np.array([[113.0-73.0, 96.0-72.0, 182.0-145.0]])


                if config.robustscaleFI_gb: # 3 bands FI self stats gbly
                    logging.info('3 bands - robust scale - FI - gb')
                    # q1s = np.array([46, 65, 21])
                    # q3s = np.array([ 99, 106, 130])
                    temp_im = (temp_im - q1s)/ (q3s - q1s)

                if config.robustscaleFI_local: # 3 bands FI self stats locally
                    logging.info('3 bands - robust scale - FI - local')
                    q1s = np.quantile(temp_im, 0.25, axis = (0,1))
                    q3s = np.quantile(temp_im, 0.75, axis = (0,1))
                    assert len(q1s) == 3 and len(q3s) == 3
                    temp_im = (temp_im - q1s)/ (q3s - q1s)

                if config.gbnorm: # 3 bands DK gb norm from training set
                    # only for inf , q20, max
                    # temp_im = np.array([0, 0, 39]) + ((temp_im - mins) / (maxs - mins)) * np.array([255, 255, 255-39])

                    logging.info('3 bands - gb norm - DK')
                    # temp_im = temp_im / 255
                    temp_im = temp_im / 255 # int16 instead of int8
                    # temp_im = (temp_im - np.array([[0.37, 0.34, 0.63]]))/ np.array([[0.74, 0.52, 0.81]])
                    temp_im = (temp_im - np.array([[0.350, 0.321, 0.560]]))/ np.array([[0.895, 0.703, 1.107]])

                if config.gbnorm_FI:
                    # normalize globally gb inf from FI training data
                    logging.info('3 bands - gb norm - FI training data')
                    temp_im = temp_im / 255
                    # temp_im = (temp_im - np.array([[0.300, 0.338, 0.343]]))/ np.array([[0.168, 0.153, 0.146]])
                    temp_im = (temp_im - np.array([[0.253, 0.300, 0.321]]))/ np.array([[0.122, 0.118, 0.127]])

                if config.localtifnorm:
                    logging.info('3 bands - local tif norm - DK')
                    temp_im = temp_im / 255
                    temp_im = image_normalize(temp_im, axis=(0,1))
            elif channels[0]==0: # this is the case for RGB trained model
                if config.gbnorm: # 3 bands DK gb norm from training set
                    # print('3 bands - gb norm - DK')
                    temp_im = temp_im / 255
                    temp_im = (temp_im - np.array([[0.317, 0.350, 0.321]]))/ np.array([[0.985, 0.895, 0.703]])
                    # print(temp_im.mean(axis = (0, 1)))
                    # print(temp_im.std(axis = (0,1)))

        # print('read', temp_im.shape)
        if upsample:
            patch[:int(window.height*2), :int(window.width*2)] = temp_im
        else:
            patch[:int(window.height), :int(window.width)] = temp_im
        # patch[:window.height, :window.width] = temp_im
        batch.append(patch)

        # this only applies to coraser resolution
        # if downsave or not upsample:
        #     batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        # elif not downsave:
        #     print('upsave-----------------')
        #     batch_pos.append((int(window.col_off*2), int(window.row_off*2), int(window.width*2), int(window.height*2)))

        if upsample:
            batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        elif not upsample: # dk-like chm prediction
            batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        
        # batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if (len(batch) == config.BATCH_SIZE):
            mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1)
            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample = 1, downsave = 1, upscale = 2, rescale_values = 1)
        batch = []
        batch_pos = []

    if maxnorm:
        mask = mask * 97.19

    return(mask, meta)



def predict_segcount_cn(config, all_files, model_segcount, model_chm, output_dir, eva = 0, inf = 1, th = 0.5, rgb2gray = 0):
    counter = 1
    th = th
    counts = {}
    if eva:
        heights_gt = np.array([])
        heights_pr = np.array([])
    outputFiles = []
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        print(outputFile)
        outputFile2 = outputFile.replace('seg.tif', 'chm.tif')
        
        if not os.path.exists(outputFile) or not os.path.exists(outputFile2):
            # print(outputFile)
            outputFiles.append(outputFile)
            # outputFileChm = os.path.join(output_dir, filename.replace(config.input_image_type, config.output_image_type))

            with rasterio.open(fullPath) as img:
                # for only south tifs
                # print(raw.profile['transform'][5])
                if config.segcountpred:
                    print('creating file', outputFile)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(config, model_segcount, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize,
                                                                                                auxData = config.aux_data, singleRaster=config.single_raster, multires = config.multires, upsample = config.upsample, downsave = config.downsave, upscale = config.upscale, rescale_values=config.rescale_values, inf = inf, rgb2gray = rgb2gray)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)
                    # writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
    
                    # density
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg.tif', 'density.tif'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                    counts[filename] = detectedMaskDens.sum()

                if config.chmpred:
                    print('creating file', outputFile2)
                    detectedMaskChm, detectedMetaChm = detect_tree_rawtif_fi(config, model_chm, img, config.channels, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize, maxnorm = config.maxnorm, upsample = config.upsample, downsave = config.downsave, upscale = config.upscale, rescale_values = config.rescale_values) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width

                    writeMaskToDiskChm(detectedMaskChm, detectedMetaChm, outputFile.replace('seg.tif', 'chm.tif'), image_type = config.output_image_type, write_as_type = config.output_dtype_chm, scale = 0)

                    if eva:
                        chmPath = config.gt_chm_dir  + config.chm_pref + filename[:-4] + config.chm_sufix
                        # print(chmPath)
                        with rasterio.open(chmPath) as chm:
                            pseg = np.squeeze(detectedMaskSeg)
                            ggt = np.squeeze(chm.read())
                else:
                    continue


            counter += 1


        else:
            print('Skipping: File already analysed!', fullPath)
    if eva:
        MAE = mean_absolute_error(heights_gt, heights_pr)
        print('MAE tree level', MAE)
        CHMerror(heights_pr, heights_gt,  nbins = 100)

        return heights_pr, heights_gt, counter
    else:
        return counter





############################################################################################################
############################################################################################################
def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result

def pooling(mat,ksize,method='max',pad=False):
    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result


def chm_seg(config, segmask, chmpred, chmgt, meta, outputFile, outputDiff, th = 0.5, save = 1, scale = 0):
    # mask height with seg masks
    # # return list of dicts
    # stat = zonal_stats("polygons.shp", "elevation.tif",
    #         stats="count min mean max median")
    segmask[segmask<th]=0
    segmask[segmask>=th]=1

    if scale:
        # rescale chm values pred
        chmpred = chmpred/100
    try:
        segmask2 = pooling(segmask,(2,2),method='mean',pad=1)

        mask1 = segmask2 * chmpred
    except:
        # chmpred saved to higher reso
        segmask2 = pooling(segmask,(2,2),method='mean',pad=1)
        chmpred = pooling(chmpred,(2,2),method='mean',pad=1)

        mask1 = segmask2 * chmpred
    # print(segmask.shape)
    # mask2 = poolingOverlap(mask1,(3,3),(2,2),method='max',pad=1)
    # mask2 = poolingOverlap(mask1,(5,5),(2,2),method='max',pad=1) # 1m pixel resolution
    # mask2 = poolingOverlap(mask1,(10,10),(4,4),method='max',pad=1) # 1m pixel resolution
    # mask2 = poolingOverlap(mask1,(20,20),(2,2),method='max',pad=1) # 1m pixel resolution
    mask2 = poolingOverlap(mask1,(15,15),(2,2),method='max',pad=1) # m pixel resolution
    # print(mask2.shape)
    segmask3 = pooling(segmask2,(2,2),method='mean',pad=1)
    # segmask2 = poolingOverlap(segmask,(4,4),(4,4),method='mean',pad=1)
    maskpred_res = mask2*segmask3
    # # for computing errors
    # maskpred_resP = poolingOverlap(maskpred_res,(2,2),(2,2),method='mean',pad=1)
    # maskpred_resP = poolingOverlap(maskpred_res,(2,2),(2,2),method='max',pad=1)
    # print(maskpred_res.shape)

    # print(chmgt.shape)
    maskgt1 = segmask2 * chmgt
    # maskgt2 = poolingOverlap(maskgt1,(10,10),(2,2),method='max',pad=1) # 1m pixel resolution
    maskgt2 = poolingOverlap(maskgt1,(15,15),(2,2),method='max',pad=1) # m pixel resolution
    # segmask3 = poolingOverlap(segmask2,(2,2),(2,2),method='mean',pad=1)
    maskgt_res = maskgt2*segmask3

    h = tf.keras.losses.Huber()
    huberLoss = h(maskgt_res, maskpred_res).numpy()
    mae = tf.keras.losses.MeanAbsoluteError()
    MaeLoss = mae(maskgt_res, maskpred_res).numpy()
    mse = tf.keras.losses.MeanSquaredError()
    MseLoss = mse(maskgt_res, maskpred_res).numpy()

    print('*******************Huber loss: {}******************'.format(huberLoss))
    print('*******************MSE loss: {}******************'.format(MseLoss))
    print('*******************MAE loss: {}******************'.format(MaeLoss))
    if save:
        writeMaskToDisk(maskpred_res, meta, outputFile, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)

    diff = maskgt_res - maskpred_res # m pixel resolution
    if save:
        writeMaskToDisk(diff, meta, outputDiff, image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)

    # errors
    # CHMerror(maskpred_res, maskgt_res,  nbins = 100)


    return maskpred_res, maskgt_res


def create_vector_vrt(vrt_out_fp, layer_fps, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file.
    Concatenates several vector files in a single VRT file with OGRVRTUnionLayers.
    Layer file paths are stored as relative paths, to allow copying of the VRT file with all its data layers.
    """
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty VRT file, skipping: {vrt_out_fp}")

    xml = f'<OGRVRTDataSource>\n' \
          f'    <OGRVRTUnionLayer name="{out_layer_name}">\n'
    for layer_fp in tqdm(layer_fps, desc="Creating VRT", disable=not pbar):
        shapefile = ogr.Open(layer_fp)
        layer = shapefile.GetLayer()
        relative_path = layer_fp.replace(f"{os.path.join(os.path.dirname(vrt_out_fp), '')}", "")
        xml += f'        <OGRVRTLayer name="{os.path.basename(layer_fp).split(".")[0]}">\n' \
               f'            <SrcDataSource relativeToVRT="1">{relative_path}</SrcDataSource>\n' \
               f'            <SrcLayer>{layer.GetName()}</SrcLayer>\n' \
               f'            <GeometryType>wkb{ogr.GeometryTypeToName(layer.GetGeomType())}</GeometryType>\n' \
               f'        </OGRVRTLayer>\n'
    xml += '    </OGRVRTUnionLayer>\n' \
           '</OGRVRTDataSource>\n'
    with open(vrt_out_fp, "w") as file:
        file.write(xml)

def create_vector_gpkg(out_fp, layer_fps, crs, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file.
    Concatenates several vector files in a single VRT file with OGRVRTUnionLayers.
    Layer file paths are stored as relative paths, to allow copying of the VRT file with all its data layers.
    """
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty file, skipping: {layer_fps}")

    dfall = gps.GeoDataFrame()
    for layer_fp in tqdm(layer_fps, desc="Creating GPKG", disable=not pbar):
        # shapefile = ogr.Open(layer_fp)
        # layer = shapefile.GetLayer()
        df = gps.read_file(layer_fp, layer='trees')
        dfall = dfall.append(df)

    # with open(out_fp, "w") as file:
    #     file.write(dfall)
    dfall.to_file(out_fp, driver="GPKG", crs = crs, layer="trees")
    return

def polygonize_chunk(params):
    """Polygonize a single window chunk of the raster image."""

    # raster_fp, raster_chm, out_fp, window = params
    raster_fp, out_fp, window = params

    polygons = []
    with rasterio.open(raster_fp) as src:
        raster_crs = src.crs
        for feature, _ in shapes(src.read(window=window), src.read(window=window), 4,
                                 src.window_transform(window)):
            polygons.append(shape(feature))
    if len(polygons) > 0:
        df = gps.GeoDataFrame({"geometry": polygons})
        df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="trees")

        # # read chm raster window
        # with rasterio.open(raster_chm) as src2:
        #     data = src2.read(window=window)
        #     print(data.shape)
        #     profile = src2.profile
        #     profile.height = data.shape[2]
        #     profile.width = data.shape[1]
        #     profile.transform = src2.window_transform(window)
        #     with rasterio.open("/vsimem/temp.tif", "w", **profile) as dst:
        #         dst.write(data)


        #     heights = zonal_stats(out_fp, "/vsimem/temp.tif",
        #         stats="max")
        #     df["max_height"] = heights
        #     print(df.head())
        #     df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="heights")

        return out_fp

# def polygonize_chunk_heights(params):
#     """Polygonize a single window chunk of the raster image."""

#     raster_fp, raster_chm, out_fp, window = params
#     polygons = []
#     with rasterio.open(raster_fp) as src:
#         raster_crs = src.crs
#         for feature, _ in shapes(src.read(window=window), src.read(window=window), 4,
#                                  src.window_transform(window)):
#             polygons.append(shape(feature))
#     if len(polygons) > 0:
#         df = gps.GeoDataFrame({"geometry": polygons})
#         df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="trees")

#         # read chm raster window
#         with rasterio.open(raster_chm) as src2:
#             data = src2.read(window=window)
#             print(data.shape)
#             profile = src2.profile
#             profile.height = data.shape[2]
#             profile.width = data.shape[1]
#             profile.transform = src2.window_transform(window)
#             with rasterio.open("/vsimem/temp.tif", "w", **profile) as dst:
#                 dst.write(data)


#             heights = zonal_stats(out_fp, "/vsimem/temp.tif",
#                 stats="max")
#             df["max_height"] = heights
#             print(df.head())
#             df.to_file(out_fp, driver="GPKG", crs=raster_crs, layer="heights")

#         return out_fp

def create_polygons(raster_dir, polygons_basedir, postprocessing_dir, postproc_gridsize = (2, 2), postproc_workers = 40, DK = 1):
    """Polygonize the raster to a vector polygons file.
    Because polygonization is slow and scales exponentially with image size, the raster is split into a grid of several
    smaller chunks, which are processed in parallel.
    As vector file merging is also very slow, the chunks are not merged into a single vector file, but instead linked in
    a virtual vector VRT file, which allows the viewing in QGIS as a single layer.
    """

    # Polygonise all raster predictions
    # for DK
    if DK:
        # raster_fps = glob.glob(f"{raster_dir}/det_1km*.tif")
        raster_fps = glob.glob(f"{raster_dir}/seg*.tif")

        # raster_chms = [i.replace('det_seg', 'det_CHM') for i in raster_fps]

    else:
        # for FI
        raster_fps = glob.glob(f"{raster_dir}/*det_seg.tif")
        # raster_chms = [os.path.join(raster_dir, 'det_chm_' + os.path.basename(i).replace('_det_seg', '')) for i in raster_fps]
    print('seg masks for polygonization:', raster_fps)
    # print('chm masks:', raster_chms)
    for ind in tqdm(range(len(raster_fps))):

        # Create a folder for the polygons VRT file, and a sub-folder for the actual gpkg data linked in the VRT
        prediction_name = os.path.splitext(os.path.basename(raster_fps[ind]))[0]
        polygons_dir = os.path.join(polygons_basedir, prediction_name)
        if os.path.exists(polygons_dir):
            print(f"Skipping, already processed {polygons_dir}")
            continue
        # os.mkdir(polygons_dir)
        os.makedirs(polygons_dir)
        os.mkdir(os.path.join(polygons_dir, "vrtdata"))

        # Create a list of rasterio windows to split the image into a grid of smaller chunks
        chunk_windows = []
        n_rows, n_cols = postproc_gridsize
        with rasterio.open(raster_fps[ind]) as raster:
            raster_crs = raster.crs
            width, height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                out_fp = os.path.join(polygons_dir, "vrtdata", f"{prediction_name}_{width * j}_{height * i}.gpkg")
                # chunk_windows.append([raster_fps[ind], raster_chms[ind], out_fp, Window(width * j, height * i, width, height)])
                chunk_windows.append([raster_fps[ind], out_fp, Window(width * j, height * i, width, height)])

        # Polygonise image chunks in parallel
        polygon_fps = []

        with multiprocessing.Pool(processes=postproc_workers) as pool:
            with tqdm(total=len(chunk_windows), desc="Polygonising raster chunks", position=1, leave=False) as pbar:
                for out_fp in pool.imap_unordered(polygonize_chunk, chunk_windows):
                    if out_fp:
                        polygon_fps.append(out_fp)
                    # dfall = dfall.append(df)
                    pbar.update()

        # Merge all polygon chunks into one polygon VRT
        create_vector_vrt(os.path.join(polygons_dir, f"polygons_{prediction_name}.vrt"), polygon_fps)
        out_dfall = os.path.join(polygons_dir, f"{prediction_name}_all_seg.gpkg")
        # dfall.to_file(out_dfall, driver="GPKG", crs=raster_crs, layer="trees")
        create_vector_gpkg(out_dfall, polygon_fps, raster_crs, out_layer_name="trees", pbar=False)

    # Create giant VRT of all polygon VRTs
    # merged_vrt_fp = os.path.join(polygons_basedir, f"all_polygons_{os.path.basename(postprocessing_dir)}.vrt")
    # create_vector_vrt(merged_vrt_fp, glob.glob(f"{polygons_basedir}/*/*.vrt"))

    # create gpkg
    retri = rasterio.open(raster_fps[0])
    raster_crs = retri.crs
    # print(raster_crs)
    loc = polygons_basedir.split('/')[-2].split('-')[-1]
    out_gpkg = os.path.join(polygons_basedir, "merged_all_polygon_" + loc + ".gpkg")
    merged_polygon_fp = glob.glob(f"{polygons_basedir}/*/seg*all_seg.gpkg")
    # ipdb.set_trace()
    create_vector_gpkg(out_gpkg, merged_polygon_fp, raster_crs, out_layer_name="trees", pbar=False)

    return

# def create_polygons_heights(raster_dir, polygons_basedir, postprocessing_dir, postproc_gridsize = (2, 2), postproc_workers = 40, DK = 1):
#     """Polygonize the raster to a vector polygons file.
#     Because polygonization is slow and scales exponentially with image size, the raster is split into a grid of several
#     smaller chunks, which are processed in parallel.
#     As vector file merging is also very slow, the chunks are not merged into a single vector file, but instead linked in
#     a virtual vector VRT file, which allows the viewing in QGIS as a single layer.
#     """

#     # Polygonise all raster predictions
#     # for DK
#     if DK:
#         raster_fps = glob.glob(f"{raster_dir}/det_1km*.tif")

#         raster_chms = [i.replace('det_seg', 'det_CHM') for i in raster_fps]

#     else:
#         # for FI
#         raster_fps = glob.glob(f"{raster_dir}/*det_seg.tif")
#         raster_chms = [os.path.join(raster_dir, 'det_chm_' + os.path.basename(i).replace('_det_seg', '')) for i in raster_fps]
#     print('seg masks for polygonization:', raster_fps)
#     print('chm masks:', raster_chms)
#     for ind in tqdm(range(len(raster_fps))):

#         # Create a folder for the polygons VRT file, and a sub-folder for the actual gpkg data linked in the VRT
#         prediction_name = os.path.splitext(os.path.basename(raster_fps[ind]))[0]
#         polygons_dir = os.path.join(polygons_basedir, prediction_name)
#         if os.path.exists(polygons_dir):
#             print(f"Skipping, already processed {polygons_dir}")
#             continue
#         # os.mkdir(polygons_dir)
#         os.makedirs(polygons_dir)
#         os.mkdir(os.path.join(polygons_dir, "vrtdata"))

#         # Create a list of rasterio windows to split the image into a grid of smaller chunks
#         chunk_windows = []
#         n_rows, n_cols = postproc_gridsize
#         with rasterio.open(raster_fps[ind]) as raster:
#             raster_crs = raster.crs
#             width, height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 out_fp = os.path.join(polygons_dir, "vrtdata", f"{prediction_name}_{width * j}_{height * i}.gpkg")
#                 chunk_windows.append([raster_fps[ind], raster_chms[ind], out_fp, Window(width * j, height * i, width, height)])

#         # Polygonise image chunks in parallel
#         polygon_fps = []

#         with multiprocessing.Pool(processes=postproc_workers) as pool:
#             with tqdm(total=len(chunk_windows), desc="Polygonising raster chunks", position=1, leave=False) as pbar:
#                 for out_fp in pool.imap_unordered(polygonize_chunk_heights, chunk_windows):
#                     if out_fp:
#                         polygon_fps.append(out_fp)
#                     # dfall = dfall.append(df)
#                     pbar.update()

#         # Merge all polygon chunks into one polygon VRT
#         create_vector_vrt(os.path.join(polygons_dir, f"polygons_{prediction_name}.vrt"), polygon_fps)
#         out_dfall = os.path.join(polygons_dir, f"{prediction_name}_all_seg.gpkg")
#         # dfall.to_file(out_dfall, driver="GPKG", crs=raster_crs, layer="trees")
#         create_vector_gpkg(out_dfall, polygon_fps, raster_crs, out_layer_name="trees", pbar=False)

#     # Create giant VRT of all polygon VRTs
#     merged_vrt_fp = os.path.join(polygons_basedir, f"all_polygons_{os.path.basename(postprocessing_dir)}.vrt")
#     create_vector_vrt(merged_vrt_fp, glob.glob(f"{polygons_basedir}/*/*.vrt"))

#     return

def CHMerror(pr, gt_mask_im,  nbins = 100):
    pr = pr.flatten()
    gt_mask_im = gt_mask_im.flatten()
    MAE = mean_absolute_error(gt_mask_im, pr)
    print('MAE for all', MAE)
    MEAE = median_absolute_error(gt_mask_im, pr)
    print('Median absolute error', MEAE)
    # plt.figure()
    # plt.hist2d(pr, gt_mask_im, cmap='Blues', bins=50, density =  1, norm=colors.LogNorm(), vmin = 0.0001)
    # plt.colorbar()
    # plt.xlim(0, 40)
    # plt.ylim(0, 40)
    # plt.plot([0, 40], [0, 40], alpha = 0.7, lw = 0.2, c = 'black', ls = '--')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('CHM ground truth vs estimation per pixel')
    # plt.xlabel('Estimation (m)')
    # plt.ylabel('Ground truth (m)')

    gtm = int(np.ceil(gt_mask_im.max()))
    xx = [0, gtm]

    inds = []
    intervals = [0, 10, 20, 30, gtm]
    for i in range(4):
        indi = [idx for idx,val in enumerate(gt_mask_im) if intervals[i] <= val < intervals[i+1]]
        inds.append(indi)


    preds = []
    gtts = []
    for i in range(4):
        predi = pr[inds[i]]
        preds.append(predi)
        gtti = gt_mask_im[inds[i]]
        gtts.append(gtti)

    maes = []

    for i in range(4):

        maes.append(abs(gtts[i] - preds[i]))

    # fig, ax = plt.subplots(figsize=(9,7))
    # ax.boxplot(maes, showfliers=False, showmeans = True)
    # ax.set_title('CHM error per pixel - interval errors', fontsize = 16)
    # labels = ['0-10m', '10-20m', '20-30m', '>30m']
    # ax.set_xticklabels(labels)
    # ax.set_xlabel('Height', fontsize = 14)
    # ax.set_ylabel('MAE', fontsize = 14)

    # ax.grid(True, axis = 'y', alpha = 0.3)
    def func(x, a, b):
        return a * x + b
        # return a * x + b

    popt, pcov = curve_fit(func, gt_mask_im, pr)


    fig = plt.figure(figsize = (15, 15))
    gs = GridSpec(6, 6)
    ax_scatter = fig.add_subplot(gs[1:5, 0:5])
    ax_hist_y = fig.add_subplot(gs[0,0:5])
    ax_hist_x = fig.add_subplot(gs[1:5, 5])
    ax_box_x = fig.add_subplot(gs[5, 0:5])

    # ax_scatter.scatter(gtt, ppd, s =1, c = 'grey')
    ax_scatter.hist2d(gt_mask_im, pr, cmap='Blues', alpha = 0.8, bins=50, density =  1, norm=colors.LogNorm(), vmin = 0.000001)
    ax_hist_y.hist(gt_mask_im, bins=80, color='navy', alpha=0.3, density = 1)
    ax_hist_y.set_ylim(0, 0.08)
    ax_hist_x.hist(pr, bins=80, color='navy', alpha=0.3, density = 1, orientation = 'horizontal')
    ax_hist_x.set_xlim(0, 0.08)
    ax_scatter.plot(xx, xx, c ='grey', alpha = 0.5)
    slope, intercept, r_value, p_value, std_err = linregress(gt_mask_im, pr)
    ax_scatter.plot(xx, func(np.array(xx), *popt), 'b--', alpha = 0.5, label='f(x) = %5.3f x + %5.3f; r2 = %5.3f ' % (popt[0], popt[1], r_value**2))
    ax_scatter.set_aspect('equal')
    ax_scatter.set_xlim(0, gtm)
    ax_scatter.set_ylim(0, gtm)
    ax_hist_y.set_xlim(0, gtm)
    ax_hist_x.set_ylim(0, gtm)
    # ax_scatter.set_xlabel('Reference height (m)', fontsize = 14)
    # ax_scatter.set_ylabel('Estimated height (m)', fontsize = 14)
    # plt.colorbar(ax = ax_scatter)
    ax_scatter.legend(prop={'size': 12})

    labels = ['0-10m', '10-20m', '20-30m', '>30m']
    # boxprops = dict(linestyle='--')
    line_props = dict(linestyle='--', color="grey", alpha=0.9)
    capprops = dict(linewidth = 1, color="grey", alpha = 0.9)
    bp = ax_box_x.boxplot(maes, widths = 0.3, showfliers=False, showmeans = True, whiskerprops=line_props, capprops = capprops)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    ax_box_x.set_xticklabels(labels)
    ax_box_x.set_xlabel('Height')
    ax_box_x.set_ylabel('MAE (m)')

    med = []
    mu = []
    for t in maes:
        med.append(np.median(t))
        mu.append(np.mean(t))

    # fig, ax = plt.subplots()
    # bp = ax.boxplot(data3, showmeans=True)

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' median: {:.2f}\n mean: {:.2f}'.format(med[i], mu[i])
        ax_box_x.annotate(text, xy=(x, y-1))

    ax_box_x.grid(True, axis = 'y', alpha = 0.3)

    plt.show()


    return



def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, output_shapefile_type, write_as_type = 'uint8', th = 0.5, create_countors = False, convert = 1, rescale = 0):
    # Convert to correct required before writing
    meta = detected_meta.copy()
    if convert:
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask<th]=0
            detected_mask[detected_mask>=th]=1

    if rescale:
        # for densty masks, multiply 10e4
        detected_mask = detected_mask*10000

    detected_mask = detected_mask.astype(write_as_type)
    if detected_mask.ndim != 2:
        detected_mask = detected_mask[0]

    meta['dtype'] =  write_as_type
    meta['count'] = 1
    if rescale:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 32767
                            }
                        )
    else:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 255
                            }
                        )
        ##################################################################################################
        ##################################################################################################
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
    if create_countors:
        wp = wp.replace(image_type, output_shapefile_type)
        # create_contours_shapefile(detected_mask, detected_meta, wp)


def writeMaskToDiskChm(detected_mask, detected_meta, wp, image_type, write_as_type = 'float32', scale = 1):
    # Convert to correct required before writing

    # scale by 100 and then store
    print('range height', detected_mask.min(), detected_mask.max())
    if scale:
        detected_mask = detected_mask * 100

    # print(detected_mask.max())

    print('mask', detected_mask.shape)
    print('meta', detected_meta['height'])
    detected_mask = detected_mask.astype(write_as_type)
    detected_meta['dtype'] =  write_as_type
    detected_meta['count'] = 1
    detected_meta.update(
                        {'compress':'lzw',
                            'nodata': 9999
                        }
                    )
    try:
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    except:
        detected_meta.update(
                        {'compress':'lzw',
                            'nodata': 9999,
                            'driver': 'GTiff',
                        }
                    )
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    return

def rgb2gray_convert(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
