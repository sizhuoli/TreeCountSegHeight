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

class predict:
    def __init__(self, config, DK = 1, largescale = 0, pred = 1, patches = 0, polygonize = 0):
        """Initialize the predictor

        load images
        load model

        """
        self.config = config
        if not polygonize:
            # run prediction
            import tensorflow as tf
            print(tf.__version__)
            print(tf.config.list_physical_devices('GPU'))
            from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
            from core2.optimizers import adaDelta, adagrad, adam, nadam
            from core2.frame_info_multires import FrameInfo, image_normalize
            from tensorflow.keras.models import load_model

            if not self.config.change_input_size:
                # do not change input image patch size (normal case: 256)
                if DK:
                    if largescale:
                        self.all_files = load_files(config)
                    else:
                        print('files not loaded correctly')
                else:
                    print('config for dk not loaded correctly')

                if pred: # make predicitons
                    OPTIMIZER = adam
                    self.models = []
                    for mod in config.trained_model_path:
                        modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                        modeli.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
                        self.models.append(modeli)

                    self.model_chm  = 0
                else:
                    print('polygonization only')


            elif self.config.change_input_size:
                # change input patch size, in case of different image resolution compared with training data
                OPTIMIZER = adam
                self.all_files = load_files(config)
                self.models = []
                for mod in config.trained_model_path:
                    modeli = load_model(mod, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
                    modeli.summary()
                    self.weiwei = modeli.get_weights()
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
                # change input size for chm model as well
                self.model_chm = load_model(self.config.trained_model_path_chm, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)

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
        """main predictor"""
        pred_segcountdk(self.all_files, self.models, self.config)



    def polyonize_DK(self, postproc_gridsize = (2, 2)):
        """polygonize output rasters after prediction"""
        from osgeo import ogr, gdal
        all_dirs = [self.config.output_dir]
        for di in all_dirs:
            polygons_dir = os.path.join(di, "polygons")
            if not os.path.exists(polygons_dir):
                create_polygons(di, polygons_dir, di, postproc_gridsize, postproc_workers = 40)


def load_files(config):
    """load all images"""
    exclude = set(['md5'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    return all_files



def pred_segcountdk(all_files, models, config):
    """load all images and chm files, run predictions, save output predictions"""
    th = 0.5
    # save count per image in a dateframe
    counts = {}
    outputFiles = []
    nochm = [] # if no chm exists for the corresponding image
    waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(config.output_dir, filename.replace(config.input_image_pref, config.output_prefix).replace(config.input_image_type, config.output_image_type))
        outputFiles.append(outputFile)
        if not os.path.isfile(outputFile) or config.overwrite_analysed_files:
            if not config.single_raster and config.aux_data:
                with rasterio.open(fullPath) as core:
                    auxx = []
                    coor = fullPath[-12:-9]+ fullPath[-8:-5]
                    chmpath = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
                    aux2 = fullPath.replace(config.input_image_pref, config.aux_prefs[-1]).replace(config.input_image_dir, chmpath)
                    auxx.append(aux2)
                    try:
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_rawtif_ndvi(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster)
                    except IOError:
                        try:
                            auxx[-1] = waterchm
                            nochm.append(outputFile)
                            detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_rawtif_ndvi(config, models, [core, *auxx], width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster)
                        except:
                            continue
                    try:
                        writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype, th = th)
                        writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg', 'density'), image_type = config.output_image_type, write_as_type = 'int16', th = th, convert = 0, rescale = 1)
                        counts[filename] = detectedMaskDens.sum()
                    except:
                        continue
            else:
                print('Single raster or multi raster without aux')
                with rasterio.open(fullPath) as img:
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount(img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, auxData = config.aux_data, singleRaster=config.single_raster)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type = config.output_image_type, write_as_type = config.output_dtype, th = th)
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg', 'density'), image_type = config.output_image_type, write_as_type = 'int16', th = th, convert = 0, rescale = 1)
                    counts[filename] = detectedMaskDens.sum()
        else:
            print('File already analysed!', fullPath)
    return



def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    """add patch predictions to the final prediction for the large tif image"""
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
    res[row:row+he, col:col+wi] =  resultant
    return (res)


def predict_using_model_segcount(model, batch, batch_pos, maskseg, maskdens, operator):
    """run model prediction to obtain segmentation masks and tree count density estimation masks"""
    b1 = batch[0]
    if len(b1) == 2: # 2 inputs
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
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    return maskseg, maskdens



def detect_tree_segcount_rawtif_ndvi(config, models, img, width=256, height=256, stride = 128, normalize=True, auxData = 0, singleRaster = 1):
    """load tif image, read patch by patch, predict patch by patch, then merge predictions"""

    if not singleRaster and auxData:
        core_img = img[0]
        nols, nrows = core_img.meta['width'], core_img.meta['height']
        meta = core_img.meta.copy()
        aux_channels1 = len(img) - 1 # aux channels with same resolution as color input

    else:
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    masks_seg = np.zeros((len(models), nrows, nols), dtype=np.float32)
    masks_dens = np.zeros((len(models), nrows, nols), dtype=np.float32)

    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)

        if not singleRaster and auxData: #multi rasters with possibly different resolutions resampled to the same resolution
            nc1 = meta['count'] + aux_channels1 #4+1 = 5 channels (normal input)
            patch1 = np.zeros((height, width, nc1)) # 256, 256, 5
            temp_im1 = core_img.read(window=window)

            # compute ndvi here
            temp_im1 = np.transpose(temp_im1, axes=(1,2,0)) # channel last

            NDVI = (temp_im1[:, :, -1].astype(float) - temp_im1[:, :, 0].astype(float)) / (temp_im1[:, :, -1].astype(float) + temp_im1[:, :, 0].astype(float))
            NDVI = NDVI[..., np.newaxis]
            temp_im1 = np.append(temp_im1, NDVI, axis = -1)

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
        else:
            patch = np.zeros((height, width, meta['count'])) #Add zero padding in case of corner images
            temp_im1 = img.read(window=window)

        temp_im2 = np.transpose(temp_im2, axes=(1,2,0))

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            temp_im2 = image_normalize(temp_im2, axis=(0,1))

        patch1[:window.height, :window.width] = temp_im1
        patch2[:window2.height, :window2.width] = temp_im2
        batch.append([patch1, patch2])
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        if (len(batch) == config.BATCH_SIZE):
            for mi in range(len(models)):
                curmask_seg = masks_seg[mi, :, :]
                curmask_dens = masks_dens[mi, :, :]
                curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')

            batch = []
            batch_pos = []
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        for mi in range(len(models)):
            curmask_seg = masks_seg[mi, :, :]
            curmask_dens = masks_dens[mi, :, :]
            curmask_seg = predict_using_model_segcount(models[mi], batch, batch_pos, curmask_seg, curmask_dens, 'MAX')
        batch = []
        batch_pos = []

    return(masks_seg, masks_dens, meta)



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

        return out_fp


def create_polygons(raster_dir, polygons_basedir, postprocessing_dir, postproc_gridsize = (2, 2), postproc_workers = 40, DK = 1):
    """Polygonize the raster to a vector polygons file.
    Because polygonization is slow and scales exponentially with image size, the raster is split into a grid of several
    smaller chunks, which are processed in parallel.
    As vector file merging is also very slow, the chunks are not merged into a single vector file, but instead linked in
    a virtual vector VRT file, which allows the viewing in QGIS as a single layer.
    """


    # for DK
    if DK:
        # raster_fps = glob.glob(f"{raster_dir}/det_1km*.tif")
        raster_fps = glob.glob(f"{raster_dir}/seg_1km*.tif")

    else:
        # for FI
        raster_fps = glob.glob(f"{raster_dir}/*det_seg.tif")
    print('seg masks for polygonization:', raster_fps)
    for ind in tqdm(range(len(raster_fps))):

        # Create a folder for the polygons VRT file, and a sub-folder for the actual gpkg data linked in the VRT
        prediction_name = os.path.splitext(os.path.basename(raster_fps[ind]))[0]
        polygons_dir = os.path.join(polygons_basedir, prediction_name)
        if os.path.exists(polygons_dir):
            print(f"Skipping, already processed {polygons_dir}")
            continue
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


    # create gpkg
    retri = rasterio.open(raster_fps[0])
    raster_crs = retri.crs
    loc = polygons_basedir.split('/')[-2].split('-')[-1]
    out_gpkg = os.path.join(polygons_basedir, "merged_all_polygon_" + loc + ".gpkg")
    merged_polygon_fp = glob.glob(f"{polygons_basedir}/*/seg*all_seg.gpkg")
    create_vector_gpkg(out_gpkg, merged_polygon_fp, raster_crs, out_layer_name="trees", pbar=False)

    return





def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, write_as_type = 'uint8', th = 0.5, convert = 1, rescale = 0):
    """Write the prediction masks to tif files
    For segmentation mask: convert from floats to binary results (0, 1) using threshold of 0.5
    For density estimation mask: multiply the raw predictions by 1e4 and save as int (to save space on disk); need to devide by 1e4 when using.
    """
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
        meta.update({'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 32767})
    else:
        meta.update({'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 255
                            })

    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
