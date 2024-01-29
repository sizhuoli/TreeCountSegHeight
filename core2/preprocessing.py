#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:51:14 2021

@author: sizhuo
"""

import os
os.environ['PROJ_LIB'] = '/usr/share/proj'
os.environ['GDAL_DATA'] = '/usr/share/gdal'
import ipdb
#import torch
#import torch.nn.functional as F
import math
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.mask
import rasterio.warp             # Reproject raster samples
import rasterio.merge
from rasterio.transform import rowcol
import fiona                     # I/O vector data (shape, geojson, ...)
import pyproj                    # Change coordinate reference system
import geopandas as gps
import pandas as pd
import shapely
from shapely.geometry import box, Point
import json

import numpy as np               # numerical array manipulation
import time
import os
import glob
from PIL import Image
import PIL.ImageDraw
#from core.visualize import display_images
from core2.visualize import display_images
#from core.frame_info import image_normalize
from core2.frame_info import image_normalize
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt  # plotting tools
# %matplotlib inline
from tqdm import tqdm
import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")

# %reload_ext autoreload
# %autoreload 2
#from IPython.core.interactiveshell import InteractiveShell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Required configurations (including the input and output paths) are stored in a separate file (such as config/Preprocessing.py)
# Please provide required info in the file before continuing with this notebook. 
 

class processor:
    def __init__(self, config, boundary = 0, aux = 0):
        self.config = config
        trainingPolygon, trainingArea = load_polygons(self.config)
        if aux: 
            # with aux
            self.inputImages = readInputImages(config.raw_image_base_dir, config.raw_image_file_type, config.prediction_pre, config.raw_image_prefix, config.raw_aux_prefix, config.single_raster)
        else:
            # no aux
            self.inputImages = readInputImages(config.raw_image_base_dir, config.raw_image_file_type, config.prediction_pre, config.raw_image_prefix, config.single_raster)
        #self.inputImages = glob.glob(r'C:/Users/a.zenonos/Desktop/Troodos/Troodos_Orthophotos2019/*.tif',recursive=True)
        #print ('error source',self.inputImages[0])
        print(f'Found a total of {len(self.inputImages)} (pair of) raw image(s) to process!')
        print('Filename:', self.inputImages)

        
        if boundary: # if compute boundary
            # areasWithPolygons contains the object polygons and weighted boundaries for each area!
            self.areasWithPolygons = dividePolygonsInTrainingAreas(trainingPolygon, trainingArea, self.config)
            print(f'Assigned training polygons in {len(self.areasWithPolygons)} training areas and created weighted boundaries for ploygons')
        else:
            # no boundaries
            self.areasWithPolygons = dividePolygonsInTrainingAreas(trainingPolygon, trainingArea, self.config, bound = 0)
            print(f'Assigned training polygons in {len(self.areasWithPolygons)} training areas and created weighted boundaries for ploygons')

    def extract_normal(self, boundary = 0, aux = 0):
        if boundary:
            
            
            # Run the main function for extracting part of ndvi and pan images that overlap with training areas
            writeCounter=0
            # multi raster with aux
            if aux:
                writeCounter = extractAreasThatOverlapWithTrainingData(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames,  self.config.extracted_annotation_filename, self.config.extracted_boundary_filename, self.config.bands , writeCounter, self.config.normalize, self.config.aux_channel_prefixs, self.config.aux_bands,  self.config.single_raster, kernel_size = 15, kernel_sigma = 4)
            else:
                # single raster or multi raster without aux
                writeCounter = extractAreasThatOverlapWithTrainingData(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames, self.config.extracted_annotation_filename, self.config.extracted_boundary_filename, self.config.bands,  writeCounter, self.config.single_raster, kernel_size = 15, kernel_sigma = 4)
            
        elif not boundary:
            # no boundary weights
            writeCounter=0
            if aux:
                # multi raster with aux
                writeCounter = extractAreasThatOverlapWithTrainingData(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames,  self.config.extracted_annotation_filename, None, self.config.bands , writeCounter, self.config.normalize, self.config.aux_channel_prefixs, self.config.aux_bands,  self.config.single_raster, kernel_size = 15, kernel_sigma = 4, detchm = self.config.detchm)
            else:
                # no aux
                writeCounter = extractAreasThatOverlapWithTrainingData(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames,  self.config.extracted_annotation_filename, None, self.config.bands , writeCounter, self.config.normalize, None, None,  self.config.single_raster, kernel_size = 15, kernel_sigma = 4)


        
            
    def extract_svls(self, boundary = 0, aux = 0):
        if not boundary:
            # no boundar weights
            writeCounter=0
            # multi raster with aux
            writeCounter = extractAreasThatOverlapWithTrainingData_svls(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames,  self.config.extracted_annotation_filename, None, self.config.bands , writeCounter, self.config.aux_channel_prefixs, self.config.aux_bands,  self.config.single_raster, kernel_size = 15, kernel_sigma = 4, kernel_size_svls = self.config.kernel_size_svls, sigma_svls = self.config.kernel_sigma_svls)
         
        

def load_polygons(config):
    #Read the training area and training polygons
    trainingArea = gps.read_file(os.path.join(config.training_base_dir, config.training_area_fn))
    trainingPolygon = gps.read_file(os.path.join(config.training_base_dir, config.training_polygon_fn))
    
    print(f'Read a total of {trainingPolygon.shape[0]} object polygons and {trainingArea.shape[0]} training areas.')
    print('Polygons will be assigned to training areas in the next steps.')
    
    
    #Check if the training areas and the training polygons have the same crs
    if trainingArea.crs  != trainingPolygon.crs:
        print('Training area CRS does not match training_polygon CRS')
        targetCRS = trainingPolygon.crs #Areas are less in number so conversion should be faster
        trainingArea = trainingArea.to_crs(targetCRS)
    print(trainingPolygon.crs)
    print(trainingArea.crs)
    assert trainingPolygon.crs == trainingArea.crs
    
    # Assign serial IDs to training areas
    trainingArea['id'] = range(trainingArea.shape[0])
    
    return trainingPolygon, trainingArea



# Create boundary from polygon file
def calculateBoundaryWeight(polygonsInArea, scale_polygon = 1.5, output_plot = True): 
    '''
    For each polygon, create a weighted boundary where the weights of shared/close boundaries is higher than weights of solitary boundaries.
    '''
    # If there are polygons in a area, the boundary polygons return an empty geo dataframe
    if not polygonsInArea:
        # print('No polygons')
        return gps.GeoDataFrame({})
    tempPolygonDf = pd.DataFrame(polygonsInArea)
    tempPolygonDf.reset_index(drop=True,inplace=True)
    tempPolygonDf = gps.GeoDataFrame(tempPolygonDf.drop(columns=['id']))
    new_c = []
    #for each polygon in area scale, compare with other polygons:
    for i in tqdm(range(len(tempPolygonDf))):
        pol1 = gps.GeoSeries(tempPolygonDf.iloc[i][0])
        sc = pol1.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
        scc = pd.DataFrame(columns=['id', 'geometry'])
        scc = scc.append({'id': None, 'geometry': sc[0]}, ignore_index=True)
        scc = gps.GeoDataFrame(pd.concat([scc]*len(tempPolygonDf), ignore_index=True))

        pol2 = gps.GeoDataFrame(tempPolygonDf[~tempPolygonDf.index.isin([i])])
        #scale pol2 also and then intersect, so in the end no need for scale
        pol2 = gps.GeoDataFrame(pol2.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center'))
        pol2.columns = ['geometry']

        #invalid intersection operations topo error
        try:

            ints = scc.intersection(pol2)
            for k in range(len(ints)):
                if ints.iloc[k]!=None:
                    if ints.iloc[k].is_empty !=1:
                        new_c.append(ints.iloc[k])
        except:
            print('Intersection error')
    new_c = gps.GeoSeries(new_c)
    new_cc = gps.GeoDataFrame({'geometry': new_c})
    new_cc.columns = ['geometry']
    
    # df may contains point other than polygons
    new_cc['type'] = new_cc['geometry'].type 
    new_cc = new_cc[new_cc['type'].isin(['Polygon', 'MultiPolygon'])]
    new_cc.drop(columns=['type'])
    
    tempPolygonDf['type'] = tempPolygonDf['geometry'].type 
    tempPolygonDf = tempPolygonDf[tempPolygonDf['type'].isin(['Polygon', 'MultiPolygon'])]
    tempPolygonDf.drop(columns=['type'])
    # print('new_cc', new_cc.shape)
    # print('tempPolygonDf', tempPolygonDf.shape)
    if new_cc.shape[0] == 0:
        print('No boundaries')
        return gps.GeoDataFrame({})
    else:
        bounda = gps.overlay(new_cc, tempPolygonDf, how='difference')

        if output_plot:
            # fig, ax = plt.subplots(figsize = (10,10))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
            bounda.plot(ax=ax1,color = 'red')
            tempPolygonDf.plot(alpha = 0.2,ax = ax1,color = 'b')
            # plt.show()
            ###########################################################

            bounda.plot(ax=ax2,color = 'red')
            plt.show()
        #change multipolygon to polygon
        bounda = bounda.explode()
        bounda.reset_index(drop=True,inplace=True)
        #bounda.to_file('boundary_ready_to_use.shp')
        return bounda

# As input we received two shapefile, first one contains the training areas/rectangles and other contains the polygon of trees/objects in those training areas
# The first task is to determine the parent training area for each polygon and generate a weight map based upon the distance of a polygon boundary to other objects.
# Weight map will be used by the weighted loss during the U-Net training

def dividePolygonsInTrainingAreas(trainingPolygon, trainingArea, config, bound = 1):
    '''
    Assign annotated ploygons in to the training areas.
    '''
    # For efficiency, assigned polygons are removed from the list, we make a copy here. 
    cpTrainingPolygon = trainingPolygon.copy()
    splitPolygons = {}
    for i in tqdm(trainingArea.index):
        spTemp = []
        allocated = []
        for j in cpTrainingPolygon.index:
            try:
                if trainingArea.loc[i]['geometry'].intersects(cpTrainingPolygon.loc[j]['geometry']):
                    spTemp.append(cpTrainingPolygon.loc[j])
                    allocated.append(j)
            except:
                print('Labeling Error: polygon number {d1} in area {d2} is nonetype (empty).'.format(d1 = j, d2 = i))


            # Order of bounds: minx miny maxx maxy
        if bound:
            boundary = calculateBoundaryWeight(spTemp, scale_polygon = 1.5, output_plot = config.show_boundaries_during_processing)
            splitPolygons[trainingArea.loc[i]['id']] = {'polygons':spTemp, 'boundaryWeight': boundary, 'bounds':list(trainingArea.bounds.loc[i]),}
        else: #no boundary weights
            splitPolygons[trainingArea.loc[i]['id']] = {'polygons':spTemp, 'bounds':list(trainingArea.bounds.loc[i]),}

        cpTrainingPolygon = cpTrainingPolygon.drop(allocated)
    return splitPolygons


def readInputImages(imageBaseDir, rawImageFileType, predictionPrefix, rawImagePre, rawAuxPrefix = None, single_raster = 1):
    """
    Reads all multichannel images in the image_base_dir directory. 
    """     
    
    if not single_raster and rawAuxPrefix:
        # raw multi tif with aux data
        inputImages = []
        for root, dirs, files in os.walk(imageBaseDir):
            for file in files:
                if type(rawImagePre) == str:
                    if file.endswith(rawImageFileType) and not file.startswith(predictionPrefix) and file.startswith(rawImagePre):
                        fileFn = os.path.join(root, file)
                          
                        auxFn = []
                        for aux in rawAuxPrefix:
                            auxImageFni = fileFn.replace(rawImagePre, aux)
                            auxFn.append(auxImageFni)
                        
                        inputImages.append((fileFn, *auxFn))
                elif type(rawImagePre) == list:
                    if file.endswith(rawImageFileType) and not file.startswith(predictionPrefix) and file[0] in rawImagePre:
                        fileFn = os.path.join(root, file)
                          
                        auxFn = []
                        for aux in rawAuxPrefix:
                            auxImageFni = fileFn.replace(file[0], aux)
                            auxFn.append(auxImageFni)
                        
                        inputImages.append((fileFn, *auxFn))
                    
    
    else: #single_raster or not rawAuxPrefix:
        # only one single raster or multi raster without aux
        inputImages = []
        
        for root, dirs, files in os.walk(imageBaseDir):
            for file in files:
                #import ipdb; ipdb.set_trace()
                # not with det in case prediction files exist
                if file.endswith(rawImageFileType) and not file.startswith(predictionPrefix):
                     inputImages.append(root + '/' + file)
                    # import ipdb; ipdb.set_trace()#

    return inputImages






def drawPolygons_kernel(polygons, shape, outline, fill):
    """
    From the polygons, create a numpy mask with fill value in the foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    #Syntax: PIL.ImageDraw.Draw.polygon(xy, fill=None, outline=None)
    #Parameters:
    #xy – Sequence of either 2-tuples like [(x, y), (x, y), …] or numeric values like [x, y, x, y, …].
    #outline – Color to use for the outline.
    #fill – Color to use for the fill.
    #Returns: An Image object.
    for polygon in polygons:
        xy = [(polygon[1], polygon[0])]
        draw.point(xy=xy, fill=1)
    mask = np.array(mask)#, dtype=bool)  
    # print('unique', np.unique(mask))
    return(mask)

def drawPolygons_ann(polygons, shape, outline, fill):
    """
    From the polygons, create a numpy mask with fill value in the foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=outline, fill=fill)
    mask = np.array(mask)#, dtype=bool)  
    return(mask)

def writeExtractedImageAndAnnotation(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize, kernel_size, kernel_sigma, chm, detchm = 0):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    """
    try:
        
        for band, imFn in zip(bands, imagesFilename):
            # Rasterio reads file channel first, so the sm[0] has the shape [1 or ch_count, x,y]
            # If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
            dt = sm[0][band].astype(profile['dtype'])
            
            if chm:
                if detchm: # using det chm
                    print('processing det chm here')
                    print('before min, max, mean, std', dt.min(), dt.max(), dt.mean(), dt.std())
                    dt = dt/100
                    dt[dt<1] = 0
                    print('rescale min, max, mean, std', dt.min(), dt.max(), dt.mean(), dt.std())
                elif not detchm:
                    print('processing reference chm here')
                    dt = dt
                if normalize:
                    # print('robust scaling for CHM')
                    # q1 = np.quantile(dt, 0.25)
                    # q3 = np.quantile(dt, 0.75)
                    # dt = (dt-q1)/(q3-q1)
                    print('normalize chm')
                    dt = image_normalize(dt, axis=None)
            else:
                if normalize: # Note: If the raster contains None values, then you should normalize it separately by calculating the mean and std without those values.
                    dt = image_normalize(dt, axis=None) # to normalize with means and std computed channel wise #  Normalize the image along the width and height, and since here we only have one channel we pass axis as None

            profile.update(compress='lzw')
            with rasterio.open(os.path.join(writePath, imFn+'_{}.png'.format(writeCounter)), 'w', **profile) as dst:
                # ipdb.set_trace()

                # np.save(os.path.join(writePath, imFn+'_{}.numpy'.format(writeCounter)))
                dst.write(dt, 1)
                    
        countFilename = os.path.join(writePath, 'count'+'_{}.npy'.format(writeCounter))
        curcount = len(polygonsInAreaDf)
        np.save(countFilename, curcount)
        
        if annotationFilename:
            #ipdb.set_trace()
            # print('saving ann')
            annotation_json_filepath = os.path.join(writePath,annotationFilename+'_{}.json'.format(writeCounter))
            # The object is given a value of 1, the outline or the border of the object is given a value of 0 and rest of the image/background is given a a value of 0
            rowColPolygons(polygonsInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, annotation_json_filepath, outline=0, fill = 1, kernel_size =kernel_size, kernel_sigma = kernel_sigma, gaussian = 1)
        if boundaryFilename:
            boundary_json_filepath = os.path.join(writePath,boundaryFilename+'_{}.json'.format(writeCounter))
            # The boundaries are given a value of 1, the outline or the border of the boundaries is also given a value of 1 and rest is given a value of 0
            rowColPolygons(boundariesInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, boundary_json_filepath, outline=1 , fill=1, kernel_size =kernel_size, kernel_sigma = kernel_sigma, gaussian = 0)
        return(writeCounter+1)
    except Exception as e:
        print(e)
        print("Something nasty happened, could not write the annotation or the mask file!")
        ipdb.set_trace()
        return writeCounter
    
    


def writeExtractedImageAndAnnotation_svls(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, kernel_size, kernel_sigma, kernel_size_svls, sigma_svls, normalize=True):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    """
    # try:
        
    for band, imFn in zip(bands, imagesFilename):
        # Rasterio reads file channel first, so the sm[0] has the shape [1 or ch_count, x,y]
        # If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
        dt = sm[0][band].astype(profile['dtype'])
        if normalize: # Note: If the raster contains None values, then you should normalize it separately by calculating the mean and std without those values.
            dt = image_normalize(dt, axis=None) # to normalize with means and std computed channel wise #  Normalize the image along the width and height, and since here we only have one channel we pass axis as None
        with rasterio.open(os.path.join(writePath, imFn+'_{}.png'.format(writeCounter)), 'w', **profile) as dst:
                dst.write(dt, 1)
                
    # countFilename = os.path.join(writePath, 'count'+'_{}.npy'.format(writeCounter))
    # curcount = len(polygonsInAreaDf)
    # np.save(countFilename, curcount)
    
    if annotationFilename:
        
        annotation_json_filepath = os.path.join(writePath,annotationFilename+'_{}.json'.format(writeCounter))
        # The object is given a value of 1, the outline or the border of the object is given a value of 0 and rest of the image/background is given a a value of 0
        rowColPolygons_svls(polygonsInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, annotation_json_filepath, outline=0, fill = 1, kernel_size =kernel_size, kernel_sigma = kernel_sigma, kernel_size_svls = kernel_size_svls, sigma_svls = sigma_svls)
    # if boundaryFilename:
    #     boundary_json_filepath = os.path.join(writePath,boundaryFilename+'_{}.json'.format(writeCounter))
    #     # The boundaries are given a value of 1, the outline or the border of the boundaries is also given a value of 1 and rest is given a value of 0
    #     rowColPolygons(boundariesInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, boundary_json_filepath, outline=1 , fill=1)
    return(writeCounter+1) 

def findOverlap_svls(img, areasWithPolygons, writePath, imageFilename, annotationFilename, boundaryFilename, bands, kernel_size, kernel_sigma, kernel_size_svls, sigma_svls, writeCounter=1):
    """
    Finds overlap of image with a training area.
    Use writeExtractedImageAndAnnotation() to write the overlapping training area and corresponding polygons in separate image files.
    """
    overlapppedAreas = set()
    
    
    for areaID, areaInfo in areasWithPolygons.items():
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gps.GeoDataFrame(areaInfo['polygons'])
        # polygonsInAreaDf = polygonsInAreaDf.explode()
        if 'boundaryWeight' in areaInfo:
            boundariesInAreaDf = gps.GeoDataFrame(areaInfo['boundaryWeight'])
        else:
            boundariesInAreaDf = None
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)
        
    
        
        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            profile = img.profile  
            sm = rasterio.mask.mask(img, [bboxArea], all_touched=True, crop=True )
            profile['height'] = sm[0].shape[1]
            profile['width'] = sm[0].shape[2]
            profile['transform'] = sm[1]
            # That's a problem with rasterio, if the height and the width are less then 256 it throws: ValueError: blockysize exceeds raster height 
            # So I set the blockxsize and blockysize to prevent this problem
            profile['blockxsize'] = 32
            profile['blockysize'] = 32
            profile['count'] = 1
            profile['dtype'] = rasterio.float32
            # writeExtractedImageAndAnnotation writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = writeExtractedImageAndAnnotation_svls(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter, kernel_size, kernel_sigma, kernel_size_svls, sigma_svls)
            overlapppedAreas.add(areaID)
    return(writeCounter, overlapppedAreas)


def findOverlap(img, areasWithPolygons, writePath, imageFilename, annotationFilename, boundaryFilename, bands, normalize, kernel_size, kernel_sigma, writeCounter=1, chm = 0, detchm = 0):
    """
    Finds overlap of image with a training area.
    Use writeExtractedImageAndAnnotation() to write the overlapping training area and corresponding polygons in separate image files.
    """
    overlapppedAreas = set()
    
    
    for areaID, areaInfo in areasWithPolygons.items():
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gps.GeoDataFrame(areaInfo['polygons'])
        if 'boundaryWeight' in areaInfo:
            boundariesInAreaDf = gps.GeoDataFrame(areaInfo['boundaryWeight'])
        else:
            boundariesInAreaDf = None
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)
        
    
        
        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            profile = img.profile  
            sm = rasterio.mask.mask(img, [bboxArea], all_touched=True, crop=True )
            # print(profile.keys())
            profile['height'] = sm[0].shape[1]
            profile['width'] = sm[0].shape[2]
            profile['transform'] = sm[1]
            # That's a problem with rasterio, if the height and the width are less then 256 it throws: ValueError: blockysize exceeds raster height 
            # So I set the blockxsize and blockysize to prevent this problem
            profile['blockxsize'] = 32
            profile['blockysize'] = 32
            profile['count'] = 1
            profile['dtype'] = rasterio.float32
            print('Do local normalizaton')
            # writeExtractedImageAndAnnotation writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = writeExtractedImageAndAnnotation(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize, kernel_size, kernel_sigma, chm, detchm)
            overlapppedAreas.add(areaID)
    return(writeCounter, overlapppedAreas)




def extractAreasThatOverlapWithTrainingData_svls(inputImages, areasWithPolygons, writePath, channelNames,  annotationFilename, boundaryFilename, bands, writeCounter, auxChannelNames = None, auxBands = None, singleRaster = 1, kernel_size = 15, kernel_sigma = 4, kernel_size_svls = 3, sigma_svls = 1):
    """
    Iterates over raw ndvi and pan images and using findOverlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
    Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
    """
    if not os.path.exists(writePath):
        os.makedirs(writePath)
    
    overlapppedAreas = set()  
    
    if not singleRaster and auxChannelNames:
        # raw tif with aux info
        print('Multi raster with aux data')             
        for imgs in tqdm(inputImages):
            # main image at imgs[0]
            Img = rasterio.open(imgs[0])
            ncimg,imOverlapppedAreasImg = findOverlap_svls(Img, areasWithPolygons, writePath=writePath, imageFilename=channelNames, annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, kernel_size = kernel_size, kernel_sigma = kernel_sigma, kernel_size_svls = kernel_size_svls, sigma_svls = sigma_svls, writeCounter=writeCounter)
            for aux in range(len(auxChannelNames)):
                auxImgi = rasterio.open(imgs[aux+1])
    
                ncauxi, imOverlapppedAreasAuxi = findOverlap_svls(auxImgi, areasWithPolygons, writePath=writePath, imageFilename=auxChannelNames[aux], annotationFilename='', boundaryFilename='', bands=auxBands[aux], kernel_size = kernel_size, kernel_sigma = kernel_sigma, kernel_size_svls = kernel_size_svls, sigma_svls = sigma_svls, writeCounter=writeCounter )
            
            if ncimg == ncauxi:
                writeCounter = ncimg
            else: 
                print('Couldnt create mask!!!')
                print(ncimg)
                print(ncauxi)
                break;
            if overlapppedAreas.intersection(imOverlapppedAreasImg):
                print(f'Information: Training area(s) {overlapppedAreas.intersection(imOverlapppedAreasImg)} spans over multiple raw images. This is common and expected in many cases. A part was found to overlap with current input image.')
            overlapppedAreas.update(imOverlapppedAreasImg)
            
    
    
    else:
        print('Single raster or multi raster without aux')
        for imgs in tqdm(inputImages):
        
            rasterImg = rasterio.open(imgs)
            writeCounter, imOverlapppedAreas = findOverlap_svls(rasterImg, areasWithPolygons, writePath=writePath, imageFilename=channelNames, annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, kernel_size = kernel_size, kernel_sigma = kernel_sigma, kernel_size_svls = kernel_size_svls, sigma_svls = sigma_svls, writeCounter=writeCounter)
            
            overlapppedAreas.update(imOverlapppedAreas)
    allAreas = set(areasWithPolygons.keys())
    if allAreas.difference(overlapppedAreas):
        print(f'Warning: Could not find a raw image correspoinding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')
    return writeCounter







def extractAreasThatOverlapWithTrainingData(inputImages, areasWithPolygons, writePath, channelNames,  annotationFilename, boundaryFilename, bands, writeCounter, normalize, auxChannelNames = None, auxBands = None, singleRaster = 1, kernel_size = 15, kernel_sigma = 4, detchm = 0):
    """
    Iterates over raw ndvi and pan images and using findOverlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
    Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
    """
    if not os.path.exists(writePath):
        os.makedirs(writePath)
    
    overlapppedAreas = set()  
    
    # auxChannelNames = 1
    
    # singleRaster = 0
    if auxChannelNames and not singleRaster:
        # raw tif with aux info
        print('Multi raster with aux data')             
        for imgs in tqdm(inputImages):
            # main image at imgs[0]
            Img = rasterio.open(imgs[0])
            ncimg,imOverlapppedAreasImg = findOverlap(Img, areasWithPolygons, writePath=writePath, imageFilename=channelNames, annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, kernel_size = kernel_size, kernel_sigma = kernel_sigma, normalize = normalize, writeCounter=writeCounter)

            for aux in range(len(auxChannelNames)):
                auxImgi = rasterio.open(imgs[aux+1])
                if aux == 0: # 
                    print('Processing CHM ')
                    if detchm:
                        ncauxi, imOverlapppedAreasAuxi = findOverlap(auxImgi, areasWithPolygons, writePath=writePath, imageFilename=auxChannelNames[aux], annotationFilename='', boundaryFilename='', bands=auxBands[aux], kernel_size = kernel_size, kernel_sigma = kernel_sigma, normalize = normalize, writeCounter=writeCounter, chm = 1, detchm = detchm )

                    else:
                    
                        ncauxi, imOverlapppedAreasAuxi = findOverlap(auxImgi, areasWithPolygons, writePath=writePath, imageFilename=auxChannelNames[aux], annotationFilename='', boundaryFilename='', bands=auxBands[aux], kernel_size = kernel_size, kernel_sigma = kernel_sigma, normalize = normalize, writeCounter=writeCounter, chm = 1 )
                    
                else:
                    ncauxi, imOverlapppedAreasAuxi = findOverlap(auxImgi, areasWithPolygons, writePath=writePath, imageFilename=auxChannelNames[aux], annotationFilename='', boundaryFilename='', bands=auxBands[aux], kernel_size = kernel_size, kernel_sigma = kernel_sigma, normalize = normalize, writeCounter=writeCounter )
                
            if ncimg == ncauxi:
                writeCounter = ncimg
            else: 
                print('Couldnt create mask!!!')
                print(ncimg)
                print(ncauxi)
                break;
            if overlapppedAreas.intersection(imOverlapppedAreasImg):
                print(f'Information: Training area(s) {overlapppedAreas.intersection(imOverlapppedAreasImg)} spans over multiple raw images. This is common and expected in many cases. A part was found to overlap with current input image.')
            overlapppedAreas.update(imOverlapppedAreasImg)
    
    else:
        print('————————————————————————Single raster or multi raster without aux')
        for imgs in tqdm(inputImages):
        
            rasterImg = rasterio.open(imgs)
            writeCounter, imOverlapppedAreas = findOverlap(rasterImg, areasWithPolygons, writePath=writePath, imageFilename=channelNames, annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, kernel_size = kernel_size, kernel_sigma = kernel_sigma, normalize = normalize, writeCounter=writeCounter )
            
            overlapppedAreas.update(imOverlapppedAreas)
    allAreas = set(areasWithPolygons.keys())
    if allAreas.difference(overlapppedAreas):
        print(f'Warning: Could not find a raw image correspoinding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')

    return writeCounter



def rowColPolygons(areaDf, areaShape, profile, filename, outline, fill, kernel_size, kernel_sigma, gaussian = 0):
    """
    Convert polygons coordinates to image pixel coordinates, create annotation image using drawPolygons() and write the results into an image file.
    """
    transform = profile['transform']
    polygons = []
    polygon_anns = []
    for i in areaDf.index:
        # print(i)
        gm = areaDf.loc[i]['geometry']
        a, b = gm.centroid.x, gm.centroid.y
        row, col = rasterio.transform.rowcol(transform, a, b)
        zipped = list((row,col)) #[list(rc) for rc in list(zip(row,col))]
        # print(zipped)
        polygons.append(zipped)
        
        c,d = zip(*list(gm.exterior.coords))
        row2, col2 = rasterio.transform.rowcol(transform, c, d)
        zipped2 = list(zip(row2,col2)) #[list(rc) for rc in list(zip(row,col))]
        polygon_anns.append(zipped2)
    with open(filename, 'w') as outfile:  
        json.dump({'Trees': polygon_anns}, outfile)
    mask = drawPolygons_ann(polygon_anns,areaShape, outline=outline, fill=fill)
    # profile['dtype'] = rasterio.int16
    
    # # using eudlican distance mask for label smoothing
    # mask2 = ndimage.distance_transform_edt(mask)
    # using spatial varying label smoothing
    # mask2 = svls_2d(mask, kernel_size = kernel_size_svls, sigma = sigma_svls, channels=1)
    profile['dtype'] = rasterio.int16
    profile['compress'] = 'lzw'
    with rasterio.open(filename.replace('json', 'png'), 'w', **profile) as dst:
        # print('mask unique', np.unique(mask.astype(rasterio.int16)))
        # plt.figure()
        # plt.imshow(mask.astype(rasterio.int16))
        dst.write(mask.astype(rasterio.int16), 1)
        # dst.write(mask2.astype(rasterio.float32), 1)
    
    if gaussian: # create gussian kernels
        # if fixedKernel:
        print('****Using fixed kernel****')
        density_map=generate_density_map_with_fixed_kernel(areaShape,polygons, kernel_size=kernel_size, sigma = kernel_sigma)
        # elif not fixedKernel:
        # print('****Using k-nearest kernel****')
        # density_map=gaussian_filter_density(areaShape,polygons)
        # print(np.unique(density_map))
        profile['dtype'] = rasterio.float32
        with rasterio.open(filename.replace('json', 'png').replace('annotation', 'ann_kernel'), 'w', **profile) as dst:
            # print('mask unique', np.unique(mask.astype(rasterio.int16)))
            # plt.figure()
            # plt.imshow(mask.astype(rasterio.int16))
            dst.write(density_map.astype(rasterio.float32), 1)
    
    
    return 


def rowColPolygons_svls(areaDf, areaShape, profile, filename, outline, fill, kernel_size, kernel_sigma, kernel_size_svls, sigma_svls):
    """
    Convert polygons coordinates to image pixel coordinates, create annotation image using drawPolygons() and write the results into an image file.
    """
    transform = profile['transform']
    polygons = []
    polygon_anns = []
    for i in areaDf.index:
        # print(i)
        gm = areaDf.loc[i]['geometry']
        a, b = gm.centroid.x, gm.centroid.y
        row, col = rasterio.transform.rowcol(transform, a, b)
        zipped = list((row,col)) #[list(rc) for rc in list(zip(row,col))]
        # print(zipped)
        polygons.append(zipped)
        
        c,d = zip(*list(gm.exterior.coords))
        row2, col2 = rasterio.transform.rowcol(transform, c, d)
        zipped2 = list(zip(row2,col2)) #[list(rc) for rc in list(zip(row,col))]
        polygon_anns.append(zipped2)
    with open(filename, 'w') as outfile:  
        json.dump({'Trees': polygon_anns}, outfile)
    mask = drawPolygons_ann(polygon_anns,areaShape, outline=outline, fill=fill)
    # profile['dtype'] = rasterio.int16
    
    # # using eudlican distance mask for label smoothing
    # mask2 = ndimage.distance_transform_edt(mask)
    # using spatial varying label smoothing
    mask2 = svls_2d(mask, kernel_size = kernel_size_svls, sigma = sigma_svls, channels=1)
    profile['dtype'] = rasterio.float32
    with rasterio.open(filename.replace('json', 'png'), 'w', **profile) as dst:
        # print('mask unique', np.unique(mask.astype(rasterio.int16)))
        # plt.figure()
        # plt.imshow(mask.astype(rasterio.int16))
        # dst.write(mask.astype(rasterio.int16), 1)
        dst.write(mask2.astype(rasterio.float32), 1)
    
    # if fixedKernel:
    print('****Using fixed kernel****')
    density_map=generate_density_map_with_fixed_kernel(areaShape,polygons, kernel_size=kernel_size, sigma = kernel_sigma)
    # elif not fixedKernel:
    # print('****Using k-nearest kernel****')
    # density_map=gaussian_filter_density(areaShape,polygons)
    # print(np.unique(density_map))
    profile['dtype'] = rasterio.float32
    with rasterio.open(filename.replace('json', 'png').replace('annotation', 'ann_kernel'), 'w', **profile) as dst:
        # print('mask unique', np.unique(mask.astype(rasterio.int16)))
        # plt.figure()
        # plt.imshow(mask.astype(rasterio.int16))
        dst.write(density_map.astype(rasterio.float32), 1)
    
    
    return 




def generate_density_map_with_fixed_kernel(shape,points,kernel_size=11,sigma=3.5):
    '''
    img: input image.
    points: annotated pedestrian's position like [row,col]
    kernel_size: the fixed size of gaussian kernel, must be odd number.
    sigma: the sigma of gaussian kernel.
    return:
    d_map: density-map we want
    '''
    def guassian_kernel(size,sigma):
        rows=size[0] # mind that size must be odd number.
        cols=size[1]
        mean_x=int((rows-1)/2)
        mean_y=int((cols-1)/2)

        f=np.zeros(size)
        for x in range(0,rows):
            for y in range(0,cols):
                mean_x2=(x-mean_x)*(x-mean_x)
                mean_y2=(y-mean_y)*(y-mean_y)
                f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
        return f

    
    [rows,cols]=shape[0], shape[1]
    d_map=np.zeros([rows,cols])
    print('Using kernel size ', kernel_size)
    print('Using kernel sigma ', sigma)
    f=guassian_kernel([kernel_size,kernel_size],sigma) # generate gaussian kernel with fixed size.

    normed_f=(1.0/f.sum())*f # normalization for each head.

    # print('total points', len(points))
    if len(points)==0:
        return d_map
    else:
        for p in points:
            r,c=int(p[0]),int(p[1])
            if r>=rows or c>=cols:
                # print('larger')
                continue
            
            ##############3
            # if r < 0 or c < 0:
            #     print('negative ro col', r, c)
            ##############
            
            
            
            
            for x in range(0,f.shape[0]):
                for y in range(0,f.shape[1]):
                    if x+((r+1)-int((f.shape[0]-1)/2))<0 or x+((r+1)-int((f.shape[0]-1)/2))>rows-1 \
                    or y+((c+1)-int((f.shape[1]-1)/2))<0 or y+((c+1)-int((f.shape[1]-1)/2))>cols-1:
                        continue
                        # print('skipping cases')
                    else:
                        d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))]+=normed_f[x,y]
    # print('density summation', d_map.sum())
    return d_map




from scipy import spatial
from scipy.ndimage.filters import gaussian_filter

def gaussian_filter_density(shape,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[shape[0],shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    # leafsize = 2048
    leafsize = 20
    # build kdtree
    tree = spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        
        if 0<int(pt[0])<img_shape[0] and 0<int(pt[1])<img_shape[1]:
            pt2d[int(pt[0]),int(pt[1])] = 1.
            
        else:
            continue
        if gt_count >= 4:
            sigma = ((distances[i][1]+distances[i][2]+distances[i][3])/3)*0.3
        elif gt_count == 3 :
            sigma = ((distances[i][1]+distances[i][2])/2)*0.3
        elif gt_count == 2:
            sigma = (distances[i][1])*0.3
            # sigma = distances[i][1]
        else:
            # count = 1
            print('**************only one point!**************')
            sigma = np.average(np.array(img_shape))/2./2. #case: 1 point
        f = gaussian_filter(pt2d, sigma, mode='nearest')
        normed_f=(1.0/f.sum())*f 
        density += normed_f
        
    print('************SUM of density map***********', density.sum())
    # print ('done.')
    return density

def get_svls_filter_3d(kernel_size=3, sigma=1, channels=4):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    # print(x_coord)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    # print(x_grid_2d.shape)
    x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    # print(y_grid.shape)
    z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    # print(z_grid.shape)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    # print(xyz_grid.shape)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1,1]
    gaussian_kernel[1,1,1] = neighbors_sum
    svls_kernel_3d = gaussian_kernel / neighbors_sum
    # print(gaussian_kernel.shape)
    # Reshape to 3d depthwise convolutional weight
    svls_kernel_3d = svls_kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    # print(svls_kernel_3d.shape)
    svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
    svls_filter_3d = torch.nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=0)
    svls_filter_3d.weight.data = svls_kernel_3d
    svls_filter_3d.weight.requires_grad = False 
    return svls_filter_3d, svls_kernel_3d[0]

def svls_3d(label, kernel_size=3, sigma=1, channels=1):
    b, c, d, h, w = 1, 1, 1, np.squeeze(label).shape[0], np.squeeze(label).shape[1]
    # print(b, c, d, h, w)
    x = label[np.newaxis, np.newaxis, np.newaxis,:,:]
    x = torch.tensor(x)
    x = x.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
    x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
    print('sv label smoothing with kernel and sigma', kernel_size, sigma)
    svls_layer, svls_kernel = get_svls_filter_3d(kernel_size, sigma, channels)
    svls_labels = svls_layer(x)/svls_kernel.sum()
    y = svls_labels.numpy()
    w = np.squeeze(y)
    return w

def get_svls_filter_2d(kernel_size=3, sigma=1, channels=4): # kernel size should be odd number
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    # print(x_coord)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    # print('x2', x_grid_2d.shape)
    # 3d
    # x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    # print('x', x_grid.shape)
    y_grid_2d = x_grid_2d.t()
    
    # y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    # print('y', x_grid.shape)
    # z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    xy_grid = torch.stack([x_grid_2d, y_grid_2d], dim=-1).float()
    # print('xygrid', xy_grid.shape)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # print('g1', gaussian_kernel.shape)
    # neighbors_sum = 1 - gaussian_kernel[1,1]
    # cent = int((kernel_size - 1) / 2) # center pixel location (same as mean)
    cent = int((kernel_size - 1) / 2) # center pixel location (same as mean)
    neighbors_sum = 1 - gaussian_kernel[cent+1,cent+1]
    gaussian_kernel[cent+1,cent+1] = neighbors_sum 
    svls_kernel_3d = gaussian_kernel / neighbors_sum
    # np.save('kern.npy', svls_kernel_3d)
    # print(svls_kernel_3d.shape)
    # Reshape to 3d depthwise convolutional weight
    svls_kernel_3d = svls_kernel_3d.view(1, 1, kernel_size, kernel_size)
    svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1)
    # svls_filter_3d = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
    #                             kernel_size=kernel_size, groups=channels,
    #                             bias=False, padding=0)
    svls_filter_3d = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=1, padding_mode = 'replicate')
    svls_filter_3d.weight.data = svls_kernel_3d
    svls_filter_3d.weight.requires_grad = False 
    # np.save('kern2.npy', svls_kernel_3d)
    return svls_filter_3d, svls_kernel_3d[0]

def svls_2d(label, kernel_size=3, sigma=1, channels=1):
    import torch
    import torch.nn.functional as F
    import math
    b, c, h, w = 1, 1, np.squeeze(label).shape[0], np.squeeze(label).shape[1]
    # print(b, c, h, w)
    x = label[np.newaxis, np.newaxis,:,:]
    x = torch.tensor(x)
    # print('x shape',x.shape)
    x = x.view(b, c, h, w).repeat(1, 1, 1, 1).float()
    # print('x shape',x.shape)
    cent = int((kernel_size - 1) / 2)
    x = F.pad(x, (cent,cent, cent, cent), mode='replicate')
    # print('x shape',x.shape)
    print('sv label smoothing with kernel and sigma', kernel_size, sigma)
    svls_layer, svls_kernel = get_svls_filter_2d(kernel_size, sigma, channels)
    svls_labels = svls_layer(x)/svls_kernel.sum()
    y = svls_labels.numpy()
    w = np.squeeze(y)
    return w

