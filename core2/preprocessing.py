#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:51:14 2021

@author: sizhuo
"""
# Required configurations (including the input and output paths) are stored in a separate file (such as config/Preprocessing.py)
# Please provide required info in the file before continuing with this notebook. 

import os
import json
import math
import logging
from typing import List, Tuple, Dict, Union
import numpy as np
import rasterio 
import geopandas as gpd
from shapely.geometry import box, Point
from PIL import Image, ImageDraw
from tqdm import tqdm
import ipdb
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter

from core2.visualize import display_images
from core2.frame_info import image_normalize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Processor:
    """
    Handles preprocessing tasks including reading input data, processing polygons, and extracting image overlaps.
    """

    def __init__(self, config, boundary:bool = False, aux:bool = False):
        self.config = config
        self.training_polygons, self.training_areas = self.load_polygons()
        self.input_images = self.read_input_images(aux=aux)
        logging.info(f"Found {len(self.input_images)} input image(s) to process.")

        if boundary:
            self.areas_with_polygons = self.divide_polygons_in_training_areas(boundary=True)
        else:
            self.areas_with_polygons = self.divide_polygons_in_training_areas(boundary=False)

    def load_polygons(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Reads training polygons and areas from configuration paths.
        Ensures CRS compatibility and assigns IDs to training areas.
        """
        training_areas = gpd.read_file(os.path.join(self.config.training_base_dir, self.config.training_area_fn))
        training_polygons = gpd.read_file(os.path.join(self.config.training_base_dir, self.config.training_polygon_fn))
        
        if training_areas.crs  != training_polygons.crs:
            logging.warning('Training area CRS does not match training_polygon CRS. Reprojecting..."')
            training_areas = training_areas.to_crs(training_polygons.crs)

        training_areas['id'] = range(training_areas.shape[0])
        return training_polygons, training_areas

    def read_input_images(self, aux: bool = False) -> List[str]:
        """
        Reads input images from the base directory and filters based on file types and prefixes.
        """     
        input_images = []
        for root, _, files in os.walk(self.config.raw_image_base_dir):
            for file in files:
                if file.endswith(self.config.raw_image_file_type) and not file.startswith(self.config.prediction_pre):
                    if aux:
                        aux_files = [os.path.join(root, file.replace(self.config.raw_image_prefix, aux_file))
                                     for aux_file in self.config.aux_channel_prefixes]
                        input_images.append(os.path.join(root, file), *aux_files)
                    else:
                        input_images.append(os.path.join(root, file))
        return input_images

    def divide_polygons_in_training_areas(self, boundary: bool = False) -> Dict[int, Dict]:
        """
        Assigns polygons to training areas and optionally computes boundary weights.
        """
        polygons_copy = self.training_polygons.copy()
        split_polygons = {}

        for index, area in self.training_areas.iterrows():
            polygons_in_area = polygons_copy[polygons_copy.intersects(area.geometry)].copy()
            polygons_copy = polygons_copy[~polygons_copy.index.isin(polygons_in_area.index)]

            if boundary:
                boundaries = self.calculate_boundary_weight(polygons_in_area)
                split_polygons[area.id] = {'polygons': polygons_in_area, 'boundaryWeight': boundaries}
            else:
                split_polygons[area.id] = {'polygons': polygons_in_area}

        return split_polygons
    
    @staticmethod
    def calculate_boundary_weight(polygons: gpd.GeoDataFrame, scale: float = 1.5) -> gpd.GeoDataFrame:
        """
        Computes weighted boundaries for polygons based on proximity.
        """
        if polygons.empty:
            logging.info("No polygons to compute boundary weights.")
            return gpd.GeoDataFrame()

        scaled_polygons = polygons.copy()
        scaled_polygons['geometry'] = scaled_polygons.geometry.buffer(scale)
        boundaries = gpd.overlay(scaled_polygons, polygons, how='difference')
        return boundaries
    
    def extract_overlapping_areas(self):
        """
        Extracts areas of overlap between input images and training areas and saves the results.
        """
        for image_path in tqdm(self.input_images):
            with rasterio.open(image_path) as src:
                for area_id, area_info in self.areas_with_polygons.items():
                    self.process_overlap(src, area_id, area_info)  

    def process_overlap(self, src, area_id: int, area_info: Dict):
        """
        Processes overlapping regions between an image and a training area.
        """
        bbox = box(*src.bounds)
        area_bbox = box(*area_info['polygons'].total_bounds)

        if not bbox.intersects(area_bbox):
            return

        overlap, transform = mask(src, [area_bbox], crop=True)
        profile = src.profile.copy()
        profile.update({
            "height": overlap.shape[1],
            "width": overlap.shape[2],
            "transform": transform
        })

        # Save the overlapping region and related annotations
        self.save_overlapping_data(overlap, profile, area_info)     

    def save_overlapping_data(self, overlap, profile, area_info: Dict):
        """
        Saves the overlapping image data, annotations, and boundaries.
        """
        pass  # Implement saving logic (images, annotations, etc.)

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
         
        



# # Create boundary from polygon file
# def calculateBoundaryWeight(polygonsInArea, scale_polygon = 1.5, output_plot = True): 
#     '''
#     For each polygon, create a weighted boundary where the weights of shared/close boundaries is higher than weights of solitary boundaries.
#     '''
#     # If there are polygons in a area, the boundary polygons return an empty geo dataframe
#     if not polygonsInArea:
#         # print('No polygons')
#         return gpd.GeoDataFrame({})
#     tempPolygonDf = pd.DataFrame(polygonsInArea)
#     tempPolygonDf.reset_index(drop=True,inplace=True)
#     tempPolygonDf = gpd.GeoDataFrame(tempPolygonDf)
#     new_c = []
#     #for each polygon in area scale, compare with other polygons:
#     for i in tqdm(range(len(tempPolygonDf))):
#         pol1 = gpd.GeoSeries(tempPolygonDf.iloc[i]['geometry'])
#         sc = pol1.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
#         scc = pd.DataFrame(columns=['id', 'geometry'])
#         scc = scc._append({'id': None, 'geometry': sc[0]}, ignore_index=True)
#         scc = gpd.GeoDataFrame(pd.concat([scc]*len(tempPolygonDf), ignore_index=True))

#         pol2 = gpd.GeoDataFrame(tempPolygonDf[~tempPolygonDf.index.isin([i])])
#         #scale pol2 also and then intersect, so in the end no need for scale
#         pol2 = gpd.GeoDataFrame(pol2.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center'))
#         pol2.columns = ['geometry']

#         #invalid intersection operations topo error
#         try:
#             ints = scc.intersection(pol2)
#             for k in range(len(ints)):
#                 if ints.iloc[k]!=None:
#                     if ints.iloc[k].is_empty !=1:
#                         new_c.append(ints.iloc[k])
#         except:
#             print('Intersection error')
#     new_c = gpd.GeoSeries(new_c)
#     new_cc = gpd.GeoDataFrame({'geometry': new_c})
#     new_cc.columns = ['geometry']
    
#     # df may contains point other than polygons
#     new_cc['type'] = new_cc['geometry'].type 
#     new_cc = new_cc[new_cc['type'].isin(['Polygon', 'MultiPolygon'])]
#     new_cc.drop(columns=['type'])
    
#     tempPolygonDf['type'] = tempPolygonDf['geometry'].type 
#     tempPolygonDf = tempPolygonDf[tempPolygonDf['type'].isin(['Polygon', 'MultiPolygon'])]
#     tempPolygonDf.drop(columns=['type'])
#     # print('new_cc', new_cc.shape)
#     # print('tempPolygonDf', tempPolygonDf.shape)
#     if new_cc.shape[0] == 0:
#         print('No boundaries')
#         return gpd.GeoDataFrame({})
#     else:
#         bounda = gpd.overlay(new_cc, tempPolygonDf, how='difference')

#         if output_plot:
#             # fig, ax = plt.subplots(figsize = (10,10))
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
#             bounda.plot(ax=ax1,color = 'red')
#             tempPolygonDf.plot(alpha = 0.2,ax = ax1,color = 'b')
#             # plt.show()
#             ###########################################################

#             bounda.plot(ax=ax2,color = 'red')
#             plt.show()
#         #change multipolygon to polygon
#         bounda = bounda.explode()
#         bounda.reset_index(drop=True,inplace=True)
#         #bounda.to_file('boundary_ready_to_use.shp')
#         return bounda

# As input we received two shapefile, first one contains the training areas/rectangles and other contains the polygon of trees/objects in those training areas
# The first task is to determine the parent training area for each polygon and generate a weight map based upon the distance of a polygon boundary to other objects.
# Weight map will be used by the weighted loss during the U-Net training



def draw_polygons(polygons: List[List[Tuple[float, float]]], shape: Tuple[int, int], outline: int = 0, fill: int = 1) -> np.ndarray:
    """
    Creates a numpy mask from polygons with specified outline and fill values.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)

    for polygon in polygons:
        draw.polygon(polygon, outline=outline, fill=fill)
    return np.array(pil_mask)


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
        polygonsInAreaDf = gpd.GeoDataFrame(areaInfo['polygons'])
        # polygonsInAreaDf = polygonsInAreaDf.explode()
        if 'boundaryWeight' in areaInfo:
            boundariesInAreaDf = gpd.GeoDataFrame(areaInfo['boundaryWeight'])
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
        polygonsInAreaDf = gpd.GeoDataFrame(areaInfo['polygons'])
        if 'boundaryWeight' in areaInfo:
            boundariesInAreaDf = gpd.GeoDataFrame(areaInfo['boundaryWeight'])
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
    Convert polygons coordinates to image pixel coordinates, 
    create annotation image using drawPolygons() and write the results into an image file.
    """
    transform = profile['transform']
    polygons = []
    polygon_anns = []
 
    for i in areaDf.index:
        gm = areaDf.loc[i]['geometry']
        a, b = gm.centroid.x, gm.centroid.y
        row, col = rasterio.transform.rowcol(transform, a, b)
        polygons.append([row, col])

        coords = list(gm.exterior.coords)
        if len(coords[0]) == 2:  # 2D coordinates (x, y)
            c, d = zip(*coords)
        elif len(coords[0]) == 3:  # 3D coordinates (x, y, z)
            c, d, _ = zip(*coords)  # Ignore z values

        row2, col2 = rasterio.transform.rowcol(transform, c, d)
        polygon_anns.append(list(zip(row2, col2)))
     
    with open(filename, 'w') as outfile:  
        json.dump({'Trees': polygon_anns}, outfile)
     
    mask = draw_polygons(polygon_anns,areaShape, outline=outline, fill=fill)
    
    # # using eudlican distance mask for label smoothing
    # mask2 = ndimage.distance_transform_edt(mask)
    # using spatial varying label smoothing
    # mask2 = svls_2d(mask, kernel_size = kernel_size_svls, sigma = sigma_svls, channels=1)
    profile['dtype'] = rasterio.int16
    profile['compress'] = 'lzw'
    with rasterio.open(filename.replace('json', 'png'), 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.int16), 1)
    
    if gaussian: # create gussian kernels
        # if fixedKernel:
        print('****Using fixed kernel****')
        density_map=generate_gaussian_density_map(areaShape,polygons, kernel_size=kernel_size, sigma = kernel_sigma)
        # elif not fixedKernel:
        # print('****Using k-nearest kernel****')
        # density_map=gaussian_filter_density(areaShape,polygons)
        # print(np.unique(density_map))
        profile['dtype'] = rasterio.float32
        with rasterio.open(filename.replace('json', 'png').replace('annotation', 'ann_kernel'), 'w', **profile) as dst:
            dst.write(density_map.astype(rasterio.float32), 1)
         

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
    density_map=generate_gaussian_density_map(areaShape,polygons, kernel_size=kernel_size, sigma = kernel_sigma)
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


def generate_gaussian_density_map(shape: Tuple[int, int], points: List[Tuple[int, int]], kernel_size: int = 11, sigma: float = 3.5) -> np.ndarray:
    """
    Generates a Gaussian density map for given points.
    """
    density_map = np.zeros(shape, dtype=np.float32)
    gaussian_kernel = gaussian_filter(np.zeros((kernel_size, kernel_size)), sigma=sigma)
    gaussian_kernel /= gaussian_kernel.sum()

    for r, c in points:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            density_map[r:r + kernel_size, c:c + kernel_size] += gaussian_kernel

    return density_map


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
