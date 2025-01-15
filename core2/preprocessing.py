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
from typing import List, Tuple, Dict
import numpy as np
import rasterio 
import geopandas as gpd
from shapely.geometry import box
from PIL import Image, ImageDraw
from tqdm import tqdm
from rasterio.mask import mask
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter
from scipy import spatial
from warnings import warn

from core2.frame_info import image_normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logging


class Processor:

    def __init__(self, config, boundary:bool = False, aux:bool = False):
        self.config = config
        self.boundary = boundary
        self.aux = aux
        self.training_polygons, self.training_areas = self._load_polygons()
        self.input_images = self._read_input_images()
        logging.info(f"Found {len(self.input_images)} input image(s) to process.")

        self.training_sets = self._assign_training_polygons_to_areas()
        # training_sets is a Dict[int, Dict] of {training_area: {training_polygons with pramaters}} 
        logging.info(f"Assigned {len(self.training_polygons)} training polygons into {len(self.training_sets)} training areas")

        if not self.boundary:
            self.config.extracted_boundary_filename = None
        if not self.aux:
            self.config.aux_channel_prefixes = None
            self.config.aux_bands = None

    def extract_training_sets(self):
        write_counter = 0
        training_areas_in_imagery = set()

        # Extract raster areas that overlap with traning area.
        if self.aux and not self.config.single_raster:
            logging.info("Processing multi-raster images with auxiliary data...")
            for imgs in tqdm(self.input_images):
                main_image = rasterio.open(imgs[0]) # What is a main image?
                ncimg, img_overlap_areas = self._find_overlap(main_image, write_counter)

                for aux_idx, aux_channel_name in enumerate(self.config.aux_channel_prefixes):
                    aux_image = rasterio.open(imgs[aux_idx + 1])
                    ncaux, aux_overlap_areas = self._find_overlap(aux_image, write_counter)
                    
                if ncimg != ncaux:
                    logging.info(f"Error: Mismatched masks between main and auxiliary images.\n"
                                 f"ncimg = {ncimg}\n"
                                 f"ncauxi = {ncaux}")
                    break

                training_areas_in_imagery.update(img_overlap_areas)

        else:
            logging.info("Processing single-raster images or multi-raster images without auxiliary data...")
            for imgs in tqdm(self.input_images):
                raster_img = rasterio.open(imgs)
                write_counter, img_overlap_areas = self._find_overlap(raster_img, write_counter)

                training_areas_in_imagery.update(img_overlap_areas)

        # Check if all training areas have corresponding imageries
        all_training_areas = set(self.training_sets.keys())
        training_areas_without_imagery = all_training_areas.difference(training_areas_in_imagery)
        if training_areas_without_imagery:
            warn(f"Missing corresponding imageries for training areas: {training_areas_without_imagery}")

    def _load_polygons(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Load training areas and polygons, ensuring CRS compatibility."""
        training_areas = gpd.read_file(os.path.join(self.config.training_base_dir, self.config.training_area_fn))
        training_polygons = gpd.read_file(os.path.join(self.config.training_base_dir, self.config.training_polygon_fn))
        
        if training_areas.crs  != training_polygons.crs:
            logging.warning('Training area CRS does not match training_polygon CRS. Reprojecting..."')
            training_areas = training_areas.to_crs(training_polygons.crs)

        training_areas['id'] = range(training_areas.shape[0])
        return training_polygons, training_areas

    def _read_input_images(self) -> List[str]:
        """Read input images based on configuration."""
        image_paths = []
        for root, _, files in os.walk(self.config.raw_image_base_dir):
            for file in files:
                if file.endswith(self.config.raw_image_file_type) and not file.startswith(self.config.prediction_pre):
                    if self.aux:
                        aux_files = [os.path.join(root, file.replace(self.config.raw_image_prefix, aux_file))
                                     for aux_file in self.config.aux_channel_prefixes]
                        image_paths.append(os.path.join(root, file), *aux_files)
                    else:
                        image_paths.append(os.path.join(root, file))
        return image_paths

    def _assign_training_polygons_to_areas(self) -> Dict[int, Dict]:
        """
        Determine the parent training area for each polygon and generate a weight map based on the distance of a polygon boundary to other objects.
        Weight map will be used by the weighted loss during the U-Net training
        """
        polygons_copy = self.training_polygons.copy()
        training_sets = {}

        for _, area in self.training_areas.iterrows():
            polygons_in_area = polygons_copy[polygons_copy.intersects(area.geometry)].copy()
            polygons_copy = polygons_copy[~polygons_copy.index.isin(polygons_in_area.index)]
            boundary_weight = self._calculate_boundary_weight(polygons_in_area) if self.boundary else None
            training_sets[area.id] = {'polygons': polygons_in_area, 
                                       'boundary_weight': boundary_weight,
                                       'bounds': area.geometry.bounds
                                       }

        return training_sets
    
    @staticmethod
    def _calculate_boundary_weight(polygons: gpd.GeoDataFrame, scale: float = 1.5) -> gpd.GeoDataFrame:
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
   
    def _find_overlap(self, img, write_counter):
        """
        Identifies and processes training areas that intersect with the given imagery.

        This method checks whether the bounding boxes of the training areas intersect with the bounding box of the raw image.
        If an intersection exists, it calls `_write_extrated_image_and_annotation()` to extract and save the overlapping imagery
        and annotations within the training area.

        Note:
            This method is necessary but may not be sufficient for ensuring complete overlap between training areas and the image.

        Args:
            img: The input image (as a rasterio object) to check for overlaps with the training areas.
            write_counter (int): A counter used to track the number of images and annotations written to output.

        Returns:
            tuple:
                - write_counter (int): Updated counter after processing the overlapping areas.
                - img_overlap_areas (set): A set of area IDs that overlap with the image.
        """
        img_overlap_areas = set()

        for area_id, area_info in tqdm(self.training_sets.items()):
            polygons_in_area = gpd.GeoDataFrame(area_info['polygons'])
            boundaries_df = gpd.GeoDataFrame(area_info.get('boundary_weight')) if self.boundary else None

            area_bbox = box(*area_info['bounds'])
            img_bbox = box(*img.bounds)

            if not area_bbox.intersects(img_bbox):
                continue

            profile = img.profile
            overlap, transform = mask(img, [area_bbox], all_touched=True, crop=True)

            profile.update({
                "height": overlap.shape[1],
                "width": overlap.shape[2],
                "transform": transform,
                "compress": 'lzw',
                "dtype": rasterio.float32
                })
            
            write_counter = self._write_extrated_image_and_annotation(overlap, profile, polygons_in_area, boundaries_df, write_counter)

            img_overlap_areas.add(area_id)

        return write_counter, img_overlap_areas

    def _write_extrated_image_and_annotation(self, overlap, profile, polygons_in_area, boundaries_df, write_counter):
        """
        Clip the imagery of the training area into separate image files.
        Call polygon_to_pixel() to genearte annotation and boundary image from training polygons.
        """
        for band, band_name in zip(self.config.bands, self.config.extracted_filenames):
            data = overlap[band].astype(profile['dtype'])
            data = image_normalize(data, axis=None) if self.config.normalize else data
            output_path = os.path.join(self.config.path_to_write, f"{band_name}_{write_counter}.{self.config.extracted_file_type}")

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)

            annotation_path = os.path.join(self.config.path_to_write, f"{self.config.extracted_annotation_filename}_{write_counter}.json")
            polygon_to_pixel(polygons_in_area, overlap.shape[1:], profile, annotation_path, gaussian=True)

        if self.boundary:
            boundary_path = os.path.join(self.config.path_to_write, f"{self.config.extracted_boundary_filename}_{write_counter}.json")
            polygon_to_pixel(boundaries_df, overlap.shape[1:], profile, boundary_path)

        return write_counter + 1

    def extract_svls(self, boundary = 0, aux = 0):
        if not boundary:
            # no boundar weights
            writeCounter=0
            # multi raster with aux
            writeCounter = extractAreasThatOverlapWithTrainingData_svls(self.inputImages, self.areasWithPolygons, self.config.path_to_write, self.config.extracted_filenames,  self.config.extracted_annotation_filename, None, self.config.bands , writeCounter, self.config.aux_channel_prefixs, self.config.aux_bands,  self.config.single_raster, kernel_size = 15, kernel_sigma = 4, kernel_size_svls = self.config.kernel_size_svls, sigma_svls = self.config.kernel_sigma_svls)


def polygon_to_pixel(area_df, area_shape, profile, filename, outline=0, fill=1, kernel_size=15, kernel_sigma=5, gaussian=False):
    """
    Convert polygon coordinates to pixel coordinates, create annotation and mask images,
    and optionally generate Gaussian kernels.
    """
    polygons_in_pixels = []
    annotations = []
    transform = profile['transform']

    for idx, row in area_df.iterrows():
        geom = row['geometry']

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            warn(f"Training data {idx} is a MultiPolygon. Please Check")
            polygons = list(geom.geoms)
        else:
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

        for polygon in polygons:
            # Convert centroid to pixel coordinates
            centroid_x, centroid_y = polygon.centroid.x, polygon.centroid.y
            centroid_row, centroid_col = rowcol(transform, centroid_x, centroid_y)
            # Convert exterior coordinates to pixel coordinates
            exterior_coords = np.array(polygon.exterior.coords)
            if exterior_coords.shape[1] == 3:  # Handle 3D coordinates
                exterior_coords = exterior_coords[:, :2]
            
            row_coords, col_coords = rowcol(transform, exterior_coords[:, 0], exterior_coords[:, 1])
            row_coords = [int(r) for r in row_coords]
            col_coords = [int(c) for c in col_coords]
            polygons_in_pixels.append([centroid_row, centroid_col])
            annotations.append(list(zip(row_coords, col_coords)))
     
    with open(filename, 'w') as outfile:  
        json.dump({'Trees': annotations}, outfile)
    
    # create mask from polygons
    mask = np.zeros(area_shape, dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)
    for annotation in annotations:
        xy = [(coord[1], coord[0]) for coord in annotation]  # Convert (x, y) to (row, col)
        draw.polygon(xy, outline=outline, fill=fill)    

    mask = np.array(pil_mask)
    
    # Write mask to a PNG file
    profile['dtype'] = rasterio.int16
    mask_filepath = filename.replace('.json', '.png')
    with rasterio.open(mask_filepath, 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.int16), 1)
    
    if gaussian: # Generate Gaussian kernel density map
        density_map = generate_gaussian_density_map(area_shape, polygons_in_pixels, kernel_size, kernel_sigma)
        profile['dtype'] = rasterio.float32
        kernel_filepath = filename.replace('.json', '.png').replace('annotation', 'ann_kernel')
        with rasterio.open(kernel_filepath, 'w', **profile) as dst:
            dst.write(density_map.astype(rasterio.float32), 1)


def create_polygon_mask(polygons: List[List[Tuple[float, float]]], shape: Tuple[int, int], outline: int = 0, fill: int = 1) -> np.ndarray:
    """
    Creates a NumPy mask from polygons with specified outline and fill values.
    
    Args:
        polygons (List[List[Tuple[float, float]]]): List of polygons where each polygon is a list of (x, y) coordinates.
        shape (Tuple[int, int]): Shape of the mask (height, width).
        outline (int): Value to assign to the outline (edge) of the polygon.
        fill (int): Value to assign to the interior (fill) of the polygon.

    Returns:
        np.ndarray: A NumPy array mask with the same dimensions as the specified shape.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)

    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]  # Convert!?
        draw.polygon(xy, outline=outline, fill=fill)
    return np.array(pil_mask)


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
        if 'boundary_weight' in areaInfo:
            boundariesInAreaDf = gpd.GeoDataFrame(areaInfo['boundary_weight'])
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
    # mask = draw_polygons(polygon_anns,areaShape, outline=outline, fill=fill)
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

    kernel_radius = kernel_size // 2
    gaussian_kernel = gaussian_filter(np.zeros((kernel_size, kernel_size)), sigma=sigma)
    gaussian_kernel /= gaussian_kernel.sum()

    for r, c in points:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            # Determine bounds for the kernel on the density map
            r_start = max(r - kernel_radius, 0)
            r_end = min(r + kernel_radius + 1, shape[0])
            c_start = max(c - kernel_radius, 0)
            c_end = min(c + kernel_radius + 1, shape[1])

            # Determine bounds for the kernel itself
            k_r_start = max(kernel_radius - r, 0)
            k_r_end = k_r_start + (r_end - r_start)
            k_c_start = max(kernel_radius - c, 0)
            k_c_end = k_c_start + (c_end - c_start)

            # Add the cropped kernel to the density map
            density_map[r_start:r_end, c_start:c_end] += gaussian_kernel[k_r_start:k_r_end, k_c_start:k_c_end]

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
