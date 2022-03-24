#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:23:25 2021

@author: sizhuo
"""

import glob
import os
from tqdm import tqdm
import rasterio
from osgeo import ogr, gdal
import multiprocessing
from itertools import product
from pathlib import Path
import math
from shapely.geometry import shape
from rasterio.windows import Window
import geopandas as gps
from rasterio.features import shapes
import time
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
        raster_fps = glob.glob(f"{raster_dir}/det_1km*.tif")
        
        raster_chms = [i.replace('det_seg', 'det_CHM') for i in raster_fps]
        
    else:
        # for FI
        raster_fps = glob.glob(f"{raster_dir}/*det_seg.tif")
        raster_chms = [os.path.join(raster_dir, 'det_chm_' + os.path.basename(i).replace('_det_seg', '')) for i in raster_fps]
    print('seg masks for polygonization:', raster_fps)
    print('chm masks:', raster_chms)
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
                chunk_windows.append([raster_fps[ind], raster_chms[ind], out_fp, Window(width * j, height * i, width, height)])

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
    merged_vrt_fp = os.path.join(polygons_basedir, f"all_polygons_{os.path.basename(postprocessing_dir)}.vrt")
    create_vector_vrt(merged_vrt_fp, glob.glob(f"{polygons_basedir}/*/*.vrt"))

    return
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
        return print(f"Warning! Attempt to create empty VRT file, skipping: {out_fp}")

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

    raster_fp, raster_chm, out_fp, window = params
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

d = '/mnt/ssdc/Denmark/summer_predictions/segcount/8101_complex5_0629/'

all_dirs = glob.glob(f"{d}/det*.tif*")


    
polygons_dir = os.path.join(d, "polygons")

# if not os.path.exists(polygons_dir):

create_polygons(d, polygons_dir, d, postproc_gridsize = (2, 2), postproc_workers = 40)
    


ds = '/media/RS_storage/Aerial/Denmark/predictions/segcount/'
all_dirs3 = glob.glob(f"{ds}/85*/")
all_dirs = all_dirs3 + all_dirs4 + all_dirs5
for di in all_dirs:
    
    polygons_dir = os.path.join(di, "polygons")
    
    if not os.path.exists(polygons_dir):
    
        create_polygons(di, polygons_dir, di, postproc_gridsize = (2, 2), postproc_workers = 40)

        time.sleep(5)
        
        cmd = 'ogrmerge.py -f GPKG -single -o ' + di + os.path.basename(di[:-1]) +'_pred_det.gpkg ' + di +'polygons/*/*.gpkg'
        os.system(cmd)
        
        time.sleep(10)






fs = glob.glob(f"{polygons_dir}/**/*.gpkg")

fs

cmd = 'echo ' + d + os.path.basename(d[:-1]) +'_pred_det.gpkg '
os.system(cmd)

cmd = 'ogrmerge.py -f GPKG -single -o ' + d + os.path.basename(d[:-1]) +'_pred_det.gpkg ' + d +'polygons/*/*.gpkg'
os.system(cmd)
