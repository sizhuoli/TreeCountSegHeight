# all functions for running deep learning prediction
import glob
import os
from pathlib import Path

import hydra
import requests

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import windows
import geopandas as gps
import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from itertools import product
import tensorflow as tf
import ipdb
from .losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from .UNet_attention_segcount import UNet
from .optimizers import adagrad, adam
from .frame_info import image_normalize
# from tensorflow import keras
import tensorflow.keras.backend as K
# mute warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


class ModelManager:
    def __init__(self):
        # download models from link
        self.trained_model_path = None
        self.model = None
        self.models = {}
        self.all_files = None
        self.model_url = 'https://sid.erda.dk/share_redirect/gS7JX84yvu'
        self.model_arch_channel = {'rgb': 3, 'rgbNir': 4, 'rgbNirNdvi': 5}
        self.calculate_ndvi = False
        self.tt = 0


    def load_files(self, config):

        self.all_files = glob.glob(config.general.input_image_dir + config.general.input_image_pref + '*' + config.general.input_image_type)
        print('Number of raw tif to predict:', len(self.all_files))
        print('=============================')


    def load_model(self, config):

        if config.predict.multires:
            from .UNet_multires_attention_segcount import UNet
        elif not config.predict.multires:
            from .UNet_attention_segcount import UNet
        # load models based on name
        for model_name, model_path in self.trained_model_path.items():
            self.model = UNet([config.predict.BATCH_SIZE, config.predict.input_size, config.predict.input_size, self.model_arch_channel[model_name]], inputBN=config.predict.inputBN)
            self.model.load_weights(model_path)
            self.model.compile(optimizer=adam, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            # prediction mode
            self.model.trainable = False
            self.models[model_name] = self.model


        print('{} models loaded with weights'.format(len(self.models)))
        print('=============================')


    def download_weights(self, config):

        if not os.path.exists(config.predict.model_save_path):
            os.makedirs(config.predict.model_save_path, exist_ok=True)
            # permisson
            os.chmod(config.predict.model_save_path, 0o777)

        weights_download_Path = config.predict.model_save_path + 'weights.zip'
        if not os.path.exists(weights_download_Path) and not len(glob.glob(config.predict.model_save_path + '**/*.h5', recursive=True)) > 0:
            print('Downloading model weights from:', self.model_url)
            import wget
            wget.download(self.model_url, weights_download_Path)


        if len(glob.glob(config.predict.model_save_path + '**/*.h5', recursive=True)) == 0:
            # unzip the file
            import zipfile
            print('Unzipping model weights:', weights_download_Path)
            with zipfile.ZipFile(weights_download_Path, 'r') as zip_ref:
                zip_ref.extractall(config.predict.model_save_path)

        # list holding the paths to models in use
        self.trained_model_path = dict()
        for model_name in config.predict.models_in_use:
            if model_name == 'rgb':
                self.trained_model_path[model_name] = glob.glob(config.predict.model_save_path + '**/*_redgreenblue_*.h5', recursive=True)[0]
            elif model_name == 'rgbNir':
                self.trained_model_path[model_name] = glob.glob(config.predict.model_save_path + '**/*_redgreenblueinfrared_*.h5', recursive=True)[0]
            elif model_name == 'rgbNirNdvi':
                self.trained_model_path[model_name] = glob.glob(config.predict.model_save_path + '**/*_redgreenblueinfraredndvi_*.h5', recursive=True)[0]
            else:
                print('Model name not recognized:', model_name)
                print('Supported model names: rgb, rgbNir, rgbNirNdvi; please open a new issue if you need support for other models')
                raise NotImplementedError

        print('Model weights:', self.trained_model_path)
        print('=============================')
        #



    def predict(self, config):

        if not os.path.exists(config.general.output_dir):
            os.makedirs(config.general.output_dir)
            # permisson
            os.chmod(config.general.output_dir, 0o777)

        # assert model and bands are consistent
        assert 'red' in config.general.channels.keys() and 'green' in config.general.channels.keys() and 'blue' in config.general.channels.keys(), 'red, green, blue bands must exist; please open a new issue if you need support for fewer bands'
        # ipdb.set_trace()
        assert config.general.channels['infrared'] != 'None' if any('Nir' in s or 'Ndvi' in s for s in
                                                                    config.predict.models_in_use) else True, 'infrared band must exist if rgbNir or rgbNirNdvi model is called'
        # prepare image dimension, number of channels with index not equal to None
        self.img_dim = len(config.general.channels.keys()) - len(
            [i for i in config.general.channels.values() if i == 'None'])
        # in case red and nir exist and nir model is called, calculate ndvi
        if config.general.channels['red'] != 'None' and config.general.channels['infrared'] != 'None' and \
                any('Ndvi' in s or 'Nir' in s for s in config.predict.models_in_use):
            self.img_dim += 1
            self.calculate_ndvi = True

        counter, not_work = self.predict_ready_run(config)
        print(f"Predicted {counter} images.")
        print(f"Failed to predict {len(not_work)} images. See below:")
        print(not_work)








    def addTOResult(self, res, prediction, row, col, he, wi, operator = 'MAX'):
        currValue = res[row:row+he, col:col+wi]
        newPredictions = prediction[:he, :wi]
        # # set the 4 borderlines to 0 to remove the border effect
        # newPredictions[:10, :] = 0
        # newPredictions[-10:, :] = 0
        # newPredictions[:, :10] = 0
        # newPredictions[:, -10:] = 0

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

    def predict_using_model_segcount_fi(self, config, models, batch, batch_pos, maskseg, maskdens, operator):
        # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
        # b1 = batch[0]
        tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
        # predict using corresponding models
        # placeholder for all results, to be averaged later
        seg = []
        dens = []

        for model_name, model in models.items():
            if model_name == 'rgb':
                segi, densi = model.predict(tm1[..., [config.general.channels['red'], config.general.channels['green'], config.general.channels['blue']]], workers = config.predict.workers, use_multiprocessing = True, verbose=0)
                seg.append(segi)
                dens.append(densi)
            elif model_name == 'rgbNir':
                segi, densi = model.predict(tm1[..., [config.general.channels['red'], config.general.channels['green'], config.general.channels['blue'], config.general.channels['infrared']]], workers = config.predict.workers, use_multiprocessing = True, verbose=0)
                seg.append(segi)
                dens.append(densi)
            elif model_name == 'rgbNirNdvi':
                segi, densi = model.predict(tm1[..., [config.general.channels['red'], config.general.channels['green'], config.general.channels['blue'], config.general.channels['infrared'], -1]], workers = config.predict.workers, use_multiprocessing = True, verbose=0)
                seg.append(segi)
                dens.append(densi)
        seg = np.nanmean(seg, axis = 0)
        dens = np.nanmean(dens, axis = 0)
        # ipdb.set_trace()
        # # rgb model
        # seg0, dens0 = models[0].predict(tm1[..., [1, 2, 3]], workers = 10, use_multiprocessing = True, verbose=0)
        # # rgb+infrared model
        # seg1, dens1 = models[1].predict(tm1[..., :4], workers = 10, use_multiprocessing = True, verbose=0)
        # # rgb+infrared+ndvi model
        # seg2, dens2 = models[2].predict(tm1, workers = 10, use_multiprocessing = True, verbose=0)
        # # merge the results
        # # nan sum to deal with nan values from ndvi calculation
        # seg = np.nanmean([seg0, seg1, seg2], axis = 0)
        # dens = np.nanmean([dens0, dens1, dens2], axis = 0)
        # ipdb.set_trace()
        for i in range(len(batch_pos)):
            (col, row, wi, he) = batch_pos[i]
            p = np.squeeze(seg[i], axis = -1)
            c = np.squeeze(dens[i], axis = -1)
            # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
            maskseg = self.addTOResult(maskseg, p, row, col, he, wi, operator)
            maskdens = self.addTOResult(maskdens, c, row, col, he, wi, operator)
        # ipdb.set_trace()
        return maskseg, maskdens


    def detect_tree_segcount_fi(self, config, models, img):
        if 'chm' in config.general.channels.keys():
            raise NotImplementedError('not supporting chm as input yet')
        else:
            CHM = 0
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()

        if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
            meta['dtype'] = np.float32

        offsets = product(range(0, nols, config.predict.STRIDE), range(0, nrows, config.predict.STRIDE))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

        masksegs = np.zeros((int(nrows), int(nols)), dtype=np.float32)
        maskdenss = np.zeros((int(nrows), int(nols)), dtype=np.float32)
        meta.update(
                    {'width': int(nols),
                     'height': int(nrows)
                    }
                    )


        batch = []
        batch_pos = [ ]
        for col_off, row_off in tqdm(offsets):
            window =windows.Window(col_off=col_off, row_off=row_off, width=config.predict.WIDTH, height=config.predict.HEIGHT).intersection(big_window)
            # prepare for all bands + ndvi (if available)
            patch1 = np.zeros((config.predict.HEIGHT, config.predict.WIDTH, self.img_dim), dtype=np.float32)
            temp_im1 = img.read(window = window)
            temp_im1 = np.transpose(temp_im1, axes=(1,2,0))
            if self.calculate_ndvi:
                NDVI = (temp_im1[..., config.general.channels['infrared']].astype(float) - temp_im1[..., config.general.channels['red']].astype(float)) \
                       / (temp_im1[..., config.general.channels['infrared']].astype(float) + temp_im1[..., config.general.channels['red']].astype(float))
                NDVI = NDVI[..., np.newaxis]

                temp_im1 = np.append(temp_im1, NDVI, axis = -1)


            if config.predict.normalize:
                temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel

            patch1[:int(window.height), :int(window.width)] = temp_im1

            batch.append(patch1)
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
            if (len(batch) == config.predict.BATCH_SIZE):
                # print('processing one batch')
                masksegs, maskdenss = self.predict_using_model_segcount_fi(config, models, batch, batch_pos, masksegs, maskdenss, 'MAX')

                batch = []
                batch_pos = []

        if batch:
            masksegs, maskdenss = self.predict_using_model_segcount_fi(config, models, batch, batch_pos, masksegs, maskdenss, 'MAX')
            batch = []
            batch_pos = []
        return masksegs, maskdenss, meta



    def predict_ready_run(self, config):
        counter = 0
        not_work = []
        # # shuffle the files if testing only a subset
        # random.shuffle(all_files)
        for fullPath in tqdm(self.all_files):
            outputSeg = os.path.join(config.general.output_dir, os.path.basename(fullPath).replace(
                config.general.input_image_type,
                config.general.output_suffix_seg + config.general.output_image_type))
            if not os.path.exists(outputSeg):

                with rasterio.open(fullPath) as img:
                    if config.predict.segcountpred:
                        detectedMaskSeg, detectedMaskDens, detectedMeta = self.detect_tree_segcount_fi(config, self.models, img)
                        self.writeMaskToDisk(detectedMaskSeg, detectedMeta, outputSeg,
                                             write_as_type = config.general.output_dtype_seg, th = config.predict.threshold)

                        # density
                        outputDensity = os.path.join(config.general.output_dir, os.path.basename(fullPath).replace(
                            config.general.input_image_type,
                            config.general.output_suffix_density + config.general.output_image_type))
                        self.writeMaskToDisk(detectedMaskDens, detectedMeta, outputDensity,
                                             write_as_type = config.general.output_dtype_density, th = config.predict.threshold, convert = 0)
                        del detectedMaskSeg, detectedMaskDens, detectedMeta
                        # clear memory
                        K.clear_session()

                    else:
                        continue
                # except:
                #     not_work.append(fullPath)
                #     continue

                counter += 1


            else:
                print('Skipping: File already analysed!', fullPath)

        return counter, not_work






    def writeMaskToDisk(self, detected_mask, detected_meta, wp, write_as_type = 'uint8', th = 0.5, convert = 1, rescale = 0):
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
