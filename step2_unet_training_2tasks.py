#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 00:22:48 2021

@author: sizhuo
"""

# seg with counting (by regression)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu

from core2.training_segcount import trainer

from config import UNetTraining
config = UNetTraining.Configuration()

trainer_segcount = trainer(config)
trainer_segcount.vis()
trainer_segcount.train_config()
if 'complex' in config.model_name:
    print('complex model')
    trainer_segcount.LOAD_model()
# check seg loss!!!!!!!!!!!1
trainer_segcount.train()
trainer_segcount.train_retrain()
trainer_segcount.train_retrain_eff()














import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

    
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import imageio
import os
import time
import rasterio.warp             # Reproject raster samples
# from functools import reduce
from tensorflow.keras.models import load_model



from core.losses import tversky, accuracy, msle_seg, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.split_frames import split_dataset
from core.visualize import display_images

import json
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# %matplotlib inline
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

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
import tensorflow.keras.backend as K

from model.init import get_model_2inputs_2outputs, get_model_1input_2outputs
from model.base import load_model_2inputs_2outputs, compile_model_2inputs_2outputs
from model.base import load_model_1input_2outputs, compile_model_1input_2outputs


from config import UNetTraining
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
import scipy
print(tf.__version__)


print(tf.config.list_physical_devices('GPU'))


config = UNetTraining.Configuration()

# Read all images/frames into memory


all_files = os.listdir(config.base_dir)
# image channel 1
all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
print(all_files_c1)
from core.frame_info_segcount import FrameInfo
from core.dataset_generator_segcount import DataGenerator
frames = []
for i, fn in enumerate(all_files_c1):
    # loop through rectangles
    img1 = rasterio.open(os.path.join(config.base_dir, fn)).read()
    if config.single_raster or not config.aux_data:
        # print('If single raster or multi raster without aux')
        for c in range(config.image_channels1):
            #loop through raw channels
            img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.channel_names1[c]))).read(), axis = 0)
        
    else: # multi raster
        print('Multi raster with aux data')
        for c in range(config.image_channels1-1): 
                #loop through raw channels
            
            img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.channel_names1[c+1]))).read(), axis = 0)
        if config.multires:
            img2 = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.channel_names2[0]))).read()
        
        

    
    img1 = np.transpose(img1, axes=(1,2,0)) #Channel at the end
    if config.multires:
        
        img2 = np.transpose(img2, axes=(1,2,0))
    
    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.imshow(img1[:,:,0])
    # plt.subplot(122)
    # plt.imshow(img2[:,:,0])
    
    print('processing CHMs', img1[..., -1].min(), img1[..., -1].max(), img1[..., -1].mean())
    img1[..., -1] = img1[..., -1]/30
    img1[..., -1][img1[..., -1]>2]=0
    img1[..., -1][img1[..., -1]<0.03]=0
    
    print('processed CHMs',  img1[..., -1].min(),  img1[..., -1].max(),  img1[..., -1].mean())
    annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.annotation_fn))).read()
    annotation = np.squeeze(annotation)
    weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.weight_fn))).read()
    weight = np.squeeze(weight)
    density = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.density_fn))).read()
    density = np.squeeze(density)
    if config.multires:
        f = FrameInfo(img1, img2, annotation, weight, density)
    elif not config.multires:
        f = FrameInfo(img1, annotation, weight, density)
        
    
    frames.append(f)




# training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
# training_frames, validation_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
training_frames = validation_frames = testing_frames  = list(range(len(frames)))


annotation_channels = config.input_label_channel + config.input_weight_channel + config.input_density_channel
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, config.boundary_weights, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
#test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)

for _ in range(1):
    train_images, real_label = next(train_generator)
    if config.multires:
        train_im1, train_im2 = train_images
    elif not config.multires:
        train_im1 = train_images
    # print(train_im1.shape)
    print('color mean', train_im1.mean(axis = (0, 1, 2)))
    print('color std', train_im1.std(axis = (0, 1, 2)))
    print('color max', train_im1.max(axis = (0, 1, 2)))
    if config.multires:
        
        print(train_im2.mean(axis = (0, 1, 2)))
        print(train_im2.std(axis = (0, 1, 2)))
        train_im2 = resize(train_im2[:, :, :], (config.BATCH_SIZE, train_im1.shape[1], train_im1.shape[2]))
    print('count', real_label['output_dens'].sum(axis  =(1,2)))
    print('density map pixel value range', real_label['output_dens'].max()-real_label['output_dens'].min())
    # print(real_label['output_dens'])
    ann = real_label['output_seg'][...,0]
    wei = real_label['output_seg'][...,1]
    print('Boundary highlighted weights:', np.unique(wei))
    chms = train_im1[...,-1]
    print('chm range', chms.min(), chms.max())
    #overlay of annotation with boundary to check the accuracy
    #8 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    overlay = ann + wei
    # overlay = overlay[:,:,:,np.newaxis]
    overlay = overlay[...,np.newaxis]
    # print(overlay.shape)
    # print(real_label['output_seg'].shape)
    print('seg mask unique', np.unique(ann))
    if config.multires:
        display_images(np.concatenate((train_im1, train_im2,real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))
    else:
        display_images(np.concatenate((train_im1, real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))
        
class ImagesCallback(Callback):
    
    def __init__(self, config, val_data, savebase, batch_size = 8):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.savebase = savebase
        self.config = config
        
        
    def on_epoch_end(self, epoch,  logs=None):
        if epoch % 50 == 0:
            # print('Callback::validation_data:',self.validation_data)
            # print(self.validation_data.shape)
            # generate new samples for checking
            avgmse = 0
            print('-------> SAVING image callbacks')
            for t in range(30):
                x_test, y_test = next(self.validation_data)
                if self.config.multires:
                    test_im1, test_im2 = x_test
                else:
                    test_im1 = x_test
                y_test_seg = y_test['output_seg'][...,0]
                y_test_dens = y_test['output_dens']
                
                # print('GT:::Counts', y_test_dens.sum(axis  =(1,2)))
                gtdens = y_test_dens.sum(axis  =(1,2))
                
                # print('GT:::Counts total', y_test_dens.sum())
                # x_test = self.validation_data[0]
                # y_test = self.validation_data[1]
                # print(len(x_test), len(y_test))
                # print(x_test[0].shape)
                pps  = self.model.predict(x_test)
                if 'complex' in self.config.model_name:
                    pred1, pred2 = pps
                    
                else:
                    pred1 = pps['output_seg']
                    pred2 = pps['output_dens']
                    
                
                predens = np.squeeze(pred2.sum(axis  =(1,2)))
                
                # pred11 = pred1.copy()
                # pred11[pred11 > 0.1] = 1 # semantic seg
                # pred12 = pred1.copy()
                # pred12[pred12 > 0.8] = 1 # for count
                
                pred11 = pred1.copy()
                pred11[pred11 > 0.5] = 1 # for count
                
                
                # predens = [str(i) for i in predens]
                # print(predens)
                bn = [str(i) for i in list(range(self.batch_size))]
                segs = ['seg']*self.batch_size
                segps1 = ['pred thre05']*self.batch_size
                # print(titles)
                mse = np.mean((pred2 - y_test_dens)**2, axis = (1,2, 3))
                # mse = [str(i) for i in mse]
                tn = list(zip(predens, mse))
                tn = [str(i) for i in tn]
                # print(tn)
                titles = np.column_stack((bn, segs, segps1, gtdens, tn))
                
                img = np.concatenate((test_im1[..., 0][..., np.newaxis], y_test_seg[..., np.newaxis], pred11, y_test_dens, pred2), axis = -1)
                cols = img.shape[-1]
                rows = img.shape[0]
            
                fig = plt.figure(figsize=(14, 14 * rows // cols))
                for i in range(rows):
                    for j in range(cols):
                        plt.subplot(rows, cols, (i*cols) + j + 1)
                        plt.axis('off')
                        plt.imshow(img[i,...,j])
                        
                        plt.title(titles[i,j])
                
                
                mean_mse = np.mean(mse)
                avgmse += mean_mse
                savedir = self.savebase + '/epoch' + str(epoch) + '/' #'imageCallbacks/epoch' + str(epoch) + '/'
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                plt.savefig(savedir + str(mean_mse) + '_itr' + str(t) + '.jpg', quality = 30)
                # plt.show()
                plt.clf()
                plt.close(fig)
                
            plt.clf()
            plt.close('all')
            avgmse = avgmse / 50
            print('------- mse on the image callback validation set-----', avgmse)
            tf.summary.scalar('imageCallbacl density mse', data=avgmse, step=epoch)
            
            
model_path = config.model_path
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)

#reduceonplatea; It can be useful when using adam as optimizer
#Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
#cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                   patience=4, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=4, min_lr=1e-16)


log_dir = './logs/UNet_20211202-1605_Adam_e4_WTversky_Mse100_redgreenblueinfraredndvichm_256_85_frames/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
# file_writer.set_as_default()

tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=False, 
                          write_images=True, embeddings_freq=0, embeddings_layer_names=None, 
                          embeddings_metadata=None, embeddings_data=None, update_freq='epoch',
                          histogram_freq = 0)

# tensorboard = TensorBoard(log_dir=log_dir, 
#                           histogram_freq = 1, profile_batch = '10,50')

# weight_adj = LossWeightAdjust(alpha)
if config.log_img:
    Imagesave = './ImageCallbacks/UNet_20211202-1605_Adam_e4_WTversky_Mse100_redgreenblueinfraredndvichm_256_85_frames/'
    
    imagecallback = ImagesCallback(config, val_generator, Imagesave, config.BATCH_SIZE)
    callbacks_list = [imagecallback, checkpoint, tensorboard]

else:
    callbacks_list = [checkpoint, tensorboard]

tf.config.run_functions_eagerly(True)



loaded_model = load_model(model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity, 'miou': miou, 'weight_miou': weight_miou}, compile=False)
loaded_model.summary()

optimizer2=tf.keras.optimizers.Adam(lr= 0.001, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)

ev_seg = [dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou]
# ev_seg = [LOSS]
ev_count = [tf.keras.metrics.RootMeanSquaredError()]
    
compile_model_1input_2outputs(loaded_model, optimizer2, tversky, 'mse', 10000, ev_seg, ev_count)

loss_history = [loaded_model.fit(train_generator, 
                         steps_per_epoch=config.MAX_TRAIN_STEPS, 
                         initial_epoch = 389,
                         epochs=500, 
                         validation_data=val_generator,
                         validation_steps=config.VALID_IMG_COUNT,
                         callbacks=callbacks_list,
                         workers=1,
#                         use_multiprocessing=True # the generator is not very thread safe
                        )]