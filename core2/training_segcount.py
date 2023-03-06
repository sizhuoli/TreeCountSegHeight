#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:40:40 2021

@author: sizhuo
"""


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
# from tensorflow.keras.utils.layer_utils import count_params



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


class trainer:
    def __init__(self, config):
        self.config = config
        
        
        self.train_generator, self.val_generator, self.no_frames = load_generators(self.config)
        
    def vis(self):
        
        patch_visualizer(self.config, self.train_generator)
        
     
    def train_config(self):
        
        self.OPTIMIZER = adam #
        self.LOSS = tversky 
        # self.LOSS = msle_seg
        timestr = time.strftime("%Y%m%d-%H%M")
        
        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)
            
        model_path = os.path.join(self.config.model_path,'trees_{}_{}_{}_{}_{}_frames_{}_{}_{}weight_{}{}_densityR_{}-{}-{}.h5'.format(
                        timestr, self.config.OPTIMIZER_NAME, self.config.chs, self.config.input_shape[0], self.no_frames, self.config.LOSS_NAME, 
                        self.config.LOSS2, self.config.boundary_weights, self.config.model_name, self.config.sufix, self.config.task_ratio[0],
                        self.config.task_ratio[1],self.config.task_ratio[2]))
            
        print('model path:', model_path)
        
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)

        #reduceonplatea; It can be useful when using adam as optimizer
        #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
        #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                           patience=4, verbose=1, mode='min',
                                           min_delta=0.0001, cooldown=4, min_lr=1e-16)
        
        
        log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}_{}_frames'.format(
                    timestr, self.config.OPTIMIZER_NAME, self.config.LOSS_NAME, self.config.LOSS2, self.config.chs, 
                    self.config.input_shape[0], self.no_frames))
        
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
        if self.config.log_img:
            Imagesave = os.path.join(self.config.callbackImSave,'UNet_{}_{}_{}_{}_{}_{}_{}_frames'.format(
                        timestr, self.config.OPTIMIZER_NAME, self.config.LOSS_NAME, self.config.LOSS2, self.config.chs, 
                        self.config.input_shape[0], self.no_frames)) 
            
            imagecallback = ImagesCallback(self.config, self.val_generator, Imagesave, self.config.BATCH_SIZE)
            self.callbacks_list = [imagecallback, checkpoint, tensorboard]
        
        else:
            self.callbacks_list = [checkpoint, tensorboard]

        tf.config.run_functions_eagerly(True)
        return
    
    def LOAD_model(self):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # change seg activation function maybe used linear instead of sigmoid!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if 'complex' in self.config.model_name:
            if self.config.multires:
                print('*********************Multires*********************')
                from core.UNet_multires_attention_segcount import UNet
            elif not self.config.multires:
                print('*********************Single res*********************')
                if not self.config.ifBN:
                    from core.UNet_attention_segcount_noBN import UNet
                elif self.config.ifBN:
                    from core.UNet_attention_segcount import UNet
                    
            self.model = UNet([self.config.BATCH_SIZE, *self.config.input_shape],self.config.input_label_channel, inputBN = self.config.inputBN)
        
        elif 'efficientnet' in self.config.model_name or 'unet' in self.config.model_name:
            if self.config.multires:
                # efficientnet with unet
                ii1,ii2, ppseg, ppcount = get_model_2inputs_2outputs(
                        (256, 256, 5),(128, 128, 1),
                        dropout=False,
                        # backbone="efficientnetb0_2inputs",
                        backbone = self.config.backbone,
                        activation="elu",
                        batch_norm=True,
                        use_sep_conv = True,
                )
            
            
                self.model, self.optimizer = load_model_2inputs_2outputs(ppseg, ppcount, ii1, ii2, lr = 0.0001)
            
            elif not self.config.multires:
                # efficientnet with unet
                print('*******LOADing single input efficientnet')
                self.ii1,self.ppseg, self.ppcount = get_model_1input_2outputs(
                        (256, 256, 6),
                        dropout=False,
                        # backbone="efficientnetb0_2inputs",
                        backbone = self.config.backbone,
                        activation="elu",
                        batch_norm=True,
                        use_sep_conv = True,
                )
            
            
                
            
        return self.model
            
    def train(self):
        if 'complex' in self.config.model_name:
            train_2tasks_steps(self.OPTIMIZER, self.LOSS, self.config, self.model, self.train_generator, self.val_generator, self.callbacks_list)

        elif 'efficientnet' in self.config.model_name or 'unet' in self.config.model_name:
            
            train_eff(self.LOSS, self.config, self.train_generator, self.val_generator, self.callbacks_list)

        return
    
    def train_retrain(self):
        
        train_2tasks_steps_mod(self.config, self.train_generator, self.val_generator)

        return
    
    def train_retrain_eff(self):
        
        train_eff_mod(self.config, self.train_generator, self.val_generator)
        
def train_2tasks_steps(OPTIMIZER, LOSS, config, model, train_generator, val_generator, callbacks_list):
    model.compile(optimizer=OPTIMIZER, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
              loss_weights={'output_seg': 1., 'output_dens': 100}, 
              metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                        'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})


    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             epochs=100, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    optimizer2=OPTIMIZER
    # print('ini optimizer para:', optimizer2.get_weights())
    optimizer2.set_weights(model.optimizer.get_weights())
    # print('new optimizer para:', optimizer2.get_weights())
    
    model.compile(optimizer=optimizer2, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
                  loss_weights={'output_seg': 1., 'output_dens': 10000}, 
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                            'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
    
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 101,
                             epochs=500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    optimizer3=OPTIMIZER
    # print('ini optimizer para:', optimizer2.get_weights())
    optimizer3.set_weights(model.optimizer.get_weights())
    
    model.compile(optimizer=optimizer3, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
                  loss_weights={'output_seg': 1., 'output_dens': 100000}, 
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                            'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
    
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 501,
                             epochs=1500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    return

def train_2tasks_steps_mod( config, train_generator, val_generator):
    
    from core.UNet_attention_segcount import UNet
        
    model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
    opt = tf.keras.optimizers.Adam(lr= 0.00005, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)
    model_path = config.model_path +'trees_20211203-1735_Adam_e4_redgreenblueinfraredndvichm_256_85_frames_WTversky_Mse100_5weight_complex5_detCHM_retrain3.h5'
    model.load_weights(model_path)
    print('*************************model weights loaded')
    
    
    LOSS = tversky 
    
    print('model path:', model_path)
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                         save_best_only=True, mode='min', save_weights_only = False)

    #reduceonplatea; It can be useful when using adam as optimizer
    #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
    #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                       patience=4, verbose=1, mode='min',
                                       min_delta=0.0001, cooldown=4, min_lr=1e-16)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}'.format(
                timestr, config.OPTIMIZER_NAME, config.LOSS_NAME, config.LOSS2, config.chs, 
                config.input_shape[0]))
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
        Imagesave = os.path.join(config.callbackImSave,'UNet_{}_{}_{}_{}_{}_{}'.format(
                    timestr, config.OPTIMIZER_NAME, config.LOSS_NAME, config.LOSS2, config.chs, 
                    config.input_shape[0])) 
        
        imagecallback = ImagesCallback(config, val_generator, Imagesave, config.BATCH_SIZE)
        callbacks_list = [imagecallback, checkpoint, tensorboard]
    
    else:
        callbacks_list = [checkpoint, tensorboard]

    tf.config.run_functions_eagerly(True)
    
    
    
    model.compile(optimizer=opt, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
                  loss_weights={'output_seg': 1., 'output_dens': 100000}, 
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy],
                            'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
    
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 501,
                             epochs=1500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    return            
        
def train_eff(LOSS, config, train_generator, val_generator, callbacks_list):
    # efficient net
    ev_seg = [dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou]
    # ev_seg = [LOSS]
    ev_count = [tf.keras.metrics.RootMeanSquaredError()]
    
    # train with pretraiend first
    ii1,ppseg, ppcount = get_model_1input_2outputs(
                (256, 256, 6),
                dropout=False,
                # backbone="efficientnetb0_2inputs",
                backbone = config.backbone,
                activation="elu",
                batch_norm=True,
                use_sep_conv = True,
        )
    
    model, optimizer = load_model_1input_2outputs(ppseg, ppcount, ii1, lr = 0.0001)
    
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    
    
    print('*******************************************************************')
    print('lr', optimizer.lr.numpy())
    print('Total params: {:,}'.format(totalParams))
    print('Trainable params: {:,}'.format(trainableParams))
    print('Non-trainable params: {:,}'.format(nonTrainableParams))
    print('*******************************************************************')

    model = compile_model_1input_2outputs(model, optimizer, LOSS, 'mse', 100, ev_seg, ev_count)
            
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             epochs=10, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    
    # unfreeze pretrained layers and train all
    model.trainable = True
    optimizer1=optimizer
    optimizer1.set_weights(model.optimizer.get_weights())
    print('*********************************************************************')
    print('Train all layers')
    print('*********************************************************************')
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    print('*******************************************************************')
    print('Total params: {:,}'.format(totalParams))
    print('Trainable params: {:,}'.format(trainableParams))
    print('Non-trainable params: {:,}'.format(nonTrainableParams))
    print('*******************************************************************')
    
    model = compile_model_1input_2outputs(model, optimizer1, LOSS, 'mse', 100, ev_seg, ev_count)
            
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 11,
                             epochs=100, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    
    optimizer2=optimizer
    optimizer2.set_weights(model.optimizer.get_weights())
    
    model = compile_model_1input_2outputs(model, optimizer2, LOSS, 'mse', 10000, ev_seg, ev_count)
        
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 101,
                             epochs=500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    
    optimizer3=optimizer
    # print('ini optimizer para:', optimizer2.get_weights())
    optimizer3.set_weights(model.optimizer.get_weights())

    model = compile_model_1input_2outputs(model, optimizer3, LOSS, 'mse', 100000, ev_seg, ev_count)
            
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 501,
                             epochs=1500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]

    return 


def train_eff_mod(config, train_generator, val_generator):
    ev_seg = [dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou]
    # ev_seg = [LOSS]
    ev_count = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = tversky 
    # train with pretraiend first
    ii1,ppseg, ppcount = get_model_1input_2outputs(
                (256, 256, 6),
                dropout=False,
                # backbone="efficientnetb0_2inputs",
                backbone = config.backbone,
                activation="elu",
                batch_norm=True,
                use_sep_conv = True,
        )
    
    model, optimizer = load_model_1input_2outputs(ppseg, ppcount, ii1, lr = 0.0001)
    model_path = config.model_path +'trees_20211204-0123_Adam_e4_redgreenblueinfraredndvichm_256_85_frames_WTversky_Mse100_5weight_efficientnet_B2_detCHM_retrain.h5'    
    model.load_weights(model_path)
    print('*************************model weights loaded')
    
    model.trainable = True
    print('*********************************************************************')
    print('Train all layers')
    print('*********************************************************************')
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    print('*******************************************************************')
    print('Total params: {:,}'.format(totalParams))
    print('Trainable params: {:,}'.format(trainableParams))
    print('Non-trainable params: {:,}'.format(nonTrainableParams))
    print('*******************************************************************')
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                         save_best_only=True, mode='min', save_weights_only = False)

    timestr = time.strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}_{}'.format(
                timestr, config.OPTIMIZER_NAME, config.LOSS_NAME, config.LOSS2, config.chs, 
                config.input_shape[0]))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=False, 
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None, 
                              embeddings_metadata=None, embeddings_data=None, update_freq='epoch',
                              histogram_freq = 0)
    
    if config.log_img:
        Imagesave = os.path.join(config.callbackImSave,'UNet_{}_{}_{}_{}_{}_{}'.format(
                    timestr, config.OPTIMIZER_NAME, config.LOSS_NAME, config.LOSS2, config.chs, 
                    config.input_shape[0])) 
        
        imagecallback = ImagesCallback(config, val_generator, Imagesave, config.BATCH_SIZE)
        callbacks_list = [imagecallback, checkpoint, tensorboard]
    
    else:
        callbacks_list = [checkpoint, tensorboard]

    tf.config.run_functions_eagerly(True)
    model = compile_model_1input_2outputs(model, optimizer, LOSS, densityLoss, 1, ev_seg, ev_count)
            
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             initial_epoch = 501,
                             epochs=900, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
       

def train_dis(OPTIMIZER, LOSS, config, model, train_generator, val_generator, callbacks_list):
    
    ################## try for distance masks
    model.compile(optimizer=OPTIMIZER, loss={'output_seg':'mse', 'output_dens':'mse'}, 
                  loss_weights={'output_seg': 1., 'output_dens': 1}, 
                  metrics={'output_seg':[specificity, sensitivity, accuracy],
                            'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
    
    
    loss_history = [model.fit(train_generator, 
                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
                             epochs=500, 
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                             workers=1,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )]
    return

        



def load_generators(config):
    if config.multires:
        print('*********************Multires*********************')
        from core.frame_info_multires_segcount import FrameInfo
        from core.dataset_generator_multires_segcount import DataGenerator
    elif not config.multires:
        print('*********************Single resolution*********************')
        from core.frame_info_segcount import FrameInfo
        from core.dataset_generator_segcount import DataGenerator
        
    # Read all images/frames into memory
    frames = []
    
    all_files = os.listdir(config.base_dir)
    # image channel 1
    all_files_c1 = [fn for fn in all_files if fn.startswith(config.extracted_filenames[0]) and fn.endswith(config.image_type)]
    print(all_files_c1)
    
    ######################################################################################################    
    ######################################################################################################    
    for i, fn in enumerate(all_files_c1):
        # loop through rectangles
        img1 = rasterio.open(os.path.join(config.base_dir, fn)).read()
        if config.single_raster or not config.aux_data:
            # print('If single raster or multi raster without aux')
            for c in range(config.image_channels1-1):
                #loop through raw channels
                img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names1[0],config.channel_names1[c+1]))).read(), axis = 0)
            
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
        
        # print('processing CHMs', img1[..., -1].min(), img1[..., -1].max(), img1[..., -1].mean())
        # img1[..., -1] = img1[..., -1]/30
        # img1[..., -1][img1[..., -1]>2]=0
        # img1[..., -1][img1[..., -1]<0.03]=0
        
        # print('processed CHMs',  img1[..., -1].min(),  img1[..., -1].max(),  img1[..., -1].mean())
        
        # convert to grayscale
        if config.grayscale: # using grayscale
            print('Using grayscale images!!!!')
            img1 = rgb2gray(img1)
            # plt.figure(figsize=(5,5))
            # plt.imshow(img1, cmap  ='gray')
            img1 = img1[..., np.newaxis] # to match no. dimension
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
    
    
    return train_generator, val_generator, len(all_files_c1)



def patch_visualizer(config, train_generator):
    
    for _ in range(1):
        train_images, real_label = next(train_generator)
        if config.multires:
            train_im1, train_im2 = train_images
            chms = train_im2[...,-1]
            print('chm range', chms.min(), chms.max())
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
            
    return




class LossWeightAdjust(Callback):
    def __init__(self, alpha = K.variable(0.000001)):
        self.alpha = alpha
    def on_train_begin(self, logs = None):
        self.alphas = []
        
    def on_epoch_end(self, epoch, logs = None):
        # cursegloss = logs['output_seg_loss']
        curdensloss = logs['val_output_dens_loss']
        
        lam = 10**(-np.floor(np.log10(curdensloss))-2)
        K.set_value(self.alpha, lam)
        
        tf.summary.scalar('task lossWeight', data=K.get_value(self.alpha), step=epoch)
        logger.info("------- Loss weights recalibrated to alpha = %s -------" % (K.get_value(self.alpha)))
        print("------- ------- Loss weights recalibrated to alpha = %s ------- ------- " % (K.get_value(self.alpha)))
        self.alphas.append(K.get_value(self.alpha))
        
   


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

# def logmse(y_true, y_pred):
#     return K.log(K.mean(K.square(y_pred - y_true)))


def densityLoss(y_true, y_pred, beta = 0.0001):
    '''' density loss == spatial loss + beta * global loss '''
    glloss = mse(K.sum(y_true, axis = (1, 2, 3)), K.sum(y_pred, axis = (1, 2, 3)))
    return mse(y_true, y_pred) + beta * glloss

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


class ImagesCallback(Callback):
    
    def __init__(self, config, val_data, savebase, batch_size = 8):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.savebase = savebase
        self.config = config
        
        
    def on_epoch_end(self, epoch,  logs=None):
        if epoch % 50 == 3:
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
            
            
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        

def mse_ssim(y_true, y_pred, delta = 0.01):
    ssiml = ssim_loss(y_true, y_pred)
    msel = mse(y_true, y_pred)
    return msel + delta * ssiml










# model_path = os.path.join(config.model_path,'trees_20210616-0122_Adam_e4_redgreenblueinfraredndvichm_256_74_frames_weightmapTversky_Mse100_5weight_complex5_NOBatchNorm_normTrain.h5')
# loaded_model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity, 'miou': miou, 'weight_miou': weight_miou}, compile=False)

# log_dir = os.path.join('./logs','UNet_20210616-0122_Adam_e4_weightmapTversky_Mse100_redgreenblueinfraredndvichm_256_74_frames')

# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
#                              save_best_only=True, mode='min', save_weights_only = False)

# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# # weight_adj = LossWeightAdjust(alpha)

# # callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta
# Imagesave = os.path.join(config.callbackImSave,
#                          'UNet_20210616-0122_Adam_e4_weightmapTversky_Mse100_redgreenblueinfraredndvichm_256_74_frames') 

# imagecallback = ImagesCallback(val_generator, Imagesave)
# callbacks_list = [imagecallback, checkpoint, tensorboard]





# tf.config.run_functions_eagerly(True)


# loaded_model.compile(optimizer=OPTIMIZER, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
#               loss_weights={'output_seg': 1., 'output_dens': 100000}, 
#               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
#                         'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})


# loss_history = [loaded_model.fit(train_generator, 
#                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                          initial_epoch = 990,
#                          epochs=1400, 
#                          validation_data=val_generator,
#                          validation_steps=config.VALID_IMG_COUNT,
#                          callbacks=callbacks_list,
#                          workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )]





# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta
# tf.config.run_functions_eagerly(True)

# loaded_model.compile(optimizer=OPTIMIZER, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
#               loss_weights={'output_seg': 1., 'output_dens': 100}, 
#               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
#                        'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})



# loss_history = [loaded_model.fit(train_generator, 
#                          steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                          initial_epoch = 204,
#                           epochs=300, 
#                          validation_data=val_generator,
#                          validation_steps=config.VALID_IMG_COUNT,
#                          callbacks=callbacks_list,
#                          workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )]




# model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
# model.load_weights(model_path)
# model.compile(optimizer=OPTIMIZER, loss={'output_seg':LOSS, 'output_dens':'mse'}, 
#               loss_weights={'output_seg': 1., 'output_dens': 100000}, 
#               metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou],
#                        'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})
# model.save('./saved_models/segcountdensity/testsave.h5')


# #change lr to 0.0001

# model_path = os.path.join(config.model_path,'trees_RGB20210218-0213_Adam_weightmap_tversky_redgreenblue_256_84_frames_chm_ndvi_5weight_subatte_resi_sub3conv_atrous_lr00003_elu.h5')
# log_dir = os.path.join('./logs','UNet_20210218-0213_Adam_weightmap_tversky_redgreenblue_256_84_frames')
# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
#                              save_best_only=True, mode='min', save_weights_only = False)

# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta
# tf.config.run_functions_eagerly(True)


# loaded_model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity, 'miou': miou, 'weight_miou': weight_miou}, compile=False)
# loaded_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity, miou, weight_miou])
# loaded_model.fit(train_generator, 
#                           steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                           initial_epoch = 3000,
#                           epochs=4000, 
#                           validation_data=val_generator,
#                           validation_steps=config.VALID_IMG_COUNT,
#                           callbacks=callbacks_list,
#                           workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )


# loaded_model.fit(train_generator, 
#                           steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                           initial_epoch = 4000,
#                           epochs=5000, 
#                           validation_data=val_generator,
#                           validation_steps=config.VALID_IMG_COUNT,
#                           callbacks=callbacks_list,
#                           workers=1,
# #                         use_multiprocessing=True # the generator is not very thread safe
#                         )

# # Load model after training
# # If you load a model with different python version, than you may run into a problem: https://github.com/keras-team/keras/issues/9595#issue-303471777
# model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity}, compile=False)

# # In case you want to use multiple GPU you can uncomment the following lines.
# # from tensorflow.python.keras.utils import multi_gpu_model
# # model = multi_gpu_model(model, gpus=2, cpu_merge=False)

# model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])





# # Print one batch on the training/test data!
# for i in range(1):
#     val_images, real_label = next(val_generator)
#     val_im1, val_im2 = val_images
#     #5 images per row: pan, ndvi, label, weight, prediction
#     ann = real_label['output_seg'][...,0][..., np.newaxis]
#     wei = real_label['output_seg'][...,1]
#     cc = real_label['output_dens'].sum()
    
#     predictions = model.predict(val_images, steps=1)
#     predc = predictions[1].sum()
#     print('gt', cc)
#     print('ored', predc)
#     prediction = predictions[0]
#     prediction[prediction>=2]=10
#     prediction[prediction<1]=0
#     display_images(np.concatenate((val_im1, ann, prediction), axis = -1))






