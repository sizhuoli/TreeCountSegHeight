#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:40:40 2021

@author: sizhuo
"""

import logging
import tensorflow as tf
logging.info(f'tensorflow version: {tf.__version__}')
logging.info(tf.config.list_physical_devices('GPU'))
import numpy as np
import rasterio

import os
import time
import rasterio.warp             # Reproject raster samples
from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.split_frames import split_dataset
from core2.visualize import display_images

from skimage.transform import resize

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
import tensorflow.keras.backend as K
from tensorflow import Variable, summary

class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_generator, self.val_generator, self.no_frames = load_generators(self.config)

    def visualize_patches(self):
        """Visualize training data patches."""
        patch_visualizer(self.config, self.train_generator)

    def configure_training(self):
        """Set up training configurations and callbacks."""
        self.optimizer = adam  # Use Adam optimizer
        self.loss = tversky  # Use Tversky loss function
        timestamp = time.strftime("%Y%m%d-%H%M")

        # Create model directory if it does not exist
        os.makedirs(self.config.model_path, exist_ok=True)

        # Define the model save path
        model_filename = (
            f"trees_{timestamp}_{self.config.OPTIMIZER_NAME}_{self.config.chs}_"
            f"{self.config.input_shape[0]}_{self.no_frames}_{self.config.LOSS_NAME}_"
            f"{self.config.LOSS2}_{self.config.boundary_weights}_{self.config.model_name}_"
            f"{self.config.sufix}_densityR_"
            f"{self.config.task_ratio[0]}-{self.config.task_ratio[1]}-{self.config.task_ratio[2]}.keras"
        )
        model_path = os.path.join(self.config.model_path, model_filename)

        logging.info(f"Model will be saved at: {model_path}")

        # Callbacks
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min', save_weights_only = False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                      patience=4, verbose=1, mode='min',
                                      min_delta=0.0001, cooldown=4, min_lr=1e-16)

        # TensorBoard logging
        log_dir = os.path.join(
            './logs',
            f"UNet_{timestamp}_{self.config.OPTIMIZER_NAME}_{self.config.LOSS_NAME}_"
            f"{self.config.LOSS2}_{self.config.chs}_{self.config.input_shape[0]}_"
            f"{self.no_frames}"
        )
        os.makedirs(log_dir, exist_ok=True)

        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq = 0,
                                  write_graph=True, 
                                  write_images=True, 
                                  update_freq='epoch',
                                  embeddings_freq=0,
                                  embeddings_metadata=None
                                  )

        self.callbacks_list = [checkpoint, tensorboard, reduce_lr]

        tf.config.run_functions_eagerly(True)

    def load_model(self):
        """Load the UNet model based on configuration."""
        if self.config.multires:
            logging.info("Using Multi-resolution UNet.")
            from core2.UNet_multires_attention_segcount import UNet
        else:
            logging.info("Using Single-resolution UNet.")
            if self.config.ifBN:
                from core2.UNet_attention_segcount import UNet
            else:
                raise ValueError("Input image patches are not normalized. Batch normalization code is required.")
                # from core2.UNet_attention_segcount_noBN import UNet

        self.model = UNet([self.config.BATCH_SIZE, *self.config.input_shape],self.config.input_label_channel, inputBN = self.config.inputBN)
        return self.model

    def train(self):
        train_2tasks_steps(self.optimizer, self.loss, self.config, self.model, self.train_generator, self.val_generator, self.callbacks_list)


def train_2tasks_steps(OPTIMIZER, LOSS, config, model, train_generator, val_generator, callbacks_list):
    model.compile(optimizer=OPTIMIZER, loss={'output_seg':LOSS, 'output_dens':'mse'},
              loss_weights={'output_seg': 1., 'output_dens': 100},
              metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                       'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})


    model.fit(train_generator,
              steps_per_epoch=config.MAX_TRAIN_STEPS,
              epochs=5, # for testing. Should be 100
              validation_data=val_generator,
              validation_steps=config.VALID_IMG_COUNT,
              callbacks=callbacks_list
              )

    optimizer_config = OPTIMIZER.get_config()
    optimizer2 = tf.keras.optimizers.Adam.from_config(optimizer_config)
    # optimizer2=OPTIMIZER
    # optimizer2.set_weights(model.get_weights())

    model.compile(optimizer=optimizer2, loss={'output_seg':LOSS, 'output_dens':'mse'},
                  loss_weights={'output_seg': 1., 'output_dens': 10000},
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                           'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})

    model.fit(train_generator,
                             steps_per_epoch=config.MAX_TRAIN_STEPS,
                             initial_epoch = 101,
                             epochs=6, # for testing. Should be 500
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
    #                         use_multiprocessing=True # the generator is not very thread safe
                            )

    # optimizer3=OPTIMIZER
    # optimizer3.set_weights(model.optimizer.get_weights())
    optimizer3 = tf.keras.optimizers.Adam.from_config(optimizer_config)

    model.compile(optimizer=optimizer3, loss={'output_seg':LOSS, 'output_dens':'mse'},
                  loss_weights={'output_seg': 1., 'output_dens': 100000},
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                           'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})

    model.fit(train_generator,
                             steps_per_epoch=config.MAX_TRAIN_STEPS,
                             initial_epoch = 501,
                             epochs=7, # for testing. Should be 1500
                             validation_data=val_generator,
                             validation_steps=config.VALID_IMG_COUNT,
                             callbacks=callbacks_list,
                            )


def load_generators(config):
    if config.multires:
        print('*********************Multires*********************')
        from core2.frame_info_multires_segcount import FrameInfo
        from core2.dataset_generator_multires_segcount import DataGenerator
    else:
        print('*********************Single resolution*********************')
        from core2.frame_info_segcount import FrameInfo
        from core2.dataset_generator_segcount import DataGenerator

    # Read all images/frames into memory
    frames = []

    all_files = os.listdir(config.base_dir)
    # image channel 1
    all_files_c1 = [fn for fn in all_files if fn.startswith(config.channel_names[0]) and fn.endswith(config.image_type)]
    print(all_files_c1)

    for i, fn in enumerate(all_files_c1):
        # loop through rectangles
        img1 = rasterio.open(os.path.join(config.base_dir, fn)).read()
        if config.single_raster or not config.aux_data:
            # print('If single raster or multi raster without aux')
            for c in range(len(config.channel_names)-1):
                #loop through raw channels
                img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)

        else: # multi raster
            print('Multi raster with aux data')
            for c in range(len(config.channel_names)-1):
                    #loop through raw channels

                img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)
            if config.multires:
                img2 = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names2[0]))).read()

        img1 = np.transpose(img1, axes=(1,2,0)) #Channel at the end
        if config.multires:

            img2 = np.transpose(img2, axes=(1,2,0))


        # convert to grayscale
        if config.grayscale: # using grayscale
            print('Using grayscale images!!!!')
            img1 = rgb2gray(img1)
            img1 = img1[..., np.newaxis] # to match no. dimension
        annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.annotation_fn))).read()
        annotation = np.squeeze(annotation)
        # print('ann', annotation.shape)
        weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.weight_fn))).read()
        weight = np.squeeze(weight)
        # print('wei', weight.shape)
        density = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.density_fn))).read()
        density = np.squeeze(density)
        # print('den', density.shape)
        # print('im1', img1.shape)
        # print('im2', img2.shape)
        if config.multires:
            f = FrameInfo(img1, img2, annotation, weight, density)
        elif not config.multires:
            f = FrameInfo(img1, annotation, weight, density)

        frames.append(f)

    # using all images for both training and validation, may also split
    training_frames = validation_frames  = list(range(len(frames)))

    annotation_channels = config.input_label_channel + config.input_weight_channel + config.input_density_channel
    train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, config.boundary_weights, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)
    val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)

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
        overlay = ann + wei
        overlay = overlay[...,np.newaxis]
        print('seg mask unique', np.unique(ann))
        if config.multires:
            display_images(np.concatenate((train_im1, train_im2,real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))
        else:
            display_images(np.concatenate((train_im1, real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))

    return


class LossWeightAdjust(Callback):
    def __init__(self, alpha=None):
        if alpha is None:
            alpha = Variable(0.000001, trainable=False, dtype="float32")
        self.alpha = alpha

    def on_train_begin(self, logs = None):
        self.alphas = []

    def on_epoch_end(self, epoch, logs = None):
        # cursegloss = logs['output_seg_loss']
        curdensloss = logs['val_output_dens_loss']
        # Recalculate alpha
        lam = 10**(-np.floor(np.log10(curdensloss))-2)
        self.alpha.assign(lam)

        summary.scalar('task lossWeight', data=self.alpha.numpy(), step=epoch)
        logger.info("------- Loss weights recalibrated to alpha = %s -------", self.alpha.numpy())
        print("------- Loss weights recalibrated to alpha = %s -------" % (self.alpha.numpy()))
        self.alphas.append(self.alpha.numpy())


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def densityLoss(y_true, y_pred, beta = 0.0001):
    '''' density loss == spatial loss + beta * global loss '''
    glloss = mse(K.sum(y_true, axis=(1, 2, 3)),
                 K.sum(y_pred, axis=(1, 2, 3)))
    return mse(y_true, y_pred) + beta * glloss


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def mse_ssim(y_true, y_pred, delta = 0.01):
    ssiml = ssim_loss(y_true, y_pred)
    msel = mse(y_true, y_pred)
    return msel + delta * ssiml
