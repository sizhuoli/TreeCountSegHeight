#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:40:40 2021

@author: sizhuo
"""

import os
import time
import logging
import numpy as np
import rasterio
import rasterio.warp
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
import tensorflow.keras.backend as K
from tensorflow import Variable, summary

from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.split_frames import split_dataset
from core2.visualize import display_images

logging.info(f'tensorflow version: {tf.__version__}')
logging.info(tf.config.list_physical_devices('GPU'))
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_generator, self.val_generator, self.no_frames = self._load_generators()

    def visualize_patches(self):
        """Visualize training data patches."""
        patch_visualizer(self.config, self.train_generator)

    def configure_training(self):
        """Set up training configurations and callbacks."""
        self.optimizer = adam  # Adam optimizer
        self.loss = tversky  # Tversky loss function
        timestamp = time.strftime("%Y%m%d-%H%M")

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

    def _load_generators(self):
        if self.config.multires:
            print('*********************Multires*********************')
            from core2.frame_info_multires_segcount import FrameInfo
            from core2.dataset_generator_multires_segcount import DataGenerator
        else:
            print('*********************Single resolution*********************')
            from core2.frame_info_segcount import FrameInfo
            from core2.dataset_generator_segcount import DataGenerator

        all_files = os.listdir(self.config.base_dir)
        all_files_c1 = [file for file in all_files if file.startswith(self.config.channel_names[0]) and file.endswith(self.config.image_type)]
        logging.info(all_files_c1)

        frames = []

        for file in all_files_c1:
            img1, img2 = self._load_image_data(file)

            if self.config.grayscale:
                logging.info("Using grayscale images!")
                img1 = rgb2gray(img1)[..., np.newaxis]

            annotation = np.squeeze(self._load_raster_data(file, self.config.annotation_fn))
            weight = np.squeeze(self._load_raster_data(file, self.config.weight_fn))
            density = np.squeeze(self._load_raster_data(file, self.config.density_fn))

            # Create FrameInfo object
            if self.config.multires:
                f = FrameInfo(img1, img2, annotation, weight, density)
            else:
                f = FrameInfo(img1, annotation, weight, density)

            frames.append(f)
        
        # Set up training and validation frames (using all images here)
        training_frames = validation_frames = list(range(len(frames))) 

        annotation_channels = self.config.input_label_channel + self.config.input_weight_channel + self.config.input_density_channel
        
        train_generator = DataGenerator(
            self.config.input_image_channel, self.config.patch_size, training_frames, frames, annotation_channels, 
            self.config.boundary_weights, augmenter='iaa'
        ).random_generator(self.config.BATCH_SIZE, normalize=self.config.normalize)

        val_generator = DataGenerator(
            self.config.input_image_channel, self.config.patch_size, validation_frames, frames, annotation_channels, 
            self.config.boundary_weights, augmenter=None
        ).random_generator(self.config.BATCH_SIZE, normalize=self.config.normalize)

        return train_generator, val_generator, len(all_files_c1)
    
    def _load_image_data(self, file):
        img1 = rasterio.open(os.path.join(self.config.base_dir, file)).read()
        
        if self.config.single_raster or not self.config.aux_data:
            for c in range(1, self.config.image_channel_count):
                img1 = np.append(
                    img1, self._load_raster_data(file, self.config.channel_names[c]), axis=0
                )
        else:
            for c in range(1, self.config.image_channel_count):
                img1 = np.append(
                    img1, self._load_raster_data(file, self.config.channel_names[c]), axis=0
                )
            if self.config.multires:
                img2 = self._load_raster_data(file, self.config.channel_names2[0])

        img1 = np.transpose(img1, axes=(1, 2, 0))  # Channel at the end
        if self.config.multires:
            img2 = np.transpose(img2, axes=(1, 2, 0))

        return img1, img2 if self.config.multires else img1
    
    def _load_raster_data(self, file, channel_name):
        """Load raster data and squeeze unnecessary dimensions"""
        return rasterio.open(os.path.join(self.config.base_dir, file.replace(self.config.channel_names[0], channel_name))).read()


def train_2tasks_steps(OPTIMIZER, LOSS, config, model, train_generator, val_generator, callbacks_list):
    """Train model in 3 stages with different learning rates and loss weights."""
    model.compile(optimizer=OPTIMIZER, 
                  loss={'output_seg':LOSS, 'output_dens':'mse'},
                  loss_weights={'output_seg': 1., 'output_dens': 100},
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                           'output_dens':[tf.keras.metrics.RootMeanSquaredError()]}
                )


    model.fit(train_generator,
              steps_per_epoch=config.MAX_TRAIN_STEPS,
              epochs=100, # for testing. Should be 100
              validation_data=val_generator,
              validation_steps=config.VALID_IMG_COUNT,
              callbacks=callbacks_list
              )
    
    # Second stage: Update optimizer and decrease learning rate
    optimizer_config = OPTIMIZER.get_config()
    optimizer2 = tf.keras.optimizers.Adam.from_config(optimizer_config)

    model.compile(optimizer=optimizer2, loss={'output_seg':LOSS, 'output_dens':'mse'},
                  loss_weights={'output_seg': 1., 'output_dens': 10000},
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                           'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})

    model.fit(train_generator,
              steps_per_epoch=config.MAX_TRAIN_STEPS,
              initial_epoch = 101,
              epochs=500, # for testing. Should be 500
              validation_data=val_generator,
              validation_steps=config.VALID_IMG_COUNT,
              callbacks=callbacks_list
              )

    # Third stage: Further update optimizer and adjust loss weights
    optimizer3 = tf.keras.optimizers.Adam.from_config(optimizer_config)

    model.compile(optimizer=optimizer3, loss={'output_seg':LOSS, 'output_dens':'mse'},
                  loss_weights={'output_seg': 1., 'output_dens': 100000},
                  metrics={'output_seg':[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou],
                           'output_dens':[tf.keras.metrics.RootMeanSquaredError()]})

    model.fit(train_generator,
              steps_per_epoch=config.MAX_TRAIN_STEPS,
              initial_epoch = 501,
              epochs=1500, # for testing. Should be 1500
              validation_data=val_generator,
              validation_steps=config.VALID_IMG_COUNT,
              callbacks=callbacks_list,
              )


def patch_visualizer(config, train_generator):
    for _ in range(1): # Only visualize one batch
        train_images, real_label = next(train_generator)

        if config.multires:
            train_im1, train_im2 = train_images
            chms = train_im2[...,-1]
            logging.info('CHM range', chms.min(), chms.max())
        else:
            train_im1 = train_images

        logging.info('color mean', train_im1.mean(axis = (0, 1, 2)))
        logging.info('color std', train_im1.std(axis = (0, 1, 2)))
        logging.info('color max', train_im1.max(axis = (0, 1, 2)))

        if config.multires:
            logging.info(train_im2.mean(axis = (0, 1, 2)))
            logging.info(train_im2.std(axis = (0, 1, 2)))
            train_im2 = resize(train_im2[:, :, :], (config.BATCH_SIZE, train_im1.shape[1], train_im1.shape[2]))

        logging.info(f'Count: {real_label["output_dens"].sum(axis=(1, 2))}')
        logging.info(f'Density map pixel value range: {real_label["output_dens"].max() - real_label["output_dens"].min()}')

        ann = real_label['output_seg'][...,0]
        wei = real_label['output_seg'][...,1]
        logging.info(f'Boundary highlighted weights: {np.unique(wei)}')

        overlay = np.expand_dims(ann + wei, axis=-1)
        logging.info(f'Segmentation mask unique values: {np.unique(ann)}')

        if config.multires:
            display_images(np.concatenate((train_im1, train_im2,real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))
        else:
            display_images(np.concatenate((train_im1, real_label['output_seg'], overlay, real_label['output_dens']), axis = -1))


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
    global_loss = mse(K.sum(y_true, axis=(1, 2, 3)), K.sum(y_pred, axis=(1, 2, 3)))
    return mse(y_true, y_pred) + beta * global_loss


def rgb2gray(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def mse_ssim(y_true, y_pred, delta = 0.01):
    """MSE + SSIM loss with a weighting factor."""
    mse_loss_value = mse(y_true, y_pred)
    ssim_loss_value = ssim_loss(y_true, y_pred)
    return mse_loss_value + delta * ssim_loss_value
