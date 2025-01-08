import logging
import os
import time
import warnings

import numpy as np
import rasterio
import rasterio.warp
import scipy
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model

from core2.dataset_generator_segcount import DataGenerator
from core2.frame_info_segcount import FrameInfo
from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
tf.config.run_functions_eagerly(True)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.frames = []

    def load_local_data(self):
        logging.info("Loading local data...")
        all_files = os.listdir(self.config.base_dir)
        
        # Filter files for channel 1
        all_files_c1 = [fn for fn in all_files if
                        fn.startswith(self.config.extracted_filenames[0]) and fn.endswith(self.config.image_type)]
        
        for _ in range(self.config.oversample_times):
            for fn in all_files_c1:
                base_path = os.path.join(self.config.base_dir, fn)
                comb_img = self._load_image(base_path)

                for c in range(1, self.config.image_channels):
                    cur_channel_fn = fn.replace(self.config.extracted_filenames[0], self.config.extracted_filenames[c])
                    cur_channel_path = os.path.join(self.config.base_dir, cur_channel_fn)
                    cur_channel_img = self._load_image(cur_channel_path)
                    comb_img = np.append(comb_img, cur_channel_img, axis=0)

                # Transpose image to (H, W, C)
                comb_img = np.transpose(comb_img, axes=(1, 2, 0))

                # Load annotation
                annotation_path = base_path.replace(self.config.extracted_filenames[0], self.config.annotation_fn)
                annotation = rasterio.open(annotation_path).read()
                annotation = np.squeeze(annotation)
                if self.config.upsample:
                    annotation = scipy.ndimage.zoom(annotation, 2, order=1)

                # Load weight map
                weight_path = base_path.replace(self.config.extracted_filenames[0], self.config.weight_fn)
                weight = rasterio.open(weight_path).read()
                weight = np.squeeze(weight)
                if self.config.upsample:
                    weight = scipy.ndimage.zoom(weight, 2, order=1)

                # Load density map
                density_path = base_path.replace(self.config.extracted_filenames[0], self.config.density_fn)
                density = rasterio.open(density_path).read()
                density = np.squeeze(density)

                if self.config.upsample:
                    new_shape = (density.shape[0] * 2, density.shape[1] * 2)
                    density = resize(density, new_shape, preserve_range=True).astype(float)
                    scale_factor = (density.shape[0] / new_shape[0]) * (density.shape[1] / new_shape[1])
                    density *= scale_factor

                # Store frame
                self.frames.append(FrameInfo(comb_img, annotation, weight, density))

        logging.info(f"Local data loaded. Total number of frames: {len(self.frames)}")

    def _load_image(self, path):
        """Loads a single raster image and applies optional upsampling."""
        img = rasterio.open(path).read()
        if self.config.upsample:
            img = resize(
                img, (1, img.shape[1] * 2, img.shape[2] * 2), preserve_range=True
            )
        return img

    def load_pretraining_data(self):
        """Loads initial training data from the secondary directory."""
        logging.info("Loading initial training data...")
        all_files2 = os.listdir(self.config.base_dir2)
        
        # Filter files for channel 1
        all_files_c12 = [fn for fn in all_files2 if
                         fn.startswith(self.config.extracted_filenames[0]) and fn.endswith(self.config.image_type)]

        for fn in all_files_c12:
            base_path = os.path.join(self.config.base_dir2, fn)
            
            # Load primary raster channels
            img1 = rasterio.open(base_path).read()

            if self.config.single_raster or not self.config.aux_data:
                for c in range(1, self.config.image_channels):
                    img1 = np.append(img1, 
                                     rasterio.open(os.path.join(self.config.base_dir2,
                                                                fn.replace(self.config.extracted_filenames[0],
                                                                           self.config.extracted_filenames[c]))).read(),
                                     axis=0)
            else: 
                for c in range(self.config.image_channels - 1):
                    img1 = np.append(img1, 
                                     rasterio.open(os.path.join(self.config.base_dir,
                                                                fn.replace(self.config.channel_names1[0],
                                                                           self.config.channel_names1[c + 1]))).read(), 
                                     axis=0)

                img2 = rasterio.open(
                    os.path.join(self.config.base_dir, fn.replace(self.config.channel_names1[0], self.config.channel_names2[0]))).read()

            # Transpose img1 to (H, W, C)
            img1 = np.transpose(img1, axes=(1, 2, 0))
            # img2 = np.transpose(img2, axes=(1,2,0))
            
            annotation = rasterio.open(
                os.path.join(self.config.base_dir2, fn.replace(self.config.extracted_filenames[0], self.config.annotation_fn))).read()
            annotation = np.squeeze(annotation)
            weight = rasterio.open(
                os.path.join(self.config.base_dir2, fn.replace(self.config.extracted_filenames[0], self.config.weight_fn))).read()
            weight = np.squeeze(weight)
            density = rasterio.open(
                os.path.join(self.config.base_dir2, fn.replace(self.config.extracted_filenames[0], self.config.density_fn))).read()
            density = np.squeeze(density)
            f = FrameInfo(img1, annotation, weight, density)
            self.frames.append(f)

        print('initial data loaded, total no. frames: ', len(self.frames))

    def wrap_data(self):
        training_frames = validation_frames = list(range(len(self.frames)))

        annotation_channels = self.config.input_label_channel + self.config.input_weight_channel + self.config.input_density_channel
        self.train_generator = DataGenerator(self.config.input_image_channel, self.config.patch_size, training_frames, self.frames,
                                        annotation_channels, self.config.boundary_weights, augmenter='iaa').random_generator(
            self.config.BATCH_SIZE, normalize=self.config.normalize)
        self.val_generator = DataGenerator(self.config.input_image_channel, self.config.patch_size, validation_frames, self.frames,
                                      annotation_channels, self.config.boundary_weights, augmenter=None).random_generator(
            self.config.BATCH_SIZE, normalize=self.config.normalize)

    def model_ready_train(self):
        OPTIMIZER = adam
        LOSS = tversky

        if self.config.multires:
            from core2.UNet_multires_attention_segcount import UNet
        else:
            from core2.UNet_attention_segcount import UNet
        # ipdb.set_trace()
        self.model = UNet([self.config.BATCH_SIZE, *self.config.input_shape],
                          self.config.input_label_channel, inputBN=self.config.inputBN)
        self.model.load_weights(self.config.model_path)

        # save logs
        timestr = time.strftime("%Y%m%d-%H%M")

        new_model_path = os.path.join(self.config.new_model_path, 'finetune_{}.h5'.format(timestr))
        checkpoint = ModelCheckpoint(new_model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min', save_weights_only=False)

        log_dir = os.path.join(self.config.log_dir, 'finetune_{}'.format(timestr))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                  embeddings_data=None, update_freq='epoch')

        callbacks_list = [checkpoint, tensorboard]

        self.model.compile(optimizer=OPTIMIZER, loss={'output_seg': LOSS, 'output_dens': 'mse'},
                             loss_weights={'output_seg': 1., 'output_dens': 10000},
                             metrics={
                                 'output_seg': [dice_coef, specificity, sensitivity, miou, weight_miou, accuracy],
                                 'output_dens': [tf.keras.metrics.RootMeanSquaredError()]})

        loss_history = [self.model.fit(self.train_generator,
                                         steps_per_epoch=self.config.MAX_TRAIN_STEPS,
                                         initial_epoch=self.config.pretrain_NBepochs + 1,
                                         epochs=self.config.pretrain_NBepochs + self.config.NB_EPOCHS,
                                         validation_data=self.val_generator,
                                         validation_steps=self.config.VALID_IMG_COUNT,
                                         callbacks=callbacks_list,
                                         workers=1,
                                         )]

        return loss_history
