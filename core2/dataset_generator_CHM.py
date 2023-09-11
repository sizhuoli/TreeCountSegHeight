#    Author: Ankit Kariryaa, University of Bremen

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        #
        # # Gaussian blur and gamma contrast
        # sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
        # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),
        # sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        # # iaa.Add(value=(-0.5,0.5),per_channel=True),
        # sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        # sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """

    def __init__(self, input_image_channel, patch_size, frame_list, frames, augmenter=None):
        """Datagenerator constructor

        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class.
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        
        self.augmenter = augmenter

    # Return all training and label images and weights, generated sequentially with the given step size
    def all_sequential_patches(self, step_size, normalize = 1, maxmin_norm = 0):

        patches = []
        chms = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            patch, patchchm = frame.sequential_patches(self.patch_size, step_size, normalize, maxmin_norm)
            patches.extend(patch)
            chms.extend(patchchm)
        data = np.array(patches)
        label = np.array(chms)
        img = data[..., self.input_image_channel]
        
        return (img, label)

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE, normalize, maxmin_norm):

        patches = []
        chms = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch, chm = frame.random_patch(self.patch_size, normalize, maxmin_norm)
            patches.append(patch)
            chms.append(chm)
        data = np.array(patches)
        chms = np.array(chms)
        img = data[..., self.input_image_channel]

        return (img, chms)

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE, normalize = 1, maxmin_norm = 0):

        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize, maxmin_norm)
            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                y = seq_det.augment_images(y)
                yield X, y
            else:
                yield X, y

