
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    # structural changes, input and target change together
    seq_struc = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        # affine transform, only geometric
        sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16))),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    # color changes, only input images change
    seq_color = iaa.Sequential([
        
        # Gaussian blur and gamma contrast
        sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),
        # this only works with 255 rgb color space
        # sometimes(iaa.WithBrightnessChannels(iaa.Add((-20, 20))), 0.3),
        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
        sometimes(iaa.Multiply((0.85, 1.15), per_channel=True), 0.3),
        # sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        # # iaa.Add(value=(-0.5,0.5),per_channel=True),
        
    ],
        random_order=True)
    return seq_struc, seq_color

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """

    def __init__(self, input_image_channel1, patch_size, frame_list, frames, annotation_channel, boundary_weights = 10, augmenter=None):
        """Datagenerator constructor

        Args:
            input_image_channel1 (list(int)): Describes which channels is the image are input channels.
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class.
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
        """
        self.input_image_channel1 = input_image_channel1
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.boundary_weights = boundary_weights
        self.augmenter = augmenter

    
    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE, normalize, gbnorm):
        """Generate patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        patches1 = []
        patches2 = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch1, patch2 = frame.random_patch(self.patch_size, normalize, gbnorm)
            patches1.append(patch1)
            patches2.append(patch2)
        data1 = np.array(patches1) #256, 256, 8
        img2 = np.array(patches2) #128, 128, 1

        img1 = data1[..., self.input_image_channel1] # 256, 256, 5
        ann_joint = data1[..., self.annotation_channel]
        return ([img1, img2], ann_joint)
#     print("Wrote {} random patches to {} with patch size {}".format(count,write_dir,patch_size))

    def random_generator(self, BATCH_SIZE, normalize = 1, gb_norm = 0):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        seq_struc, seq_color = imageAugmentationWithIAA()
    
        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize, gb_norm)
            if self.augmenter == 'iaa':
                seq_struc_det = seq_struc.to_deterministic()
                seq_color_det = seq_color.to_deterministic()
                X1 = seq_struc_det.augment_images(X[0])
                X2 = seq_struc_det.augment_images(X[1])
                X1 = seq_color_det.augment_images(X[0])
                X2 = seq_color_det.augment_images(X[1])
                # y would have two channels, i.e. annotations and weights. We need to augment y for operations such as crop and transform
                y = seq_struc_det.augment_images(y)
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  y[...,[0]]
                ann[ann<0.5] = 0
                ann[ann>=0.5] = 1
                # boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                # weights[weights>=0.5] = 10
                weights[weights>=0.5] = self.boundary_weights #try lower weights
                weights[weights<0.5] = 1
                density = y[...,[2]]
                ann_joint = np.concatenate((ann,weights), axis=-1)
                
                yield ([X1, X2], {'output_seg': ann_joint, "output_dens": density})
                
            else:
                # y would have two channels, i.e. annotations and weights.
                ann =  y[...,[0]]
                #boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                # weights[weights>=0.5] = 10
                weights[weights>=0.5] = self.boundary_weights
                weights[weights<0.5] = 1

                density = y[...,[2]]
                ann_joint = np.concatenate((ann,weights), axis=-1)
                yield (X, {'output_seg': ann_joint, "output_dens": density})
        
