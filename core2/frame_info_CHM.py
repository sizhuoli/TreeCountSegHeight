#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen

import numpy as np
import cv2

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)
   
  
# Each area (ndvi, pan, annotation, weight) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation and weights (for weighted loss).
    """

    def __init__(self, img, CHM, dtype=np.float32):
        """FrameInfo constructor.

        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels, height and width must be same as img.
            weight: ndarray
                3D array containing weights for certain losses.
            dtype: np.float32, optional
                datatype of the array.
        """
        self.img = img
        self.CHM = CHM
        self.dtype = dtype

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def getPatch(self, i, j, patch_size, img_size, normalize=1.0, maxmin_norm = 0):
        """Function to get patch from the given location of the given size.

        Args:
            i: int
                Starting location on first dimension (x axis).
            y: int
                Starting location on second dimension (y axis).
            patch_size: tuple(int, int)
                Size of the patch.
            img_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        patch = np.zeros((patch_size[0], patch_size[1], patch_size[2]-1), dtype=self.dtype)
        patchchm = np.zeros((int(patch_size[0]/2), int(patch_size[1]/2), 1), dtype=self.dtype)
        im = self.img[i:i + img_size[0], j:j + img_size[1]]
        
        
        # an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        # an = np.expand_dims(an, axis=-1)
        # we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        # we = np.expand_dims(we, axis=-1)
        
        i2 = int(i/2)
        j2 = int(j/2)
        chmm = self.CHM[i2:i2 + int(img_size[0]/2), j2:j2 + int(img_size[1]/2)]
        chmm = np.expand_dims(chmm, axis=-1)
        r = np.random.random(1)
        # if normalize >= r[0]:
        #     # print('yes, the patch is normalized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     # print('previous max', im.mean())
        #     im = image_normalize(im, axis=(0, 1))
        #     # also normalize chm
        #     chmm = image_normalize(chmm, axis=(0, 1))
            
        # normalize globally
        im = im / 255
        im = (im - np.array([[0.31, 0.37, 0.34, 0.63, 0]]))/ np.array([[0.83, 0.74, 0.52, 0.81, 1]])
        
        
        
        if maxmin_norm:
            chmm = chmm/97.19
        patchchm[:int(img_size[0]/2), :int(img_size[1]/2), ] = chmm
        # comb_img = np.concatenate((im, an, we), axis=-1)
        patch[:img_size[0], :img_size[1], ] = im
        return (patch, patchchm)

    # Returns all patches in a image, sequentially generated
    def sequential_patches(self, patch_size, step_size, normalize, maxmin_norm):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Step size.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        if (img_shape[0] <= patch_size[0]):
            x = [0]
        if (img_shape[1] <= patch_size[1]):
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        chm_patches = []
        for i, j in xy:
            img_patch, chm = self.getPatch(i, j, patch_size, ic, normalize)
            img_patches.append(img_patch)
            chm_patches.append(chm)
        # print(len(img_patches))
        return (img_patches, chm_patches)

    # Returns a single patch, startring at a random image
    def random_patch(self, patch_size, normalize, maxmin_norm):
        """A random from this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch, chm = self.getPatch(x, y, patch_size, ic, normalize, maxmin_norm)
        return (img_patch, chm)
    
