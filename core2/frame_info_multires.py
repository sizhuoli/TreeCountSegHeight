#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen

import numpy as np

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)
   
 
# Each area (ndvi, pan, annotation, weight) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation and weights (for weighted loss).
    """

    def __init__(self, img1, img2, annotations, weight, dtype=np.float32):
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
        self.img1 = img1 # 256, 256, 5
        self.img2 = img2 # 128, 128, 1
        self.annotations = annotations # 256, 256
        self.weight = weight # 256, 256
        self.dtype = dtype



    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
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
        patch1 = np.zeros((256, 256, 7), dtype=self.dtype)
        # for lower resolution patch (patch size 128, 128)
        # patch2 = np.zeros((patch_size/2).astype(np.int32), dtype=self.dtype)
        patch2 = np.zeros((128, 128, 1), dtype=self.dtype)
        img2_size = (int(img_size[0]/2), int(img_size[1]/2))
        i2 = int(i/2)
        j2 = int(j/2)
        
        im1 = self.img1[i:i + img_size[0], j:j + img_size[1]]
        im2 = self.img2[i2:i2 + img2_size[0], j2:j2 + img2_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im1 = image_normalize(im1, axis=(0, 1))
            im2 = image_normalize(im2, axis=(0, 1))
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)
        we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        we = np.expand_dims(we, axis=-1)
        comb_img = np.concatenate((im1, an, we), axis=-1)
        patch1[:img_size[0], :img_size[1], ] = comb_img # zero padding for empty area
        patch2[:img2_size[0], :img2_size[1], ] = im2
        return (patch1, patch2)

    # Returns a single patch, startring at a random image
    def random_patch(self, patch_size, normalize):
        """A random from this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img1.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1])) #image size (compared with patch size)
        img_patch1, img_patch2 = self.getPatch(x, y, patch_size, ic, normalize)
        return (img_patch1, img_patch2)
