import matplotlib.pyplot as plt  # plotting tools
from matplotlib.patches import Polygon
import os
# def display_images(img, titles=None, cmap=None, norm=None, interpolation=None, savefig = None, savename = None):
#     """Display the given set of images, optionally with titles.
#     images: array of image tensors in Batch * Height * Width * Channel format.
#     titles: optional. A list of titles to display with each image.
#     cmap: Optional. Color map to use. For example, "Blues".
#     norm: Optional. A Normalize instance to map values to colors.
#     interpolation: Optional. Image interpolation to use for display.
#     """
#     cols = img.shape[-1]
#     rows = img.shape[0]
#     titles = titles if titles is not None else [""] * (rows*cols)

#     plt.figure(figsize=(14, 14 * rows // cols))
#     for i in range(rows):
#         for j in range(cols):
#             plt.subplot(rows, cols, (i*cols) + j + 1)
#             plt.axis('off')
#             plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
#             if titles:
#                 plt.title(titles[j])
#         if savefig and savename:
#             plt.savefig(savename)

def display_images(img, titles=None, cmap=None, norm=None, interpolation=None, savefig = None, savename = None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    cols = img.shape[-1]
    rows = img.shape[0]
    titles = titles if titles is not None else [""] * (rows*cols)

    plt.figure(figsize=(14, 14 * rows // cols))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, (i*cols) + j + 1)
            plt.axis('off')
            plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
            
            plt.title(titles[j])
        if savefig and savename:
            # if not os.path.exists(savename):
            #     os.makedirs(savename)
    
            plt.savefig(savename)