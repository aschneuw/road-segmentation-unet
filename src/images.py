import glob
import os

import numpy as np
import matplotlib.image as mpimg

PIXEL_DEPTH = 255  # TODO move


def img_crop(im, w, h):
    """Extract patches from a given image"""
    list_patches = []
    img_width = im.shape[0]
    img_height = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def load_images(directory):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    images = []
    for i, file_path in enumerate(glob.glob(os.path.join(directory, '*.png'))):
        if os.path.isfile(file_path):
            print('Loading ' + file_path)
            img = mpimg.imread(file_path)
            images.append(img)
        else:
            print('File {} does not exist'.format(file_path))
    return np.asarray(images)


def extract_patches(patch_size, *images):
    img_patches = [img_crop(image, patch_size, patch_size) for image in images]
    return np.asarray([patch for patches in img_patches for patch in patches])