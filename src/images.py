import glob
import os

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

from constants import PIXEL_DEPTH, FOREGROUND_THRESHOLD


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


def load(directory):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    images = []
    for i, file_path in enumerate(glob.glob(os.path.join(directory, '*.png'))):
        img = mpimg.imread(file_path)
        images.append(img)
    print("Loaded {} images from {}".format(len(images), directory))
    return np.asarray(images)


def extract_patches(patch_size, *images):
    img_patches = [img_crop(image, patch_size, patch_size) for image in images]
    return np.asarray([patch for patches in img_patches for patch in patches])


def overlay(image, mask):
    w = image.shape[0]
    h = image.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = mask * PIXEL_DEPTH

    image = img_float_to_uint8(image)
    background = Image.fromarray(image, 'RGB').convert("RGBA")
    overlay_img = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay_img, 0.2)
    return new_img


def labels_for_patches(patches):
    foreground = patches.mean(axis=(1, 2)) > FOREGROUND_THRESHOLD
    return np.int64(foreground)


def image_from_patches(patches, im_width, im_height):
    """patches: [N, patch_height, patch_width, channels?]"""
    im_shape = list(patches.shape[1:])
    im_shape[0] = im_height
    im_shape[1] = im_width

    patch_size = patches.shape[1]

    im = np.ndarray(shape=im_shape, dtype=patches.dtype)
    i = 0
    for x in range(0, im_width, patch_size):
        for y in range(0, im_height, patch_size):
            im[y:y+patch_size,x:x+patch_size] = patches[i]
            i += 1

    return im


def image_from_predictions(predictions, patch_size, im_width, im_height):
    im = np.ndarray(shape=(im_height, im_width))
    i = 0
    for x in range(0, im_width, patch_size):
        for y in range(0, im_height, patch_size):
            im[y:y+patch_size,x:x+patch_size] = predictions[i]
            i += 1

    return im
