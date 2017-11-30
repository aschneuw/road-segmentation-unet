import glob
import os
import re

import matplotlib.image as mpimg
import numpy as np
from datetime import datetime

from constants import PIXEL_DEPTH, FOREGROUND_THRESHOLD, IMG_PATCH_SIZE


def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images"""
    return (img * PIXEL_DEPTH).round().astype(np.uint8)


def load(directory):
    """Extract the images in `directory` into a tensor [num_images, height, width(, channels)]"""
    print('Loading images from {} ...'.format(directory))
    images = []
    for i, file_path in enumerate(glob.glob(os.path.join(directory, '*.png'))):
        img = mpimg.imread(file_path)
        images.append(img)
    print("Loaded {} images from {}".format(len(images), directory))
    return np.asarray(images)


def extract_patches(images, patch_size):
    """extract square patches from a batch of images

    images:
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    patch_size:
        should divide the image width and height

    returns:
        4D input: [num_patches, patch_size, patch_size, num_channel]
        3D input: [num_patches, patch_size, patch_size]
    """
    has_channels = (len(images.shape) == 4)
    if not has_channels:
        images = np.expand_dims(images, -1)

    num_images, image_height, image_width, num_channel = images.shape
    assert image_height % patch_size == 0 and image_width % patch_size == 0
    num_patches = num_images * int(image_height / patch_size) * int(image_width / patch_size)

    patches = np.zeros((num_patches, patch_size, patch_size, num_channel))

    patch_idx = 0
    for n in range(0, num_images):
        for x in range(0, image_width, patch_size):
            for y in range(0, image_height, patch_size):
                patches[patch_idx] = images[n, y:y + patch_size, x:x + patch_size, :]
                patch_idx += 1

    if not has_channels:
        patches = np.squeeze(patches, -1)

    return patches


def labels_for_patches(patches):
    """Compute the label for a some patches
    Change FOREGROUND_THRESHOLD to modify the ratio for positive/negative

    patches:
        shape: [num_batches, patch_size, patch_size]
    returns:
        the label 1 = road, 0 = other
        [num_batches, ]
    """
    foreground = patches.mean(axis=(1, 2)) > FOREGROUND_THRESHOLD
    return np.int64(foreground)


def overlays(images, masks, fade=0.2):
    """Add the masks on top of the images with red transparency

    images:
        array of images
        shape: [num_images, im_height, im_width, num_channel]

    masks:
        array of masks
        shape: [num_images, im_height, im_width, 1]

    returns:
        [num_images, im_height, im_width, num_channel]
    """
    num_images, im_height, im_width, num_channel = images.shape
    color_masks = np.zeros((num_images, im_width, im_height, num_channel), dtype=np.uint8)
    color_masks[:, :, :, 0] = masks[:, :, :, 0] * PIXEL_DEPTH
    if num_channel == 4:
        color_masks[:, :, :, 3] = masks[:, :, :, 0] * PIXEL_DEPTH
    images = img_float_to_uint8(images)

    return (1 - fade) * images + fade * color_masks


def images_from_patches(patches):
    """Transform a list of patches into images

    patches:
        array of patches by image. We assume that the image is squared, num_channel should be 1, 3 or 4.
        shape: [num_images, num_patches, patch_size, patch_size, num_channel]

    returns:
        num_images square images from 2D concat patches
    """
    num_images, num_patches, patch_size, _, num_channel = patches.shape
    num_patches_side = int(np.sqrt(num_patches))
    assert np.sqrt(num_patches) == num_patches_side, "Square image assumption broken"
    image_size = num_patches_side * patch_size

    images = np.ndarray(shape=(num_images, image_size, image_size, num_channel), dtype=patches.dtype)

    for n in range(0, num_images):
        patch_idx = 0
        for x in range(0, image_size, patch_size):
            for y in range(0, image_size, patch_size):
                images[n, y:y + patch_size, x:x + patch_size] = patches[n, patch_idx]
                patch_idx += 1

    return images


def predictions_to_patches(predictions, patch_size):
    """Expand each prediction to a square patch

    predictions:
        array of prediction
        shape: [num_predictions,]

    returns:
        [num_predictions, patch_size, patch_size, 1]
    """
    num_predictions = predictions.shape[0]
    predictions = np.resize(predictions, (num_predictions, 1, 1, 1))
    patches = np.broadcast_to(predictions, (num_predictions, patch_size, patch_size, 1))
    return patches

def save_all(images, directory, format_="images_{:03d}.png"):
    """Save the `images` in the `directory`
    images: 3D or 4D tensor of images (with or without channels)
    directory: target directory
    format: naming with a placeholder for a integer index
    """
    for n in range(images.shape[0]):
        mpimg.imsave(directory + format_.format(n + 1), images[n])


def mask_to_submission_strings(mask_number, mask):
    """Generates submission output for a single mask"""
    patch_size = IMG_PATCH_SIZE
    for j in range(0, mask.shape[1], patch_size):
        for i in range(0, mask.shape[0], patch_size):
            patch = mask[i:i + patch_size, j:j + patch_size]
            label = int(np.mean(patch) > FOREGROUND_THRESHOLD)
            yield "{:03d}_{}_{},{}".format(mask_number, j, i, label)

def save_submission_csv(masks, path):
    """Save the masks in the expected format for submission
    masks: [num_mask, mask_height, mask_width] with 1 for road, 0 for other
    path: target CSV file path, will add new submission with datetime in filename
    """
    submission_filename = 'submission-{}.csv'.format(datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"))
    submission_path = os.path.abspath(os.path.join(path, submission_filename))

    with open(submission_path, 'w') as f:
        f.write('id,prediction\n')
        for i, mask in enumerate(masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(i, mask))

    print("Submission file written to: {}".format(submission_path))
