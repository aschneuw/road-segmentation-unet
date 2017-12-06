import glob
import os

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.stats import mode

from constants import PIXEL_DEPTH, FOREGROUND_THRESHOLD


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


def extract_patches(images, patch_size, stride=None):
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

    # stride = 16
    # patch_size = 128
    if not stride:
        stride = patch_size

    has_channels = (len(images.shape) == 4)

    if not has_channels:
        images = np.expand_dims(images, -1)


    num_images, image_height, image_width, num_channel = images.shape
    assert image_height == image_width

    border_margin = int((patch_size - stride) / 2)
    assert border_margin % 1 == 0

    num_patches_image = int(((image_width - border_margin*2) / stride))**2
    assert num_patches_image % 1 == 0

    num_total_patches = images.shape[0] * num_patches_image

    patches = np.zeros((num_total_patches, patch_size, patch_size, num_channel))

    patch_idx = 0
    for n in range(0, num_images):
        for x in range(0, image_width - patch_size, stride):
            for y in range(0, image_height - patch_size, stride):
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
    return foreground.astype(np.int64)


def overlays(imgs, masks, fade=0.2):
    """Add the masks on top of the images with red transparency

    imgs:
        array of images
        shape: [num_images, im_height, im_width, num_channel]

    masks:
        array of masks
        shape: [num_images, im_height, im_width, 1]

    returns:
        [num_images, im_height, im_width, num_channel]
    """
    num_images, im_height, im_width, num_channel = imgs.shape
    assert num_channel == 3, 'Predict image should be colored'

    imgs = img_float_to_uint8(imgs)
    masks = img_float_to_uint8(masks.squeeze())
    masks_red = np.zeros((num_images, im_height, im_width, 4), dtype=np.uint8)
    masks_red[:, :, :, 0] = masks
    masks_red[:, :, :, 3] = masks * fade

    results = np.zeros((num_images, im_width, im_height, 4), dtype=np.uint8)
    for i in range(num_images):
        x = Image.fromarray(imgs[i]).convert('RGBA')
        y = Image.fromarray(masks_red[i])
        results[i] = np.array(Image.alpha_composite(x, y))

    return results


def images_from_patches(patches, stride=None, border_majority_only=True, normalize=True):
    """Transform a list of patches into images

    patches:
        array of patches by image. We assume that the image is squared, num_channel should be 1, 3 or 4.
        shape: [num_images, num_patches, patch_size, patch_size, num_channel]

    returns:
        num_images square images from 2D concat patches
    """

    num_images, num_patches, patch_size, _, num_channel = patches.shape

    if stride is None:
        stride = patch_size


    num_patches_side = int(np.sqrt(num_patches))

    border_margin = patch_size - stride
    border_margin_os = int(border_margin / 2)

    total_stride_length = num_patches_side * stride
    image_size = total_stride_length + border_margin

    assert np.sqrt(num_patches) == (image_size - border_margin) / stride, "Square image assumption broken"

    count = np.zeros(shape=(num_images, image_size, image_size, num_channel))
    sum_ = np.zeros(shape=(num_images, image_size, image_size, num_channel))

    for n in range(0, num_images):
        patch_idx = 0
        for x in range(0, image_size - border_margin, stride):
            x_start = x
            x_stop = x + patch_size
            for y in range(0, image_size - border_margin, stride):
                y_start = y
                y_stop = y + patch_size
                count[n, y_start:y_stop, x_start:x_stop, :] += 1
                sum_[n, y_start:y_stop, x_start:x_stop, :] += patches[n, patch_idx]
                patch_idx += 1
    if normalize:
        images = sum_ / count
    else:
        images = sum_

    if border_majority_only:
        for n in range(0, num_images):
            patch_idx = 0
            for x in range(border_margin_os, image_size - border_margin_os, stride):
                x_start = x
                x_stop = x + stride
                for y in range(border_margin_os, image_size - border_margin_os, stride):
                    y_start = y
                    y_stop = y + stride
                    images[n,y_start:y_stop, x_start:x_stop, :] = patches[n, patch_idx, border_margin_os:-border_margin_os, border_margin_os:-border_margin_os]
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

    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(images.shape) == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    for n in range(images.shape[0]):
        mpimg.imsave(os.path.join(directory, format_.format(n + 1)), images[n])


def save_submission_csv(masks, path, patch_size):
    """Save the masks in the expected format for submission

    masks: binary mask at pixel level with 1 for road, 0 for other
        shape: can be 3D [num_mask, mask_height, mask_width] or 4D [num_mask, mask_height, mask_width, 1]
    path: target CSV file path, will remove existing
    """
    if len(masks.shape) == 4:
        masks = masks.squeeze(-1)

    num_mask, mask_height, mask_width = masks.shape
    assert mask_height == mask_width, "images should be square"
    patches_per_side = int(mask_height / patch_size)

    patches = extract_patches(masks, patch_size)
    labels = labels_for_patches(patches)
    labels.resize((num_mask, patches_per_side, patches_per_side))

    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.abspath(os.path.join(path, "submission.csv"))

    with open(filename, 'w') as file:
        print("Saving predictions in {}".format(filename))
        file.write("id,prediction\n")
        for image_idx in range(num_mask):
            for j in range(patches_per_side):
                for i in range(patches_per_side):
                    label = labels[image_idx, j, i]
                    file.write("{:03d}_{}_{},{}\n".format(image_idx + 1, patch_size * j, patch_size * i, label))
        print("Done")


def load_train_data(directory, patch_size):
    """load images from `directory`, create patches and labels

    returns:
        images: [num_images, img_height, img_width, num_channel]
        labels: [num_images, img_height, img_width]
    """
    train_data_dir = os.path.abspath(os.path.join(directory, 'images/'))
    train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

    train_images = load(train_data_dir)

    num_images, img_height, img_width, num_channel = train_images.shape

    train_groundtruth = load(train_labels_dir)

    return train_images, train_groundtruth
