import glob
import os

import matplotlib.image as mpimg
import numpy as np

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

    n_patches = (image_width - patch_size/stride) + 1
    

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
    return foreground.astype(np.int64)


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

    images = (1 - fade) * images + fade * color_masks
    images = img_float_to_uint8(images)

    return images


def images_from_patches(patches, stride=None):
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
        labels: [num_images, num_patches_side, num_patches_side]
    """
    train_data_dir = os.path.abspath(os.path.join(directory, 'images/'))
    train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

    train_images = load(train_data_dir)

    num_images, img_height, img_width, num_channel = train_images.shape
    num_patches_side = int(img_height / patch_size)

    train_groundtruth = load(train_labels_dir)
    train_groundtruth_patches = extract_patches(train_groundtruth, patch_size)
    train_labels = labels_for_patches(train_groundtruth_patches)
    train_labels = train_labels.reshape((num_images, num_patches_side, num_patches_side))

    return train_images, train_labels
