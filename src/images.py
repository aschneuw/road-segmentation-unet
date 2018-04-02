import glob
import os

import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

from constants import PIXEL_DEPTH, FOREGROUND_THRESHOLD

"""
IMAGES
    This modules gather a handful of utility functions to manipulate images and masks.
    Most of the functions are defined for batches of images (4D tensors).
"""


def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images"""
    return (img * PIXEL_DEPTH).round().astype(np.uint8)


def load(directory):
    """Extract the images in `directory` into a tensor [num_images, height, width(, channels)]"""
    print('Loading images from {} ...'.format(directory))
    images = []
    for i, file_path in enumerate(sorted(glob.glob(os.path.join(directory, '*.png')))):
        img = mpimg.imread(file_path)
        images.append(img)
    print("Loaded {} images from {}".format(len(images), directory))
    return np.asarray(images)


def extract_patches(images, patch_size, stride=None, predict_patch_size=None):
    """extract square patches from a batch of images
    images:
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    patch_size:
        should divide the image width and height
    predict_patch_size:
        inside image than would need to be predicted (no mirror)
    returns:
        4D input: [num_patches, patch_size, patch_size, num_channel]
        3D input: [num_patches, patch_size, patch_size]
    """
    if not predict_patch_size:
        predict_patch_size = patch_size

    assert (patch_size - predict_patch_size) % 2 == 0 and predict_patch_size <= patch_size
    predict_patch_offset = int((patch_size - predict_patch_size) / 2)

    if not stride:
        stride = patch_size

    has_channels = (len(images.shape) == 4)

    num_images, image_height, image_width = images.shape[:3]
    assert image_height == image_width, "Assume square images"
    assert (image_height - patch_size) % stride == 0, "Stride sliding should cover the whole image"

    # expanded_size = image_width + 2 * predict_patch_offset
    # expanded_images = mirror_border(images, predict_patch_offset)

    patches_per_side = int((image_height - patch_size) / stride) + 1
    num_patches = images.shape[0] * patches_per_side * patches_per_side

    if has_channels:
        patches = np.zeros((num_patches, patch_size, patch_size, images.shape[-1]))
    else:
        patches = np.zeros((num_patches, patch_size, patch_size))

    patch_idx = 0
    for n in range(0, images.shape[0]):
        for x in range(0, image_width - patch_size + 1, stride):
            for y in range(0, image_height - patch_size + 1, stride):
                if has_channels:
                    patches[patch_idx] = images[n, y:y + patch_size, x:x + patch_size, :]
                else:
                    patches[patch_idx] = images[n, y:y + patch_size, x:x + patch_size]

                patch_idx += 1

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


def overlays(imgs, masks, fade=0.95):
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
    masks_red[:, :, :, 0] = 255
    masks_red[:, :, :, 3] = masks * fade

    results = np.zeros((num_images, im_width, im_height, 4), dtype=np.uint8)
    for i in range(num_images):
        x = Image.fromarray(imgs[i]).convert('RGBA')
        y = Image.fromarray(masks_red[i])
        results[i] = np.array(Image.alpha_composite(x, y))

    return results


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
    image_size = (num_patches_side - 1) * stride + patch_size

    images = np.zeros(shape=(num_images, image_size, image_size, num_channel), dtype=patches.dtype)
    count_hits = np.zeros(shape=(num_images, image_size, image_size, num_channel), dtype=np.uint64)

    for n in range(0, num_images):
        patch_idx = 0
        for x in range(0, image_size - patch_size + 1, stride):
            for y in range(0, image_size - patch_size + 1, stride):
                images[n, y:y + patch_size, x:x + patch_size] += patches[n, patch_idx]
                count_hits[n, y:y + patch_size, x:x + patch_size] += 1
                patch_idx += 1

    images = images / count_hits

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


def save_all(images, directory, format_="images_{:03d}.png", greyscale=False):
    """Save the `images` in the `directory`

    images: 3D or 4D tensor of images (with or without channels)
    directory: target directory
    format: naming with a placeholder for a integer index
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(images.shape) == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    if greyscale:
        cmap = "gray"
    else:
        cmap = mpl.rcParams.get("image.cmap")

    for n in range(images.shape[0]):
        mpimg.imsave(os.path.join(directory, format_.format(n + 1)), images[n], cmap=cmap)


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


def load_train_data(directory):
    """load images from `directory`, create patches and labels

    returns:
        images: [num_images, img_height, img_width, num_channel]
        labels: [num_images, img_height, img_width]
    """
    train_data_dir = os.path.abspath(os.path.join(directory, 'images/'))
    train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

    train_images = load(train_data_dir)
    train_groundtruth = load(train_labels_dir)

    return train_images, train_groundtruth


def quantize_mask(masks, threshold, patch_size):
    num_images, img_size, _, _ = masks.shape

    quantized_masks = masks.copy()
    for n in range(num_images):
        for y in range(0, img_size, patch_size):
            for x in range(0, img_size, patch_size):
                label = (masks[n, y:y + patch_size, x:x + patch_size, 0] >= 0.5).mean() > threshold
                quantized_masks[n, y:y + patch_size, x:x + patch_size, 0] = label

    return quantized_masks


def mirror_border(images, n):
    """mirrors border n border pixels on each side and corner:
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    returns:
        4D input: [num_patches, patch_size, patch_size, num_channel]
        3D input: [num_patches, patch_size, patch_size]
    """
    has_channels = (len(images.shape) == 4)
    if has_channels:
        return np.pad(images, ((0, 0), (n, n), (n, n), (0, 0)), "symmetric")
    else:
        return np.pad(images, ((0, 0), (n, n), (n, n)), "symmetric")


def overlap_pred_true(pred, true):
    num_images, im_height, im_width = pred.shape
    true_mask = img_float_to_uint8(true)
    pred_mask = img_float_to_uint8(pred)

    overlapped_mask = np.zeros((num_images, im_height, im_width, 3), dtype=np.uint8)
    overlapped_mask[:, :, :, 0] = pred_mask
    overlapped_mask[:, :, :, 1] = true_mask
    overlapped_mask[:, :, :, 2] = 0

    return overlapped_mask


def overlapp_error(pred, true):
    num_images, im_height, im_width = pred.shape
    true_mask = img_float_to_uint8(true).astype("bool", copy=False)
    pred_mask = img_float_to_uint8(pred).astype("bool", copy=False)
    error = np.logical_xor(true_mask, pred_mask)
    np.logical_not(error, out=error)
    error = img_float_to_uint8(error * 1)

    error_mask = np.zeros((num_images, im_height, im_width, 3), dtype=np.uint8)
    error_mask[:, :, :, 0] = error
    error_mask[:, :, :, 1] = error
    error_mask[:, :, :, 2] = error

    return error_mask


def rotate_imgs(imgs, angle):
    """safeguard to avoid useless rotation by 0"""
    if angle == 0:
        return imgs
    return rotate(imgs, angle=angle, axes=(1, 2), order=0)


def expand_and_rotate(imgs, angles, offset=0):
    """rotate some images by an angle, mirror image for missing part and expanding to output_size
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    angles: list of angle to rotate
    output_size: new size of image
    returns:
        4D input: [num_images * num_angles, output_size, output_size, num_channel]
        3D input: [num_images * num_angles, output_size, output_size]
    """

    has_channels = (len(imgs.shape) == 4)
    if not has_channels:
        imgs = np.expand_dims(imgs, -1)

    batch_size, height, width, num_channel = imgs.shape
    assert height == width

    output_size = height + 2 * offset
    padding = int(np.ceil(height * (np.sqrt(2) - 1) / 2)) + int(np.ceil(offset / np.sqrt(2)))

    print("Applying rotations: {} degrees... ".format(", ".join([str(a) for a in angles])))
    imgs = mirror_border(imgs, padding)
    rotated_imgs = np.zeros((batch_size * len(angles), output_size, output_size, num_channel))
    for i, angle in enumerate(angles):
        rotated_imgs[i * batch_size:(i + 1) * batch_size] = crop_imgs(rotate_imgs(imgs, angle), output_size)
    print("Done")

    if not has_channels:
        rotated_imgs = np.squeeze(rotated_imgs, -1)

    return rotated_imgs


def crop_imgs(imgs, crop_size):
    """
    imgs:
        3D or 4D images batch
    crop_size:
        width and height of the input
    """
    batch_size, height, width = imgs.shape[:3]
    assert height == width and height >= crop_size
    assert crop_size % 2 == 0
    half_crop = int(crop_size / 2)
    center = int(height / 2)

    has_channels = (len(imgs.shape) == 4)
    if has_channels:
        croped = imgs[:, center - half_crop:center + half_crop, center - half_crop:center + half_crop, :]
    else:
        croped = imgs[:, center - half_crop:center + half_crop, center - half_crop:center + half_crop]

    return croped


def image_augmentation_ensemble(imgs):
    """create ensemble of images to be predicted

    imgs: 4D images batch [num_images, height, width, channels]
    returns:  4D images batch [6 * num_images, height, width, channels]
    """
    num_imgs = imgs.shape[0]
    augmented_imgs = np.zeros((num_imgs * 6,) + imgs.shape[1:])

    # originals
    augmented_imgs[:num_imgs] = imgs

    # horizontal and vertical flip
    augmented_imgs[num_imgs:2 * num_imgs] = np.flip(imgs, axis=2)
    augmented_imgs[2 * num_imgs:3 * num_imgs] = np.flip(imgs, axis=1)

    # rotated images
    for i, k in enumerate([1, 2, 3]):
        augmented_imgs[(3 + i) * num_imgs:(4 + i) * num_imgs] = np.rot90(imgs, k=k, axes=(1, 2))

    return augmented_imgs


def invert_image_augmentation_ensemble(masks):
    """assemble masks of prediction images created by `image_augmentation_ensemble`

    masks: 3D masks batch [6 * num_images, height, width]
    returns: 3D masks batch [num_images, height, width]
    """
    assert masks.shape[0] % 6 == 0
    num_imgs = int(masks.shape[0] / 6)

    result = masks[:num_imgs]

    result += np.flip(masks[num_imgs:2 * num_imgs], axis=2)
    result += np.flip(masks[2 * num_imgs:3 * num_imgs], axis=1)

    # rotated images
    for i, k in enumerate([-1, -2, -3]):
        result += np.rot90(masks[(3 + i) * num_imgs:(4 + i) * num_imgs], k=k, axes=(1, 2))

    return result / 6
