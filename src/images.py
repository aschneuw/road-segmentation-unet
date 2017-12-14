import glob
import os

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

from constants import PIXEL_DEPTH, FOREGROUND_THRESHOLD


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


def extract_patches(images, patch_size, stride=None, augmented=False):
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
    if augmented:
        print("Start image augmentation: {} initial images".format(images.shape[0]))
        rotations = []
        for k in range(0, 4):
            rotations.append(np.rot90(images, k, axes=(1, 2)))
        images = np.concatenate(rotations, axis=0)
        print("After 90 degree rotations: {} images".format(images.shape[0]))

        flips = [images, np.fliplr(images), np.flipud(images)]
        images = np.concatenate(flips, axis=0)
        print("After ud lr flips: {} images".format(images.shape[0]))

    if not stride:
        stride = patch_size

    has_channels = (len(images.shape) == 4)
    if not has_channels:
        images = np.expand_dims(images, -1)

    num_images, image_height, image_width, num_channel = images.shape
    assert image_height == image_width, "Assume square images"
    assert (image_height - patch_size) % stride == 0, "Stride sliding should cover the whole image"

    patches_per_side = int((image_height - patch_size) / stride) + 1
    num_patches = num_images * patches_per_side * patches_per_side

    patches = np.zeros((num_patches, patch_size, patch_size, num_channel))

    patch_idx = 0
    for n in range(0, num_images):
        for x in range(0, image_width - patch_size + 1, stride):
            for y in range(0, image_height - patch_size + 1, stride):
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


def load_train_data(directory, rot_angles=None):
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

    if rot_angles:
        print("Applying rotations: {} degrees.".format(",".join(str(a) for a in rot_angles)))
        rot_train_images = mirror_rotate(train_images, rot_angles)
        rot_train_groundtruth = mirror_rotate(train_groundtruth, rot_angles)
        train_images = np.concatenate((train_images, rot_train_images))
        train_groundtruth = np.concatenate((train_groundtruth, rot_train_groundtruth))
        print("Total images, including rotations: {}".format(train_groundtruth.shape[0]))

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
    if not has_channels:
        images = np.expand_dims(images, -1)

    num_images, image_height, image_width, num_channel = images.shape

    extended = np.pad(images, ((0, 0), (n, n), (n, n), (0, 0)), "symmetric")

    if not has_channels:
        extended = np.squeeze(extended, -1)

    return extended


def overlap_pred_true(pred, true):
    num_images, im_height, im_width = pred.shape
    true_mask = img_float_to_uint8(true)
    pred_mask = img_float_to_uint8(pred)

    overlapped_mask = np.zeros((num_images, im_height, im_width, 3), dtype=np.uint8)
    overlapped_mask[:, :, :, 0] = pred_mask
    overlapped_mask[:, :, :, 1] = true_mask
    overlapped_mask[:, :, :, 2] = 0

    return overlapped_mask


def mirror_rotate(img, angles):
    """TODO"""
    rotated = [_mirror_rotate_crop(img, angle) for angle in angles]
    rot_concat = np.concatenate(rotated)
    return rot_concat


def _mirror_rotate_crop(img, angle):
    assert 360 > angle > 0 and angle < 360

    has_channels = (len(img.shape) == 4)
    if not has_channels:
        img = np.expand_dims(img, -1)

    # consistency check
    num_imgs, img_height, img_width, num_channel = img.shape
    assert img_height == img_width

    # mirror and extend
    ext_size = int(np.ceil(img_height * (np.sqrt(2) - 1) / 2))
    extended_img = mirror_border(img, ext_size)

    # rotate
    rot = rotate(extended_img, angle=angle, axes=(1, 2), order=0)
    rot_imgs, rot_height, rot_width, num_channel = rot.shape
    assert rot_height == rot_width

    # crop
    margin_h = int(np.floor((rot_height - img_height) / 2))
    margin_w = int(np.floor((rot_width - img_height) / 2))

    rot_s = rot[:, margin_h:margin_h + img_height, margin_w:margin_w + img_width, :]
    rot_s_imgs, rot_s_length, rot_s_width, rot_s_num_channel = rot_s.shape

    # consistency check
    assert rot_s_length == img_height
    assert rot_s_width == img_width

    if not has_channels:
        rot_s = np.squeeze(rot_s, -1)

    return rot_s
