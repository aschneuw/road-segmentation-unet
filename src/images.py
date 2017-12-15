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


def extract_patches(images, patch_size, stride=None, angles=None, predict_patch_size=None):
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

    if not angles:
        angles = [0]

    assert (patch_size - predict_patch_size) % 2 == 0 and predict_patch_size <= patch_size
    predict_patch_offset = int((patch_size - predict_patch_size) / 2)

    if not stride:
        stride = patch_size

    has_channels = (len(images.shape) == 4)
    if not has_channels:
        images = np.expand_dims(images, -1)

    num_images, image_height, image_width, num_channel = images.shape
    assert image_height == image_width, "Assume square images"
    assert (image_height - predict_patch_size) % stride == 0, "Stride sliding should cover the whole image"

    expanded_size = image_width + 2 * predict_patch_offset
    expanded_images = rotate_and_mirror(images, angles, expanded_size)

    patches_per_side = int((image_height - predict_patch_size) / stride) + 1
    num_patches = expanded_images.shape[0] * patches_per_side * patches_per_side

    patches = np.zeros((num_patches, patch_size, patch_size, num_channel))

    patch_idx = 0
    for n in range(0, expanded_images.shape[0]):
        for x in range(0, expanded_size - patch_size + 1, stride):
            for y in range(0, expanded_size - patch_size + 1, stride):
                patches[patch_idx] = expanded_images[n, y:y + patch_size, x:x + patch_size, :]
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


def rotate_imgs(imgs, angle):
    return rotate(imgs, angle=angle, axes=(1, 2), order=0)


def rotate_and_mirror(imgs, angles, output_size=None, auto_expand=True):
    """rottate some images by an angle, mirror image for missing part and expanding to output_size
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

    if not output_size:
        output_size = height

    assert output_size >= height
    offset = output_size - height
    padding = int(height / 2) + int(np.ceil(offset / np.sqrt(2)))

    if (0 not in angles) and (auto_expand is True):
        angles = [0] + angles

    imgs = mirror_border(imgs, padding)
    print("Applying rotations: {} degrees... ".format(", ".join(str(a) for a in angles)))
    imgs = np.concatenate([crop_imgs(rotate_imgs(imgs, angle), output_size) for angle in angles], axis=0)
    print("Done")

    if not has_channels:
        imgs = np.squeeze(imgs, -1)

    return imgs


def crop_imgs(imgs, crop_size):
    """
    imgs:
        3D or 4D images batch
    crop_size:
        width and height of the input
    """
    has_channels = (len(imgs.shape) == 4)
    if not has_channels:
        imgs = np.expand_dims(imgs, -1)

    batch_size, height, width, num_channel = imgs.shape
    assert height == width and height >= crop_size

    assert crop_size % 2 == 0
    half_crop = int(crop_size / 2)
    center = int(height / 2)
    croped = imgs[:, center - half_crop:center + half_crop, center - half_crop:center + half_crop, :]

    if not has_channels:
        croped = np.squeeze(croped, -1)

    return croped

def flip_hor_imgs(imgs):
    hor_flip = np.flip(imgs, axis=2)
    return hor_flip

def flip_vert_imgs(imgs):
    vert_flip = np.flip(imgs, axis=1)
    return vert_flip

def augment_pred_rot_and_flip(imgs, invert = False):
    if not invert:
        hor_flip = flip_hor_imgs(imgs)
        vert_flip = flip_vert_imgs(imgs)
        rot_imgs = rotate_and_mirror(imgs, angles=[90, 180, 270], auto_expand=False)
        aug_imgs = np.concatenate((imgs, hor_flip, vert_flip, rot_imgs))
        assert aug_imgs.shape[0] == imgs.shape[0]*6
        return aug_imgs

    else:
        aug_imgs = imgs
        n_img = aug_imgs.shape[0] / 6
        assert n_img % 1 == 0
        n_img = int(n_img)

        imgs = aug_imgs[0:n_img]

        rev_aug = np.zeros((6,) + tuple(imgs.shape))

        rev_aug[0] = imgs
        rev_aug[1] = flip_hor_imgs(aug_imgs[n_img:2 * n_img])

        rev_aug[2] = flip_vert_imgs(aug_imgs[n_img * 2:3 * n_img])

        rev_aug[3], rev_aug[4], rev_aug[5] = tuple([rotate_and_mirror(aug_imgs[(index - 1) * n_img:index * n_img], angles=[angle], auto_expand=False) for angle, index in
                                                    zip([270, 180, 90], [4, 5, 6])])

        return np.average(rev_aug, axis=0)