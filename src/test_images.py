from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

import images
from constants import PIXEL_DEPTH


class TestStridePatchGeneration(TestCase):
    def test_608_608_image_to_patches_stride(self):
        N_IMAGES = 10
        N_CHANNEL = 3
        IMAGE_WIDTH = IMAGE_HEIGHT = 608
        PATCH_SIZE = 128
        STRIDE = 16
        N_PATCHES = 10 * 31 * 31

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)
        patches = patches.reshape((N_IMAGES, int(N_PATCHES / N_IMAGES), PATCH_SIZE, PATCH_SIZE, N_CHANNEL))

        n_images, n_patches, p_height, p_width, channels = patches.shape

        assert n_images == 10
        assert n_patches == 31 * 31
        assert p_height == 128
        assert p_width == 128
        assert channels == 3

    def test_608_608_image_to_patches_no_stride(self):
        N_IMAGES = 10
        N_CHANNEL = 3
        IMAGE_WIDTH = IMAGE_HEIGHT = 608
        PATCH_SIZE = 32

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE)

        n_patches, p_height, p_width, channels = patches.shape

        assert n_patches == 3610
        assert p_height == 32
        assert p_width == 32
        assert channels == 3

    def test_608_608_patches_to_image_stride(self):
        N_IMAGES = 10
        N_CHANNEL = 3
        IMAGE_WIDTH = IMAGE_HEIGHT = 608
        PATCH_SIZE = 128
        STRIDE = 16
        N_PATCHES = 10 * 31 * 31

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)

        patches = patches.reshape((N_IMAGES, int(N_PATCHES / N_IMAGES), PATCH_SIZE, PATCH_SIZE, N_CHANNEL))

        reconstructed_images = images.images_from_patches(patches, stride=STRIDE)

        num_images, image_height, image_width, n_channel = reconstructed_images.shape

        self.assertTrue(num_images == N_IMAGES)
        self.assertTrue(image_height == IMAGE_HEIGHT)
        self.assertTrue(image_width == IMAGE_WIDTH)
        self.assertTrue(n_channel == N_CHANNEL)

    def test_608_608_patches_to_image_stride_cummulative(self):
        N_IMAGES = 10
        N_CHANNEL = 3
        IMAGE_WIDTH = IMAGE_HEIGHT = 608
        PATCH_SIZE = 128
        STRIDE = 16
        N_PATCHES = 10 * 31 * 31

        random_rgb_image = np.ones((N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)

        patches = patches.reshape((N_IMAGES, int(N_PATCHES / N_IMAGES), PATCH_SIZE, PATCH_SIZE, N_CHANNEL))

        reconstructed_images = images.images_from_patches(patches, stride=STRIDE, normalize=False)

        num_images, image_height, image_width, n_channel = reconstructed_images.shape

        self.assertTrue(num_images == N_IMAGES)
        self.assertTrue(image_height == IMAGE_HEIGHT)
        self.assertTrue(image_width == IMAGE_WIDTH)
        self.assertTrue(n_channel == N_CHANNEL)

        reconstructed_images
        plt.imsave("bla.png", reconstructed_images[0])

        print(reconstructed_images[0, 50, 100, 0])

    def test_400_400_image_to_patches_no_stride(self):
        N_IMAGES = 10
        N_CHANNEL = 3
        IMAGE_WIDTH = IMAGE_HEIGHT = 400
        PATCH_SIZE = 80
        STRIDE = 80
        N_PATCHES = 5 * 5 * 10

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)
        patches = patches.reshape((N_IMAGES, int(N_PATCHES / N_IMAGES), PATCH_SIZE, PATCH_SIZE, N_CHANNEL))

        n_images, n_patches, p_height, p_width, channels = patches.shape

        assert n_images == 10
        assert n_patches == 25
        assert p_height == 80
        assert p_width == 80
        assert channels == 3

        reconstructed_images = images.images_from_patches(patches)
        num_images, image_height, image_width, n_channel = reconstructed_images.shape
        self.assertTrue(num_images == N_IMAGES)
        self.assertTrue(image_height == IMAGE_HEIGHT)
        self.assertTrue(image_width == IMAGE_WIDTH)
        self.assertTrue(n_channel == N_CHANNEL)

    def test_visual(self):
        import matplotlib.image as mpimg
        test_image = mpimg.imread(r".\..\data\training\images\satImage_001.png")
        images_ = np.empty((2, 400, 400, 3))
        images_[0] = test_image.copy()
        images_[1] = test_image.copy()
        patches = images.extract_patches(images_, 80, stride=16)
        patches = patches.reshape((2, 21 * 21, 80, 80, 3))
        reconstructed_images = images.images_from_patches(patches, border_majority_only=True, stride=16)

        plt.imsave("bla.png", reconstructed_images[0])


class TestImages(TestCase):
    def test_predictions_to_patches(self):
        predictions = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        result = images.predictions_to_patches(predictions, 2)
        zero_patch = np.array([[[0], [0]], [[0], [0]]])
        one_patch = np.array([[[1], [1]], [[1], [1]]])
        expected = np.array([zero_patch, one_patch, one_patch, zero_patch, one_patch, one_patch, one_patch, zero_patch])
        self.assertEqual(result.shape, (8, 2, 2, 1))
        self.assertTrue((result == expected).all())
