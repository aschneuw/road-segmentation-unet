from unittest import TestCase

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

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)

        n_patches, p_height, p_width, channels = patches.shape

        assert n_patches == 10*31*31
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

        random_rgb_image = np.random.randint(0, PIXEL_DEPTH, size=(N_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL))
        patches = images.extract_patches(random_rgb_image, PATCH_SIZE, STRIDE)






class TestImages(TestCase):
    def test_predictions_to_patches(self):
        predictions = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        result = images.predictions_to_patches(predictions, 2)
        zero_patch = np.array([[[0], [0]], [[0], [0]]])
        one_patch = np.array([[[1], [1]], [[1], [1]]])
        expected = np.array([zero_patch, one_patch, one_patch, zero_patch, one_patch, one_patch, one_patch, zero_patch])
        self.assertEqual(result.shape, (8, 2, 2, 1))
        self.assertTrue((result == expected).all())
