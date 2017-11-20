from unittest import TestCase

import numpy as np

import images


class TestImages(TestCase):
    def test_predictions_to_patches(self):
        predictions = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        result = images.predictions_to_patches(predictions, 2)
        zero_patch = np.array([[[0], [0]], [[0], [0]]])
        one_patch = np.array([[[1], [1]], [[1], [1]]])
        expected = np.array([zero_patch, one_patch, one_patch, zero_patch, one_patch, one_patch, one_patch, zero_patch])
        self.assertEqual(result.shape, (8, 2, 2, 1))
        self.assertTrue((result == expected).all())
