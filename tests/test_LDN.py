import unittest
import cupy as cp
from lfepy.Descriptor import LDN


class TestLDN(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing (e.g., 8x8 grayscale image) using CuPy
        self.image = cp.array([
            [52, 55, 61, 59, 79, 61, 76, 61],
            [62, 59, 55, 104, 94, 85, 59, 71],
            [63, 65, 66, 113, 144, 104, 63, 72],
            [64, 70, 70, 126, 154, 109, 71, 69],
            [67, 73, 68, 106, 122, 88, 68, 68],
            [68, 79, 60, 70, 77, 66, 58, 75],
            [69, 85, 64, 58, 55, 61, 65, 83],
            [70, 87, 69, 68, 65, 73, 78, 90]
        ], dtype=cp.uint8)

    def test_ldn_default_mode(self):
        # Test LDN with default parameters
        ldn_hist, imgDesc = LDN(self.image)
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(ldn_hist.ndim, 1)  # Should be a 1D array

    def test_ldn_histogram_mode(self):
        # Test LDN with histogram mode ('h')
        ldn_hist, imgDesc = LDN(self.image, mode='h')
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(ldn_hist.ndim, 1)  # Should be a 1D array

    def test_ldn_normalization_mode(self):
        # Test if the LDN histogram is normalized in 'nh' mode
        ldn_hist, _ = LDN(self.image, mode='nh')
        self.assertAlmostEqual(cp.sum(ldn_hist).get(), 1.0)  # `.get()` to move the result to CPU

    def test_ldn_invalid_mode(self):
        # Test LDN with an invalid mode
        with self.assertRaises(ValueError):
            LDN(self.image, mode='invalid_mode')

    def test_ldn_invalid_mask(self):
        # Test LDN with an invalid mask
        with self.assertRaises(ValueError):
            LDN(self.image, mask='invalid_mask')

    def test_ldn_gaussian_mask(self):
        # Test LDN with Gaussian mask
        ldn_hist, imgDesc = LDN(self.image, mask='gaussian', start=0.5)
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_ldn_kirsch_mask(self):
        # Test LDN with Kirsch mask
        ldn_hist, imgDesc = LDN(self.image, mask='kirsch', msize=3)
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_ldn_prewitt_mask(self):
        # Test LDN with Prewitt mask
        ldn_hist, imgDesc = LDN(self.image, mask='prewitt')
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_ldn_sobel_mask(self):
        # Test LDN with Sobel mask
        ldn_hist, imgDesc = LDN(self.image, mask='sobel')
        self.assertIsInstance(ldn_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_ldn_with_none_image(self):
        # Test LDN with None as input
        with self.assertRaises(TypeError):
            LDN(None)

    def test_ldn_with_non_array_image(self):
        # Test LDN with a non-CuPy array image
        with self.assertRaises(TypeError):
            LDN("invalid_image")

    def test_ldn_shape(self):
        # Ensure that the LDN descriptor shape is correctly handled
        ldn_hist, imgDesc = LDN(self.image)
        self.assertTrue(len(imgDesc) > 0)  # Ensure that imgDesc contains at least one descriptor

    def test_ldn_feature_extraction(self):
        # Check if the feature extraction part of LDN works
        ldn_hist, imgDesc = LDN(self.image)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue('fea' in imgDesc[0])  # Check if 'fea' key is in the first descriptor dictionary
        unique_values = cp.unique(imgDesc[0]['fea'])
        self.assertTrue(
            cp.all(cp.in1d(unique_values, cp.arange(1, 63)))  # Check if feature values are within expected range
        )


if __name__ == '__main__':
    unittest.main()
