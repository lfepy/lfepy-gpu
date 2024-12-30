import unittest
import cupy as cp
from lfepy.Descriptor import LFD


class TestLFD(unittest.TestCase):

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

    def test_lfd_default_mode(self):
        # Test LFD with default parameters
        lfd_hist, imgDesc = LFD(self.image)
        self.assertIsInstance(lfd_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertEqual(lfd_hist.ndim, 1)  # Should be a 1D array

    def test_lfd_histogram_mode(self):
        # Test LFD with histogram mode ('h')
        lfd_hist, imgDesc = LFD(self.image, mode='h')
        self.assertIsInstance(lfd_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertEqual(lfd_hist.ndim, 1)  # Should be a 1D array

    def test_lfd_normalization_mode(self):
        # Test if the LFD histogram is normalized in 'nh' mode
        lfd_hist, _ = LFD(self.image, mode='nh')
        self.assertAlmostEqual(cp.sum(lfd_hist).get(), 1.0)  # `.get()` to transfer from GPU to CPU

    def test_lfd_invalid_mode(self):
        # Test LFD with an invalid mode
        with self.assertRaises(ValueError):
            LFD(self.image, mode='invalid_mode')

    def test_lfd_with_none_image(self):
        # Test LFD with None as input
        with self.assertRaises(TypeError):
            LFD(None)

    def test_lfd_with_non_array_image(self):
        # Test LFD with a non-CuPy array image
        with self.assertRaises(TypeError):
            LFD("invalid_image")

    def test_lfd_feature_extraction(self):
        # Check if the feature extraction part of LFD works
        lfd_hist, imgDesc = LFD(self.image)
        self.assertTrue(len(lfd_hist) > 0)
        self.assertEqual(len(imgDesc), 2)
        for desc in imgDesc:
            self.assertIn('fea', desc)
            self.assertEqual(desc['fea'].ndim, 2)
            self.assertEqual(desc['fea'].dtype, cp.float64)

if __name__ == '__main__':
    unittest.main()
