import unittest
import cupy as cp
from lfepy.Descriptor import MBP


class TestMBP(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing using CuPy (GPU arrays)
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

    def test_mbp_default_params(self):
        # Test MBP with default parameters (on GPU)
        mbp_hist, imgDesc = MBP(self.image)
        self.assertIsInstance(mbp_hist, cp.ndarray)  # CuPy array
        self.assertIsInstance(imgDesc, cp.ndarray)  # CuPy array
        self.assertTrue(len(mbp_hist) > 0)
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_mbp_custom_mode(self):
        # Test MBP with custom mode (on GPU)
        mbp_hist, imgDesc = MBP(self.image, mode='h')
        self.assertIsInstance(mbp_hist, cp.ndarray)  # CuPy array
        self.assertIsInstance(imgDesc, cp.ndarray)  # CuPy array
        self.assertTrue(len(mbp_hist) > 0)
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_mbp_invalid_mode(self):
        # Test MBP with an invalid mode
        with self.assertRaises(ValueError):
            MBP(self.image, mode='invalid_mode')

    def test_mbp_with_none_image(self):
        # Test MBP with None as input
        with self.assertRaises(TypeError):
            MBP(None)

    def test_mbp_with_non_array_image(self):
        # Test MBP with a non-CuPy array image
        with self.assertRaises(TypeError):
            MBP("invalid_image")

    def test_mbp_feature_extraction(self):
        # Check if the feature extraction part of MBP works
        mbp_hist, imgDesc = MBP(self.image)
        self.assertTrue(len(mbp_hist) > 0)
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertTrue(cp.issubdtype(imgDesc.dtype, cp.integer))  # Ensure imgDesc contains integer data

if __name__ == '__main__':
    unittest.main()
