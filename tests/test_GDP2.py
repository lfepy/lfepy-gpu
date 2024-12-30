import unittest
import cupy as cp
from lfepy.Descriptor import GDP2


class TestGDP2(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing (e.g., 8x8 grayscale image)
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

    def test_gdp2_default_mode(self):
        # Test GDP2 with default mode
        gdp2_hist, imgDesc = GDP2(self.image)
        self.assertIsInstance(gdp2_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, cp.ndarray)
        self.assertEqual(gdp2_hist.ndim, 1)  # Should be a 1D array

    def test_gdp2_histogram_mode(self):
        # Test GDP2 with histogram mode ('h')
        gdp2_hist, imgDesc = GDP2(self.image, mode='h')
        self.assertIsInstance(gdp2_hist, cp.ndarray)
        self.assertIsInstance(imgDesc, cp.ndarray)
        self.assertEqual(gdp2_hist.ndim, 1)  # Should be a 1D array

    def test_gdp2_normalization_mode(self):
        # Test if the GDP2 histogram is normalized in 'nh' mode
        gdp2_hist, _ = GDP2(self.image, mode='nh')
        self.assertAlmostEqual(cp.sum(gdp2_hist).get(), 1.0)

    def test_gdp2_invalid_mode(self):
        # Test GDP2 with an invalid mode
        with self.assertRaises(ValueError):
            GDP2(self.image, mode='invalid_mode')

    def test_gdp2_with_none_image(self):
        # Test GDP2 with None as input
        with self.assertRaises(TypeError):
            GDP2(None)

    def test_gdp2_with_non_array_image(self):
        # Test GDP2 with a non-CuPy array image
        with self.assertRaises(TypeError):
            GDP2("invalid_image")

    def test_gdp2_shape(self):
        # Ensure that the image shape is correctly handled
        gdp2_hist, imgDesc = GDP2(self.image)
        expected_shape = (self.image.shape[0] - 2, self.image.shape[1] - 2)  # Image reduced by border
        self.assertEqual(imgDesc.shape, expected_shape)

    def test_gdp2_selected_bins(self):
        # Check if the selected bins in GDP2 histogram match the expected bins
        gdp2_hist, _ = GDP2(self.image, mode='nh')
        transitionSelected = [0, 1, 3, 7, 8, 12, 14, 15]
        self.assertEqual(gdp2_hist.size, len(transitionSelected))


if __name__ == '__main__':
    unittest.main()
