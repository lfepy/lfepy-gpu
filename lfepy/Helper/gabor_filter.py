import cupy as cp
from lfepy.Helper.construct_Gabor_filters import construct_Gabor_filters
from lfepy.Helper.filter_image_with_Gabor_bank import filter_image_with_Gabor_bank


def gabor_filter(image, orienNum, scaleNum):
    """
    Apply a Gabor filter bank to an image and organize the results into a multidimensional array.
    This version uses vectorized operations for better performance.

    Args:
        image (cp.ndarray): Input image to be filtered. Should be a 2D CuPy array.
        orienNum (int): Number of orientation filters in the Gabor filter bank.
        scaleNum (int): Number of scale filters in the Gabor filter bank.

    Returns:
        cp.ndarray: Multidimensional array containing the Gabor magnitude responses. Shape is (height, width, orienNum, scaleNum).

    Example:
        >>> import numpy as np
        >>> from skimage.data import camera
        >>> image = camera()
        >>> gabor_magnitudes = gabor_filter(image, orienNum=8, scaleNum=5)
        >>> print(gabor_magnitudes.shape)
        (512, 512, 8, 5)
    """
    r, c = image.shape

    # Construct Gabor filter bank
    filter_bank = construct_Gabor_filters(orienNum, scaleNum, [r, c])
    # Apply Gabor filter bank to the image
    result = filter_image_with_Gabor_bank(image, filter_bank, 1)

    # Calculate number of pixels in each filter response
    pixel_num = len(result) // (orienNum * scaleNum)

    # Reshape result to (orienNum * scaleNum, r, c)
    reshaped_result = result.reshape(orienNum * scaleNum, r, c)

    # Create orientation and scale indices
    orien_indices = cp.arange(orienNum * scaleNum) % orienNum
    scale_indices = cp.arange(orienNum * scaleNum) // orienNum

    # Initialize the output array
    gaborMag = cp.zeros((r, c, orienNum, scaleNum))

    # Use advanced indexing to assign values
    gaborMag[:, :, orien_indices, scale_indices] = reshaped_result.transpose(1, 2, 0)

    return gaborMag