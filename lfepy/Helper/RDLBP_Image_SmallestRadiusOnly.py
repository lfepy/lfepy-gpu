import cupy as cp
from lfepy.Helper.cirInterpSingleRadiusNew import cirInterpSingleRadiusNew


def RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode):
    """
    Compute the Radial Difference Local Binary Pattern (RDLBP) for an image with a focus on the smallest radius.

    This function calculates the RDLBP descriptor for an image by comparing the radial differences between
    the original image and a smoothed version of the image. The descriptor is computed using a circular neighborhood
    defined by the given radius and number of points.

    Args:
        imgCenSmooth (numpy.ndarray): Smoothed image from which the radial difference is computed.
        img (numpy.ndarray): Original image for extracting circularly interpolated blocks.
        lbpRadius (int): Radius of the circular neighborhood for Local Binary Pattern (LBP).
        lbpPoints (int): Number of points used in the LBP pattern.
        mapping (dict or None): Optional mapping dictionary for converting LBP result to a different bin scheme.
            Must contain 'num' (number of bins) and 'table' (mapping from old bin to new bin). If None, no mapping is applied.
        mode (str): Output mode. 'h' or 'hist' for histogram of the RDLBP, 'nh' for normalized histogram.

    Returns:
        numpy.ndarray: RDLBP descriptor, either as a histogram or image depending on the `mode` parameter.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> imgCenSmooth = data.coins()
        >>> lbpRadius = 1
        >>> lbpPoints = 8
        >>> mapping = {'num': 256, 'table': np.arange(256)}
        >>> hist = RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode='nh')
        >>> print(hist.shape)
        (256,)
    """
    # Extract circularly interpolated blocks from the original image
    blocks1, dx, dy = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
    blocks1 = blocks1.T  # Transpose to match the expected shape

    # Adjust the smoothed image size based on the radius
    imgTemp = imgCenSmooth[lbpRadius:-lbpRadius, lbpRadius:-lbpRadius]

    # Create a tiled version of the smoothed image to match the size of the LBP blocks
    blocks2 = cp.tile(imgTemp.ravel(), (lbpPoints, 1)).T

    # Compute the radial difference between the blocks of the original image and the smoothed image
    radialDiff = blocks1 - blocks2
    radialDiff[radialDiff >= 0] = 1
    radialDiff[radialDiff < 0] = 0

    # Compute the LBP value by weighting the binary differences
    bins = 2 ** lbpPoints
    weight = 2 ** cp.arange(lbpPoints)  # Use CuPy for weights
    radialDiff = radialDiff * weight
    radialDiff = cp.sum(radialDiff, axis=1)

    # Reshape the result to match the dimensions of the original image
    result = radialDiff
    result = cp.reshape(result, (dx + 1, dy + 1))

    # Apply mapping if it is defined
    if isinstance(mapping, dict):
        bins = mapping['num']
        # Vectorized mapping using NumPy indexing
        result = mapping['table'][result.astype(cp.uint8)]

    # Return result as histogram or image depending on mode
    if mode in ['h', 'hist', 'nh']:
        hist_result = cp.histogram(result, bins=cp.arange(bins + 1))[0]
        if mode == 'nh':
            hist_result = hist_result / cp.sum(hist_result)
        return hist_result
    else:
        # Return result as matrix of unsigned integers
        max_val = bins - 1
        if max_val <= cp.iinfo(cp.uint8).max:
            return result.astype(cp.uint8)
        elif max_val <= cp.iinfo(cp.uint16).max:
            return result.astype(cp.uint16)
        else:
            return result.astype(cp.uint32)