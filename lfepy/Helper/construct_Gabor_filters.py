import cupy as cp
import numpy as np


def construct_Gabor_filters(num_of_orient, num_of_scales, size1, fmax=0.25,
                            ni=cp.sqrt(2), gamma=cp.sqrt(2), separation=cp.sqrt(2)):
    """
    Constructs a bank of Gabor filters using CuPy for GPU acceleration.

    Args:
        num_of_orient (int): Number of orientations.
        num_of_scales (int): Number of scales.
        size1 (int or tuple): Size of the filters. Can be an integer for square filters or a tuple for rectangular filters.
        fmax (float, optional): Maximum frequency. Default is 0.25.
        ni (float, optional): Bandwidth parameter. Default is sqrt(2).
        gamma (float, optional): Aspect ratio. Default is sqrt(2).
        separation (float, optional): Frequency separation factor. Default is sqrt(2).

    Returns:
        dict: A dictionary containing the spatial and frequency representations of the Gabor filters.
              The dictionary has the following keys:
              'spatial': A 2D array where each element is a 2D array representing the spatial domain Gabor filter.
              'freq': A 2D array where each element is a 2D array representing the frequency domain Gabor filter.
              'scales': The number of scales used.
              'orient': The number of orientations used.

    Raises:
        ValueError: If 'size1' is neither an integer nor a tuple of length 2.
    """
    # Initialize the filter bank
    filter_bank = {
        'spatial': np.empty((num_of_scales, num_of_orient), dtype=object),
        'freq': np.empty((num_of_scales, num_of_orient), dtype=object),
        'scales': num_of_scales,
        'orient': num_of_orient
    }

    # Check and adjust the size input
    if isinstance(size1, int):
        size1 = (size1, size1)
    elif len(size1) == 2:
        size1 = tuple(size1)
    else:
        raise ValueError("The parameter determining the size of the filters is not valid.")

    sigma_x = size1[1]
    sigma_y = size1[0]

    # Create meshgrid for x and y coordinates
    X, Y = cp.meshgrid(cp.arange(-sigma_x, sigma_x), cp.arange(-sigma_y, sigma_y))

    # Vectorize the filter creation
    for u in range(num_of_scales):  # for each scale
        fu = fmax / (separation ** u)
        alfa = fu / gamma
        beta = fu / ni

        for v in range(num_of_orient):  # for each orientation
            theta_v = (v / num_of_orient) * cp.pi

            # Rotate coordinates using the orientation angle
            X_rot = X * cp.cos(theta_v) + Y * cp.sin(theta_v)
            Y_rot = -X * cp.sin(theta_v) + Y * cp.cos(theta_v)

            # Compute the Gabor filter using vectorized operations
            gabor = (fu**2 / (cp.pi * gamma * ni)) * cp.exp(-(alfa**2 * X_rot**2 + beta**2 * Y_rot**2)) * cp.exp(1j * 2 * cp.pi * fu * X_rot)

            # Ensure the filter size matches the expected size
            filter_bank['spatial'][u, v] = gabor
            filter_bank['freq'][u, v] = cp.fft.fft2(gabor)

    return filter_bank