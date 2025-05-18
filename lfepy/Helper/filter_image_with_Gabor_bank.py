import cupy as cp
from lfepy.Helper.construct_Gabor_filters import construct_Gabor_filters
from skimage.transform import resize


def filter_image_with_Gabor_bank(image, filter_bank, down_sampling_factor=64):
    """
    Apply a Gabor filter bank to an image and return the filtered features.

    This function applies a bank of Gabor filters to an input image, performs down-sampling,
    and returns the concatenated features obtained from the filtered image. Gabor's filters are
    used for texture analysis and feature extraction in image processing.

    Args:
        image (np.ndarray): Input image to be filtered. Should be a 2D numpy array representing a grayscale image.
        filter_bank (dict): Dictionary containing Gabor filter bank with the following keys:
            'spatial': A list of 2D arrays representing spatial domain Gabor filters.
            'freq': A list of 2D arrays representing frequency domain Gabor filters.
            'orient': Number of orientations in the filter bank.
            'scales': Number of scales in the filter bank.
        down_sampling_factor (int, optional): Factor for down-sampling the filtered images. Default is 64.

    Returns:
        np.ndarray: Concatenated filtered features from the Gabor filter bank, flattened into a 1D array.

    Raises:
        ValueError: If the inputs are not as expected, dimensions do not match, or required fields are missing in the filter bank.

    Example:
        >>> import numpy as np
        >>> from skimage.data import camera
        >>> from skimage.transform import resize
        >>> from scipy.fftpack import fft2, ifft2
        >>> image = camera()
        >>> filter_bank = construct_Gabor_filters(num_of_orient=8, num_of_scales=5, size1=31)
        >>> features = filter_image_with_Gabor_bank(image, filter_bank)
        >>> print(features.shape)
    """
    # Check inputs
    if not isinstance(image, cp.ndarray):
        raise ValueError("The first input parameter must be an image in the form of a CuPy array.")

    if not isinstance(filter_bank, dict):
        raise ValueError("The second input parameter must be a dictionary containing the Gabor filter bank.")

    if down_sampling_factor is None:
        down_sampling_factor = 64

    if not isinstance(down_sampling_factor, (int, float)) or down_sampling_factor < 1:
        print("The down-sampling factor needs to be a numeric value larger or equal than 1! Switching to defaults: 64")
        down_sampling_factor = 64

    # Check filter bank fields
    required_fields = ['spatial', 'freq', 'orient', 'scales']
    if not all(key in filter_bank for key in required_fields):
        raise ValueError("Filter bank missing required fields!")

    # Check image and filter size
    a, b = image.shape
    c, d = filter_bank['spatial'].shape[-2:]

    if a == 2 * c or b == 2 * d:
        raise ValueError("The dimension of the input image and Gabor filters do not match!")

    # Compute output size
    dim_spec_down_sampl = int(cp.round(cp.sqrt(down_sampling_factor)).get())
    new_size = (a // dim_spec_down_sampl, b // dim_spec_down_sampl)

    # Filter image in the frequency domain
    image_tmp = cp.zeros((2 * a, 2 * b))
    image_tmp[:a, :b] = image
    image_fft = cp.fft.fft2(image_tmp)

    # Reshape image_fft for broadcasting with filters
    image_fft = image_fft[None, None, :, :]  # Shape: (1, 1, 2*a, 2*b)

    # Filter image in frequency domain for all scales and orientations at once
    filtered_fft = filter_bank['freq'] * image_fft

    # Inverse FFT for all filters at once
    filtered_spatial = cp.fft.ifft2(filtered_fft)

    # Extract the relevant portion and compute magnitude
    # Shape: (num_of_scales, num_of_orient, a, b)
    gabout = cp.abs(filtered_spatial[:, :, a:2 * a, b:2 * b])

    # First, reshape to combine scales and orientations
    gabout_reshaped = gabout.reshape(-1, a, b)

    # Create output array for down-sampled images
    downsampled = cp.zeros((gabout_reshaped.shape[0], *new_size))

    # Perform down-sampling using array operations
    for i in range(gabout_reshaped.shape[0]):
        # Reshape the image into blocks
        blocks = gabout_reshaped[i].reshape(a // dim_spec_down_sampl, dim_spec_down_sampl,
                                            b // dim_spec_down_sampl, dim_spec_down_sampl)
        # Take mean of each block
        downsampled[i] = cp.mean(blocks, axis=(1, 3))

    # Reshape back to (num_of_scales, num_of_orient, new_height, new_width)
    downsampled = downsampled.reshape(filter_bank['scales'], filter_bank['orient'], *new_size)

    # Zero mean unit variance normalization for all filters at once
    mean = cp.mean(downsampled, axis=(-2, -1), keepdims=True)
    std = cp.std(downsampled, axis=(-2, -1), keepdims=True)
    normalized = (downsampled - mean) / std

    # Flatten and concatenate all features
    filtered_image = normalized.reshape(-1)

    return filtered_image