import cupy as cp


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

    # Create arrays for scales and orientations
    scales = cp.arange(num_of_scales)
    orientations = cp.arange(num_of_orient)

    # Calculate frequencies for all scales
    frequencies = fmax / (separation ** scales)

    # Calculate alpha and beta for all scales
    alphas = frequencies / gamma
    betas = frequencies / ni

    # Calculate orientation angles for all orientations
    thetas = (orientations / num_of_orient) * cp.pi

    # Reshape X and Y for broadcasting with orientations
    X = X[None, None, :, :]  # Shape: (1, 1, height, width)
    Y = Y[None, None, :, :]  # Shape: (1, 1, height, width)

    # Reshape thetas for broadcasting
    cos_theta = cp.cos(thetas)[None, :, None, None]  # Shape: (1, num_of_orient, 1, 1)
    sin_theta = cp.sin(thetas)[None, :, None, None]  # Shape: (1, num_of_orient, 1, 1)

    # Calculate rotated coordinates for all orientations at once using broadcasting
    X_rot = X * cos_theta + Y * sin_theta  # Shape: (1, num_of_orient, height, width)
    Y_rot = -X * sin_theta + Y * cos_theta  # Shape: (1, num_of_orient, height, width)

    # Calculate Gabor filters for all scales and orientations
    # Reshape arrays for broadcasting
    alphas = alphas[:, None, None, None]
    betas = betas[:, None, None, None]
    frequencies = frequencies[:, None, None, None]

    # Compute the Gabor filter using vectorized operations
    gabor = ((frequencies ** 2 / (cp.pi * gamma * ni)) *
             cp.exp(-(alphas ** 2 * X_rot ** 2 + betas ** 2 * Y_rot ** 2)) *
             cp.exp(1j * 2 * cp.pi * frequencies * X_rot))

    # Compute FFT for all filters at once
    gabor_freq = cp.fft.fft2(gabor)

    # Create filter bank with 4D arrays
    filter_bank = {
        'spatial': gabor,
        'freq': gabor_freq,
        'scales': num_of_scales,
        'orient': num_of_orient
    }

    return filter_bank