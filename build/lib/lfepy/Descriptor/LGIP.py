import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LGIP(image, **kwargs):
    """
    Compute Local Gradient Increasing Pattern (LGIP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGIP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGIP_hist (cupy.ndarray): Histogram(s) of LGIP descriptors.
            imgDesc (cupy.ndarray): LGIP descriptors themselves.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGIP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        Z. Lubing, and W. Han,
        Local Gradient Increasing Pattern for Facial Expression Recognition,
        Image Processing (ICIP), 2012 19th IEEE International Conference on, IEEE,
        2012, pp. 2601-2604.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Get image dimensions
    r, c = image.shape

    # Compute LGIP components
    v000 = (-image[1:-3, 2:-2] + image[1:-3, 3:-1] - 2 * image[2:-2, 2:-2] +
            2 * image[2:-2, 3:-1] - image[3:-1, 2:-2] + image[3:-1, 3:-1] > 0)
    v001 = (-image[1:-3, 1:-3] + image[:-4, 2:-2] - 2 * image[2:-2, 2:-2] +
            2 * image[1:-3, 3:-1] - image[3:-1, 3:-1] + image[2:-2, 4:])
    v010 = (-image[2:-2, 1:-3] + image[1:-3, 1:-3] - 2 * image[2:-2, 2:-2] +
            2 * image[1:-3, 2:-2] - image[2:-2, 3:-1] + image[1:-3, 3:-1] > 0)
    v011 = (-image[3:-1, 1:-3] + image[2:-2, 0:-4] - 2 * image[2:-2, 2:-2] +
            2 * image[1:-3, 1:-3] - image[1:-3, 3:-1] + image[0:-4, 2:-2] > 0)
    v100 = (-image[1:-3, 2:-2] + image[1:-3, 1:-3] - 2 * image[2:-2, 2:-2] +
            2 * image[2:-2, 1:-3] - image[3:-1, 2:-2] + image[3:-1, 1:-3] > 0)
    v101 = (-image[1:-3, 1:-3] + image[2:-2, 0:-4] - 2 * image[2:-2, 2:-2] +
            2 * image[3:-1, 1:-3] - image[3:-1, 3:-1] + image[4:, 2:-2] > 0)
    v110 = (-image[2:-2, 1:-3] + image[3:-1, 1:-3] - 2 * image[2:-2, 2:-2] +
            2 * image[3:-1, 2:-2] - image[2:-2, 3:-1] + image[3:-1, 3:-1] > 0)
    v111 = (-image[3:-1, 1:-3] + image[4:, 2:-2] - 2 * image[2:-2, 2:-2] +
            2 * image[3:-1, 3:-1] - image[1:-3, 3:-1] + image[2:-2, 4:] > 0)

    # Compute Orientation Tensor Vectors (OTV)
    OTVx = cp.ravel(v000 + v001 + v111 - v011 - v100 - v101)
    OTVy = cp.ravel(v001 + v010 + v011 - v101 - v110 - v111)

    # Define pattern mask
    patternMask = cp.array([[-1, -1, 30, 29, 28, -1, -1],
                            [-1, 16, 15, 14, 13, 12, -1],
                            [31, 17, 4, 3, 2, 11, 27],
                            [32, 18, 5, 0, 1, 10, 26],
                            [33, 19, 6, 7, 8, 9, 25],
                            [-1, 20, 21, 22, 23, 24, -1],
                            [-1, -1, 34, 35, 36, -1, -1]])

    # Clip OTV values to be within the pattern mask range
    OTVx_clipped = cp.clip(OTVx + 4, 0, 6).astype(int)
    OTVy_clipped = cp.clip(OTVy + 4, 0, 6).astype(int)

    # Map the OTV values to pattern mask indices
    idx = cp.ravel_multi_index((OTVx_clipped, OTVy_clipped), patternMask.shape)
    LGIP = cp.take(patternMask, idx)  # Using cp.take for correct indexing
    imgDesc = cp.reshape(LGIP, (r - 4, c - 4))

    # Set bin vectors
    options['binVec'] = cp.arange(0, 37)

    # Compute LGIP histogram
    LGIP_hist = cp.zeros(len(options['binVec']))
    LGIP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LGIP_hist = LGIP_hist / cp.sum(LGIP_hist)

    return LGIP_hist, imgDesc