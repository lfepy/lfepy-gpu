import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LMP(image, **kwargs):
    """
    Compute Local Monotonic Pattern (LMP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LMP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LMP_hist (cupy.ndarray): Histogram(s) of LMP descriptors.
            imgDesc (cupy.ndarray): LMP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LMP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Mohammad, and M.L. Ali,
        Robust Facial Expression Recognition Based on Local Monotonic Pattern (LMP),
        Computer and Information Technology (ICCIT), 2011 14th International Conference on, IEEE,
        2011, pp. 572-576.
    """
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define link list for LMP computation
    link = [[[3, 4], [3, 5]],
            [[2, 4], [1, 5]],
            [[2, 3], [1, 3]],
            [[2, 2], [1, 1]],
            [[3, 2], [3, 1]],
            [[4, 2], [5, 1]],
            [[4, 3], [5, 3]],
            [[4, 4], [5, 5]]]

    # Compute LMP descriptors
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape

    # Initialize LMP descriptor matrix
    imgDesc = cp.zeros((rSize, cSize), dtype=cp.float64)

    for n in range(8):
        corner = link[n]
        x_i1 = image[corner[0][0] - 1:corner[0][0] + rSize - 1, corner[0][1] - 1:corner[0][1] + cSize - 1]
        x_i2 = image[corner[1][0] - 1:corner[1][0] + rSize - 1, corner[1][1] - 1:corner[1][1] + cSize - 1]
        imgDesc += ((((x_i1 - x_c) >= 0) & ((x_i2 - x_i1) >= 0)) * 2 ** (8 - n - 1))

    # Set bin vectors
    options['binVec'] = cp.arange(256)  # CuPy array

    # Compute LMP histogram
    LMP_hist = cp.zeros(len(options['binVec']), dtype=cp.float64)  # CuPy array
    for i, bin_val in enumerate(options['binVec']):
        LMP_hist[i] = cp.sum(imgDesc == bin_val)  # CuPy sum operation
    if 'mode' in options and options['mode'] == 'nh':
        LMP_hist = LMP_hist / cp.sum(LMP_hist)  # Normalize histogram

    return LMP_hist, imgDesc