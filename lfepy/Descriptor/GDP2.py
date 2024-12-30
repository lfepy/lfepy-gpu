import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def GDP2(image, **kwargs):
    """
    Compute Gradient Direction Pattern (GDP2) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing GDP2 extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            GDP2_hist (cupy.ndarray): Histogram(s) of GDP2 descriptors.
            imgDesc (cupy.ndarray): GDP2 descriptors.

    Raises:
        TypeError: If the `image` is not a valid `cupy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option ('nh' or 'h').

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = GDP2(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Gender Classification using Gradient Direction Pattern,
        in Science International,
        vol. 25, 2013.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define link list
    linkList = cp.asarray([[[1, 1], [3, 3]], [[1, 2], [3, 2]], [[1, 3], [3, 1]], [[2, 3], [2, 1]]])

    # Compute pattern
    x_c = image[1:-1, 1:-1]
    rSize, cSize = x_c.shape

    # Initialize pattern with zeros
    pattern = cp.zeros_like(x_c)

    for n in range(len(linkList)):
        corner1 = linkList[n][0]
        corner2 = linkList[n][1]
        x_1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]

        pattern += (((x_1 - x_2) >= 0) * 2 ** (len(linkList) - n - 1))

    imgDesc = pattern

    binNum = cp.array(2 ** len(linkList))
    transitionSelected = cp.array([0, 1, 3, 7, 8, 12, 14, 15])
    options['selected'] = transitionSelected

    # Set bin vectors
    options['binVec'] = cp.arange(binNum)

    # Compute GDP2 histogram
    GDP2_hist = cp.zeros(len(options['binVec']))
    for i, bin_val in enumerate(cp.asarray(options['binVec'])):
        GDP2_hist[i] = cp.sum(imgDesc == bin_val)
    GDP2_hist = GDP2_hist[transitionSelected]

    if 'mode' in options and options['mode'] == 'nh':
        GDP2_hist = GDP2_hist / cp.sum(GDP2_hist)

    return GDP2_hist, imgDesc