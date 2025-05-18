import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Helper import descriptor_LPQ
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_windowSize


def LPQ(image, **kwargs):
    """
    Compute Local Phase Quantization (LPQ) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LPQ extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            windowSize (int): Size of the sliding window for LPQ. Default is 5.

    Returns:
        tuple: A tuple containing:
            LPQ_hist (cupy.ndarray): Histogram of LPQ descriptors.
            imgDesc (cupy.ndarray): LPQ descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray` or `cupy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LPQ(image, mode='nh', windowSize=5)

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        V. Ojansivu, and J. Heikkilä,
        Blur Insensitive Texture Classification Using Local Phase Quantization,
        International Conference on Image and Signal Processing, Springer,
        2008, pp. 236-243.

        A. Dhall, A. Asthana, R. Goecke, and T. Gedeon,
        Emotion Recognition Using PHOG and LPQ Features,
        Automatic Face & Gesture Recognition and Workshops (FG 2011), IEEE,
        2011, pp. 878-883.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    wSz = validate_windowSize(options)

    imgDesc, _ = descriptor_LPQ(image, wSz)

    options['binVec'] = cp.arange(256)

    # Compute LPQ histogram
    LPQ_hist = cp.zeros(len(options['binVec']))
    LPQ_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    # Normalize histogram if required
    if 'mode' in options and options['mode'] == 'nh':
        LPQ_hist = LPQ_hist / cp.sum(LPQ_hist)

    return LPQ_hist, imgDesc