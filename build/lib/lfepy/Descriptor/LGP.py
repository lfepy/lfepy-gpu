import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LGP(image, **kwargs):
    """
    Compute Local Gradient Pattern (LGP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGP_hist (numpy.ndarray): Histogram(s) of LGP descriptors.
            imgDesc (numpy.ndarray): LGP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Local Gradient Pattern-A Novel Feature Representation for Facial Expression Recognition,
        Journal of AI and Data Mining 2,
        (2014), pp. 33-38.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Compute binary patterns for four pairs of pixels
    a1a3 = ((image[:-2, :-2]) - (image[2:, 2:]) > 0)
    a2a4 = ((image[:-2, 2:]) - (image[2:, :-2]) > 0)
    path1 = a1a3 * 2 + a2a4 * 1

    b1b3 = ((image[:-2, 1:-1]) - (image[2:, 1:-1]) > 0)
    b2b4 = ((image[1:-1, 2:]) - (image[1:-1, :-2]) > 0)
    path2 = b1b3 * 2 + b2b4 * 1 + 4

    # Combine paths to form the final descriptor
    imgDesc = path1 + path2

    # Set bin vectors
    options['binVec'] = cp.arange(4, 11)

    # Compute LGP histogram
    LGP_hist = cp.zeros(len(options['binVec']))
    LGP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LGP_hist = LGP_hist / cp.sum(LGP_hist)

    return LGP_hist, imgDesc