import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LDiPv(image, **kwargs):
    """
    Compute Local Directional Pattern Variance (LDiPv) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LDiPv extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LDiPv_hist (cupy.ndarray): Histogram(s) of LDiPv descriptors.
            imgDesc (cupy.ndarray): LDiPv descriptors themselves.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LDiPv(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.H. Kabir, T. Jabid, and O. Chae,
        A Local Directional Pattern Variance (LDPv) Based Face Descriptor for Human Facial Expression Recognition,
        Advanced Video and Signal Based Surveillance (AVSS), 2010 Seventh IEEE International Conference on, IEEE,
        2010, pp. 526-532.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define Kirsch masks
    Kirsch = [cp.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              cp.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              cp.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              cp.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              cp.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              cp.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              cp.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              cp.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute the response of the image to each Kirsch mask
    maskResponses = cp.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch, start=1):
        maskResponses[:, :, i - 1] = convolve2d(image, kirsch_mask, mode='same')

    # Take the absolute value of the mask responses
    maskResponsesAbs = cp.abs(maskResponses)

    # Sort the mask responses to find the strongest responses
    ind = cp.argsort(maskResponsesAbs, axis=2)[:, :, ::-1]

    # Create a binary 8-bit array based on the top 3 strongest responses
    bit8array = cp.zeros((image.shape[0], image.shape[1], 8))
    bit8array[(ind == 0) | (ind == 1) | (ind == 2)] = 1

    # Generate the LDiPv descriptor for each pixel
    imgDesc = cp.zeros_like(image)
    flipped_bit8array = cp.flip(bit8array, axis=2)  # Reverse the third dimension
    reshaped_bit8array = cp.reshape(flipped_bit8array, (image.shape[0], image.shape[1], -1))
    power_matrix = cp.power(2, cp.arange(reshaped_bit8array.shape[2] - 1, -1, -1))
    imgDesc = cp.dot(reshaped_bit8array, power_matrix)

    # Define the unique bins for the histogram
    uniqueBin = cp.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56, 67, 69,
                          70, 73, 74, 76, 81, 82, 84, 88, 97, 98, 100, 104, 112, 131, 133, 134, 137, 138, 140, 145,
                          146, 148, 152, 161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Compute the variance of the mask responses
    varianceImg = cp.var(maskResponsesAbs, axis=2)
    options['weight'] = varianceImg
    options['binVec'] = uniqueBin

    # Compute LDiPv histogram
    LDiPv_hist = cp.zeros(len(options['binVec']))
    imgDesc_flat = cp.ravel(imgDesc)
    weight_flat = cp.ravel(options['weight'])
    bin_indices = cp.searchsorted(options['binVec'], imgDesc_flat)
    LDiPv_hist = cp.bincount(bin_indices, weights=weight_flat, minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LDiPv_hist = LDiPv_hist / cp.sum(LDiPv_hist)

    return LDiPv_hist, imgDesc