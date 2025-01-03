import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LDiP(image, **kwargs):
    """
    Compute Local Directional Pattern (LDiP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LDiP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LDiP_hist (cupy.ndarray): Histogram(s) of LDiP descriptors.
            imgDesc (cupy.ndarray): LDiP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LDiP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Jabid, M.H. Kabir, and O. Chae,
        Local Directional Pattern (LDP) – A Robust Image Descriptor for Object Recognition,
        Advanced Video and Signal Based Surveillance (AVSS), 2010 Seventh IEEE International Conference on, IEEE,
        2010, pp. 482-487.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define Kirsch Masks
    Kirsch = [cp.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              cp.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              cp.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              cp.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              cp.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              cp.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              cp.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              cp.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute mask responses
    maskResponses = cp.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch):
        maskResponses[:, :, i] = cp.abs(convolve2d(image, kirsch_mask, mode='same'))

    # Sort responses and construct binary pattern
    ind = cp.argsort(maskResponses, axis=2)[:, :, ::-1]
    bit8array = cp.zeros((image.shape[0], image.shape[1], 8))
    bit8array[cp.logical_or(cp.logical_or(ind == 0, ind == 1), ind == 2)] = 1
    imgDesc = cp.zeros_like(image)
    for r in range(image.shape[0]):
        codebit = cp.reshape(bit8array[r, :, 7::-1], (image.shape[1], -1))
        imgDesc[r, :] = cp.packbits(codebit.astype(bool)).flatten()

    # Define unique bins for histogram
    uniqueBin = cp.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56, 67, 69,
                          70, 73, 74, 76, 81, 82, 84, 88, 97, 98, 100, 104, 112, 131, 133, 134, 137, 138, 140,
                          145, 146, 148, 152, 161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Set binVec option
    options['binVec'] = uniqueBin

    # Compute LDiP histogram
    LDiP_hist = cp.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LDiP_hist[i] = cp.sum(imgDesc == bin_val)
    if 'mode' in options and options['mode'] == 'nh':
        LDiP_hist = LDiP_hist / cp.sum(LDiP_hist)

    return LDiP_hist, imgDesc