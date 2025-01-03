import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def MBP(image, **kwargs):
    """
    Compute Median Binary Pattern (MBP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MBP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            MBP_hist (cupy.ndarray): Histogram(s) of MBP descriptors.
            imgDesc (cupy.ndarray): MBP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MBP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust facial expression recognition based on median ternary pattern (MTP),
        Electrical Information and Communication Technology (EICT), 2013 International Conference on, IEEE,
        2014, pp. 1-5.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for MBP computation
    link = cp.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]], dtype=cp.int32)
    ImgIntensity = cp.zeros((rSize * cSize, link.shape[0]), dtype=cp.float64)

    # Compute MBP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        x_slice = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        ImgIntensity[:, n] = x_slice.reshape(-1)

    medianMat = cp.median(ImgIntensity, axis=1)
    MBP = (ImgIntensity > medianMat.reshape(-1, 1))

    imgDesc = cp.dot(MBP.astype(cp.uint8), 1 << cp.arange(MBP.shape[1] - 1, -1, -1)).reshape(rSize, cSize)

    # Set bin vectors
    options['binVec'] = cp.arange(256)  # CuPy array for bin vectors

    # Compute MBP histogram
    MBP_hist = cp.zeros(len(options['binVec']), dtype=cp.float64)  # CuPy array for histogram
    for i, bin_val in enumerate(options['binVec']):
        MBP_hist[i] = cp.sum(imgDesc == bin_val)  # CuPy sum operation

    if 'mode' in options and options['mode'] == 'nh':
        MBP_hist = MBP_hist / cp.sum(MBP_hist)  # Normalize histogram using CuPy sum

    return MBP_hist, imgDesc