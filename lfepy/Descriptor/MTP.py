import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_t_MTP


def MTP(image, **kwargs):
    """
    Compute Median Ternary Pattern (MTP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            t (float): Threshold value for MTP computation. Default is 10.

    Returns:
        tuple: A tuple containing:
            MTP_hist (cupy.ndarray): Histogram(s) of MTP descriptors.
            imgDesc (list of dicts): List of dictionaries containing MTP descriptors for positive and negative thresholds.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MTP(image, mode='nh', t=10)

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
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
    t = validate_t_MTP(options)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for MTP computation
    link = cp.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]], dtype=cp.int32)
    ImgIntensity = cp.zeros((rSize * cSize, link.shape[0]), dtype=cp.float64)

    # Compute MTP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        x_slice = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        ImgIntensity[:, n] = x_slice.reshape(-1)

    medianMat = cp.median(ImgIntensity, axis=1)

    Pmtp = (ImgIntensity > (medianMat + t).reshape(-1, 1))
    Nmtp = (ImgIntensity < (medianMat - t).reshape(-1, 1))

    imgDesc = [
        {'fea': cp.dot(Pmtp.astype(cp.uint8), 1 << cp.arange(Pmtp.shape[1] - 1, -1, -1)).reshape(rSize, cSize)},
        {'fea': cp.dot(Nmtp.astype(cp.uint8), 1 << cp.arange(Nmtp.shape[1] - 1, -1, -1)).reshape(rSize, cSize)}
    ]

    options['binVec'] = [cp.arange(256), cp.arange(256)]  # CuPy array for bin vectors

    # Compute MTP histogram
    MTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = cp.sum(imgReg == bin_val)  # CuPy sum operation
            MTP_hist.append(hh)
    MTP_hist = cp.array(MTP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        MTP_hist = MTP_hist / cp.sum(MTP_hist)  # Normalize histogram using CuPy sum

    return MTP_hist, imgDesc