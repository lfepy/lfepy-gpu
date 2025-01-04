import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_t_LTeP


def LTeP(image, **kwargs):
    """
    Compute Local Ternary Pattern (LTeP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LTeP extraction.
            t (int): Threshold value for ternary pattern computation. Default is 2.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LTeP_hist (cupy.ndarray): Histogram(s) of LTeP descriptors.
            imgDesc (list of dicts): List of dictionaries containing LTeP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LTeP(image, mode='nh', t=2)

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust Facial Expression Recognition Based on Median Ternary Pattern (MTP),
        Electrical Information and Communication Technology (EICT), IEEE,
        2014, pp. 1-5.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    t = validate_t_LTeP(options)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for LTeP computation
    link = cp.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])
    ImgIntensity = cp.zeros((rSize * cSize, 8))

    # Compute LTeP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        ImgIntensity[:, n] = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1].flatten()

    centerMat = image[1:-1, 1:-1].flatten()

    Pltp = ImgIntensity > (centerMat[:, None] + t)
    Nltp = ImgIntensity < (centerMat[:, None] - t)

    imgDesc = [{'fea': Pltp.dot(2 ** cp.arange(Pltp.shape[-1])).reshape(rSize, cSize)},
               {'fea': Nltp.dot(2 ** cp.arange(Nltp.shape[-1])).reshape(rSize, cSize)}]

    # Set bin vectors
    options['binVec'] = [cp.arange(256), cp.arange(256)]

    # Compute LTeP histogram
    LTeP_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        LTeP_hist.extend(hist)
    LTeP_hist = cp.array(LTeP_hist)

    if 'mode' in options and options['mode'] == 'nh':
        LTeP_hist = LTeP_hist / cp.sum(LTeP_hist)

    return LTeP_hist, imgDesc