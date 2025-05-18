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

    # Get center matrix
    centerMat = image[1:-1, 1:-1].flatten()

    # Create arrays for patch extraction
    y_indices = cp.arange(rSize)[:, None, None] + link[None, :, 0] - 1
    x_indices = cp.arange(cSize)[None, :, None] + link[None, None, :, 1] - 1

    # Extract all patches at once using advanced indexing
    patches = image[y_indices, x_indices]
    ImgIntensity = patches.reshape(rSize * cSize, 8)

    # Compute patterns using CuPy operations
    Pltp = (ImgIntensity > (centerMat[:, None] + t)).astype(cp.float64)
    Nltp = (ImgIntensity < (centerMat[:, None] - t)).astype(cp.float64)

    # Compute binary patterns using vectorized operations
    powers = 2 ** cp.arange(8)  # Pre-compute powers of 2
    pos_pattern = cp.dot(Pltp, powers).reshape(rSize, cSize)
    neg_pattern = cp.dot(Nltp, powers).reshape(rSize, cSize)

    imgDesc = [
        {'fea': pos_pattern},
        {'fea': neg_pattern}
    ]

    # Set bin vectors
    options['binVec'] = [cp.arange(256), cp.arange(256)]

    # Compute LTeP histogram using vectorized operations
    LTeP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        binVec = options['binVec'][s]
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        LTeP_hist.extend(hist)
    LTeP_hist = cp.array(LTeP_hist)

    if 'mode' in options and options['mode'] == 'nh':
        LTeP_hist = LTeP_hist / cp.sum(LTeP_hist)

    return LTeP_hist, imgDesc