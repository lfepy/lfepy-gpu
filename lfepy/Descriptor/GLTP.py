import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Descriptor.LTeP import LTeP
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_DGLP


def GLTP(image, **kwargs):
    """
    Compute Gradient-based Local Ternary Pattern (GLTP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing GLTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            t (int): Threshold value for ternary pattern computation. Default is 10.
            DGLP (int): Flag to include Directional Gradient-based Local Pattern.
            If set to 1, includes DGLP. Default is 0.

    Returns:
        tuple: A tuple containing:
            GLTP_hist (cupy.ndarray): Histogram(s) of GLTP descriptors.
            imgDesc (list): List of dictionaries containing GLTP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` or `DGLP` in `kwargs` are not valid options.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = GLTP(image, mode='nh', t=10, DGLP=1)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M. Valstar, and M. Pantic,
        "Fully automatic facial action unit detection and temporal analysis",
        in *Computer Vision and Pattern Recognition Workshop, IEEE*,
        2006.

        F. Ahmed, and E. Hossain,
        "Automated facial expression recognition using gradient-based ternary texture patterns",
        in *Chinese Journal of Engineering*,
        vol. 2013, 2013.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    options = validate_DGLP(options)

    EPSILON = 1e-7

    # Define Sobel masks for gradient computation
    maskA = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    maskB = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Compute gradients using 2D convolution
    Gx = convolve2d(image, maskA, mode='same')
    Gy = convolve2d(image, maskB, mode='same')

    # Compute gradient magnitude
    img_gradient = cp.abs(Gx) + cp.abs(Gy)

    # Compute Local Ternary Pattern (LTeP) on gradient image
    _, imgDesc = LTeP(img_gradient, t=options.get('t', 10))
    options['binVec'] = [cp.arange(256) for _ in range(2)]

    # If DGLP flag is set, include directional gradient pattern
    if options['DGLP'] == 1:
        r, c = Gx.shape
        img_angle = cp.arctan2(Gy, Gx + EPSILON)
        img_angle = cp.degrees(img_angle)
        img_angle = cp.where(Gx < 0, img_angle + 180, img_angle)
        img_angle = cp.where((Gx >= 0) & (Gy < 0), img_angle + 360, img_angle)
        img_angle = img_angle[1:r - 1, 1:c - 1]
        img_angle = cp.floor(img_angle / 22.5).astype(int)

        imgDesc.append({'fea': img_angle})
        options['binVec'].append(cp.arange(16))

    # Compute GLTP histogram
    GLTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = cp.sum(imgReg == bin_val)
            GLTP_hist.append(hh)
    GLTP_hist = cp.array(GLTP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        GLTP_hist = GLTP_hist / cp.sum(GLTP_hist)

    return GLTP_hist, imgDesc