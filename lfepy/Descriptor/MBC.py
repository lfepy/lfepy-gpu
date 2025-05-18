import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Helper import monofilt, descriptor_LBP, lxp_phase
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_mbcMode


def MBC(image, **kwargs):
    """
    Compute Monogenic Binary Coding (MBC) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MBC extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            mbcMode (str): Mode for MBC computation. Options: 'A' (amplitude), 'O' (orientation), 'P' (phase). Default is 'A'.

    Returns:
        tuple: A tuple containing:
            MBC_hist (numpy.ndarray): Histogram of MBC descriptors.
            imgDesc (list): List of dictionaries containing MBC descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` or `mbcMode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MBC(image, mode='nh', mbcMode='A')

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M. Yang, L. Zhang, S.C.-K. Shiu, and D. Zhang,
        Monogenic binary coding: An efficient local feature extraction approach to face recognition,
        IEEE Transactions on Information Forensics and Security,
        7 (2012) 1738-1751.

        X.X. Xia, Z.L. Ying, and W.J. Chu,
        Facial Expression Recognition Based on Monogenic Binary Coding,
        Applied Mechanics and Materials, Trans Tech Publ,
        2014, pp. 437-440.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    options = validate_mbcMode(options)

    # Parameters for monogenic filters
    minWaveLength = 4
    sigmaOnf = 0.64
    mult = 1.7
    nscale = 3
    neigh = 8
    MAPPING = 0

    imgDesc = []
    options['binVec'] = []

    # Amplitude-based MBC
    if options['mbcMode'] == 'A':
        orientWrap = 0
        radius = 3
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = cp.array((A1[v] - cp.min(A1[v])) / (cp.max(A1[v]) - cp.min(A1[v])) * 255, dtype=cp.uint8).astype(cp.float64)
            LBPHIST, _ = descriptor_LBP(Tem_img, radius, neigh, MAPPING, 'i')
            matrix2 = cp.zeros(h1f1[v].shape)
            matrix3 = cp.zeros(h2f1[v].shape)
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius:-radius, radius:-radius]
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius:-radius, radius:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + LBPHIST.astype(cp.float64)
            N_LBPHIST = N_LBPHIST.astype(cp.uint16)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(cp.arange(1024))

    # Orientation-based MBC
    elif options['mbcMode'] == 'O':
        orientWrap = 0
        radius = 4
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = cp.array((theta1[v] - cp.min(theta1[v])) / (cp.max(theta1[v]) - cp.min(theta1[v])) * 360, dtype=cp.uint16)
            LBPHIST = lxp_phase(Tem_img, radius, neigh, 0, 'i')
            matrix2 = cp.zeros(h1f1[v].shape)
            matrix3 = cp.zeros(h2f1[v].shape)
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius + 1:-radius, radius + 1:-radius]
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius + 1:-radius, radius + 1:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + LBPHIST.astype(cp.float64)
            N_LBPHIST = N_LBPHIST.astype(cp.uint16)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(cp.arange(1024))

    # Phase-based MBC
    elif options['mbcMode'] == 'P':
        orientWrap = 1
        radius = 4
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = cp.array((psi1[v] - cp.min(psi1[v])) / (cp.max(psi1[v]) - cp.min(psi1[v])) * 360, dtype=cp.uint16)
            LBPHIST = lxp_phase(Tem_img, radius, neigh, 0, 'i')
            matrix2 = cp.zeros(h1f1[v].shape)
            matrix3 = cp.zeros(h2f1[v].shape)
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius + 1:-radius, radius + 1:-radius]
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius + 1:-radius, radius + 1:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + LBPHIST.astype(cp.float64)
            N_LBPHIST = N_LBPHIST.astype(cp.uint16)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(cp.arange(1024))

    # Compute MBC histogram
    MBC_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        MBC_hist.extend(hist)
    MBC_hist = cp.array(MBC_hist)

    if 'mode' in options and options['mode'] == 'nh':
        MBC_hist = MBC_hist / cp.sum(MBC_hist)

    return MBC_hist, imgDesc