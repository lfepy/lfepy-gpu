import cupy as cp
from cupyx.scipy.signal import convolve2d


def descriptor_LPQ(image, winSize=3, decorr=1, freqestim=1, mode='im'):
    """
    Compute the Local Phase Quantization (LPQ) descriptor for a given grayscale image.

    This function computes the LPQ descriptor, which captures local texture information
    by analyzing the phase of the image's frequency components. The descriptor can be
    computed using different frequency estimation methods and can be returned as an image
    or a histogram based on the specified mode.

    Args:
        image (cupy.ndarray): Grayscale input image. Must be a 2D array.
        winSize (int, optional): Size of the window used for LPQ calculation. Must be an odd number ≥ 3. Default is 3.
        decorr (int, optional): Flag to apply decorrelation. 0 for no decorrelation, 1 for decorrelation. Default is 1.
        freqestim (int, optional): Frequency estimation method.
                                    1 for STFT uniform window,
                                    2 for STFT Gaussian window,
                                    3 for Gaussian derivative quadrature filter pair. Default is 1.
        mode (str, optional): Specifies the output format.
                              'im' for image-like output,
                              'nh' for normalized histogram,
                              'h' for histogram. Default is 'im'.

    Returns:
        tuple: A tuple containing:
            LPQdesc (cupy.ndarray): The LPQ descriptor of the image. Depending on `mode`, it could be an image or a histogram.
            freqRespAll (cupy.ndarray): The frequency responses for all filter pairs.
    """
    # Initialize parameters
    rho = 0.90
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)
    convmode = 'valid'

    # Check inputs
    if image.ndim != 2:
        raise ValueError("Only gray scale image can be used as input")
    if winSize < 3 or winSize % 2 == 0:
        raise ValueError("Window size winSize must be an odd number and greater than or equal to 3")
    if decorr not in [0, 1]:
        raise ValueError("decorr parameter must be set to 0 for no decorrelation or 1 for decorrelation")
    if freqestim not in [1, 2, 3]:
        raise ValueError("freqestim parameter must be 1, 2, or 3")
    if mode not in ['nh', 'h', 'im']:
        raise ValueError("mode must be 'nh', 'h', or 'im'")

    # Initialize
    r = (winSize - 1) // 2
    x = cp.arange(-r, r + 1)
    u = cp.arange(1, r + 1)

    # Form 1-D filters
    if freqestim == 1:  # STFT uniform window
        w0 = cp.ones_like(x)
        w1 = cp.exp(-2j * cp.pi * x * STFTalpha)
        w2 = cp.conj(w1)
    elif freqestim == 2:  # STFT Gaussian window
        w0 = cp.exp(-0.5 * (x / sigmaS) ** 2) / (cp.sqrt(2 * cp.pi) * sigmaS)
        w1 = cp.exp(-2j * cp.pi * x * STFTalpha)
        w2 = cp.conj(w1)
        gs = cp.exp(-0.5 * (x / sigmaS) ** 2) / (cp.sqrt(2 * cp.pi) * sigmaS)
        w0 *= gs
        w1 *= gs
        w2 *= gs
        w1 -= cp.mean(w1)
        w2 -= cp.mean(w2)
    elif freqestim == 3:  # Gaussian derivative quadrature filter pair
        G0 = cp.exp(-x ** 2 * (cp.sqrt(2) * sigmaA) ** 2)
        G1 = cp.concatenate((cp.zeros_like(u), u * cp.exp(-u ** 2 * sigmaA ** 2), [0]))
        G0 = G0 / cp.max(cp.abs(G0))
        G1 = G1 / cp.max(cp.abs(G1))
        w0 = cp.real(cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(G0))))
        w1 = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(G1)))
        w2 = cp.conj(w1)
        w0 = w0 / cp.max(cp.abs([cp.real(cp.max(w0)), cp.imag(cp.max(w0))]))
        w1 = w1 / cp.max(cp.abs([cp.real(cp.max(w1)), cp.imag(cp.max(w1))]))
        w2 = w2 / cp.max(cp.abs([cp.real(cp.max(w2)), cp.imag(cp.max(w2))]))

    # Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
    filterResp = convolve2d(convolve2d(image, w0[:, cp.newaxis], mode=convmode), w1[cp.newaxis, :], mode=convmode)
    freqResp = cp.zeros((filterResp.shape[0], filterResp.shape[1], 8), dtype=cp.complex128)
    freqResp[:, :, 0] = cp.real(filterResp)
    freqResp[:, :, 1] = cp.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, cp.newaxis], mode=convmode), w0[cp.newaxis, :], mode=convmode)
    freqResp[:, :, 2] = cp.real(filterResp)
    freqResp[:, :, 3] = cp.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, cp.newaxis], mode=convmode), w1[cp.newaxis, :], mode=convmode)
    freqResp[:, :, 4] = cp.real(filterResp)
    freqResp[:, :, 5] = cp.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, cp.newaxis], mode=convmode), w2[cp.newaxis, :], mode=convmode)
    freqResp[:, :, 6] = cp.real(filterResp)
    freqResp[:, :, 7] = cp.imag(filterResp)
    freqRespAll = filterResp
    freqRow, freqCol, freqNum = freqResp.shape

    # If decorrelation is used, compute covariance matrix and corresponding whitening transform
    if decorr == 1:
        xp, yp = cp.meshgrid(cp.arange(1, winSize + 1), cp.arange(1, winSize + 1))
        pp = cp.column_stack((xp.flatten(), yp.flatten()))
        dd = cp.linalg.norm(pp[:, cp.newaxis] - pp[cp.newaxis, :], axis=-1)
        C = rho ** dd
        q1 = cp.outer(w0, w1)
        q2 = cp.outer(w1, w0)
        q3 = cp.outer(w1, w1)
        q4 = cp.outer(w1, w2)
        u1, u2 = cp.real(q1), cp.imag(q1)
        u3, u4 = cp.real(q2), cp.imag(q2)
        u5, u6 = cp.real(q3), cp.imag(q3)
        u7, u8 = cp.real(q4), cp.imag(q4)
        M = cp.array([u1.flatten(), u2.flatten(), u3.flatten(), u4.flatten(), u5.flatten(), u6.flatten(),
                      u7.flatten(), u8.flatten()])
        D = M @ C @ M.T
        A = cp.diag([1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1])
        U, S, Vt = cp.linalg.svd(A @ D @ A)
        idx = cp.argmax(cp.abs(Vt), axis=0)
        V = Vt * cp.diag(1 - 2 * (Vt[idx, range(Vt.shape[1])] < -cp.finfo(cp.float64).eps))
        freqResp = freqResp.reshape(freqRow * freqCol, freqNum)
        freqResp = (V.T @ freqResp.T).T
        freqResp = freqResp.reshape(freqRow, freqCol, freqNum)

    LPQdesc = cp.zeros_like(freqResp[:, :, 0])
    LPQdesc += cp.sum((freqResp > 0) * (2 ** cp.arange(freqNum)), axis=2)

    # Histogram if needed
    if mode == 'im':
        LPQdesc = cp.abs(LPQdesc).astype(cp.uint8)

    if mode == 'nh' or mode == 'h':
        LPQdesc = cp.histogram(LPQdesc.flatten(), bins=256, range=(0, 255))[0]

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / cp.sum(LPQdesc)

    return LPQdesc, freqRespAll