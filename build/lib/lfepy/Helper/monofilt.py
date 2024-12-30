import cupy as cp


def monofilt(im, nscale, minWaveLength, mult, sigmaOnf, orientWrap=0, thetaPhase=1):
    """
    Apply a multiscale directional filter bank to a 2D grayscale image using Log-Gabor filters.

    Args:
        im (numpy.ndarray): 2D grayscale image.
        nscale (int): Number of scales in the filter bank.
        minWaveLength (float): Minimum wavelength of the filters.
        mult (float): Scaling factor between consecutive scales.
        sigmaOnf (float): Bandwidth of the Log-Gabor filter.
        orientWrap (int, optional): If 1, wrap orientations to the range [0, π]. Default is 0 (no wrapping).
        thetaPhase (int, optional): If 1, compute phase angles (theta and psi). Default is 1.

    Returns:
        tuple: A tuple containing:
            f (list of numpy.ndarray): Filter responses in the spatial domain.
            h1f (list of numpy.ndarray): x-direction filter responses in the spatial domain.
            h2f (list of numpy.ndarray): y-direction filter responses in the spatial domain.
            A (list of numpy.ndarray): Amplitude of the filter responses.
            theta (list of numpy.ndarray, optional): Phase angles of the filter responses, if `thetaPhase` is 1.
            psi (list of numpy.ndarray, optional): Orientation angles of the filter responses, if `thetaPhase` is 1.

    Raises:
        ValueError: If the input image is not 2D.

    Example:
        >>> import numpy as np
        >>> from scipy import ndimage
        >>> image = np.random.rand(100, 100)
        >>> nscale = 4
        >>> minWaveLength = 6
        >>> mult = 2.0
        >>> sigmaOnf = 0.55
        >>> f, h1f, h2f, A, theta, psi = monofilt(image, nscale, minWaveLength, mult, sigmaOnf)
        >>> print(len(f))
        4
        >>> print(f[0].shape)
        (100, 100)
    """
    if cp.ndim(im) == 2:
        rows, cols = im.shape
    else:
        raise ValueError("Input image must be 2D.")

        # Compute the 2D Fourier Transform of the image
    IM = cp.fft.fft2(im)

    # Generate frequency coordinates
    u1, u2 = cp.meshgrid(
        (cp.arange(cols) - (cols // 2 + 1)) / (cols - cols % 2),
        (cp.arange(rows) - (rows // 2 + 1)) / (rows - rows % 2)
    )

    # Shift the frequency coordinates
    u1 = cp.fft.ifftshift(u1)
    u2 = cp.fft.ifftshift(u2)

    # Compute the radius in the frequency domain
    radius = cp.sqrt(u1 ** 2 + u2 ** 2)
    radius[1, 1] = 1  # Avoid division by zero at the origin

    # Initialize filter responses
    H1 = 1j * u1 / radius
    H2 = 1j * u2 / radius

    # Initialize empty lists to store filter responses
    f = cp.empty((0, rows, cols), dtype=cp.float32)
    h1f = cp.empty((0, rows, cols), dtype=cp.float32)
    h2f = cp.empty((0, rows, cols), dtype=cp.float32)
    A = cp.empty((0, rows, cols), dtype=cp.float32)
    theta = cp.empty((0, rows, cols), dtype=cp.float32) if thetaPhase else None
    psi = cp.empty((0, rows, cols), dtype=cp.float32) if thetaPhase else None

    for s in range(1, nscale + 1):
        # Calculate wavelength and filter frequency
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength

        # Create Log-Gabor filter
        logGabor = cp.exp(-((cp.log(radius / fo)) ** 2) / (2 * cp.log(sigmaOnf) ** 2))
        logGabor[0, 0] = 0  # Avoid division by zero at the origin

        # Apply filter in frequency domain
        H1s = H1 * logGabor
        H2s = H2 * logGabor

        # Convert back to spatial domain
        f_spatial = cp.real(cp.fft.ifft2(IM * logGabor))
        h1f_spatial = cp.real(cp.fft.ifft2(IM * H1s))
        h2f_spatial = cp.real(cp.fft.ifft2(IM * H2s))

        # Compute amplitude
        A_s = cp.sqrt(f_spatial ** 2 + h1f_spatial ** 2 + h2f_spatial ** 2)

        # Concatenate the results into cupy arrays
        f = cp.concatenate((f, f_spatial[cp.newaxis, ...]), axis=0)
        h1f = cp.concatenate((h1f, h1f_spatial[cp.newaxis, ...]), axis=0)
        h2f = cp.concatenate((h2f, h2f_spatial[cp.newaxis, ...]), axis=0)
        A = cp.concatenate((A, A_s[cp.newaxis, ...]), axis=0)

        if thetaPhase:
            # Compute phase angles
            theta_s = cp.arctan2(h2f_spatial, h1f_spatial)
            psi_s = cp.arctan2(f_spatial, cp.sqrt(h1f_spatial ** 2 + h2f_spatial ** 2))

            if orientWrap:
                # Wrap orientations to [0, π] range
                theta_s[theta_s < 0] += cp.pi
                psi_s[theta_s < 0] = cp.pi - psi_s[theta_s < 0]
                psi_s[psi_s > cp.pi] -= 2 * cp.pi

            # Concatenate phase results
            theta = cp.concatenate((theta, theta_s[cp.newaxis, ...]), axis=0)
            psi = cp.concatenate((psi, psi_s[cp.newaxis, ...]), axis=0)

    if thetaPhase:
        return f, h1f, h2f, A, theta, psi
    else:
        return f, h1f, h2f, A