import cupy as cp

# Raw kernel for phase angle calculations
_phase_kernel = cp.RawKernel(r'''
#define M_PI 3.14159265358979323846f

extern "C" __global__
void compute_phase_angles(
    const float* f_spatial, const float* h1f_spatial, const float* h2f_spatial,
    float* theta, float* psi, int orientWrap, int nscale, int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    for (int s = 0; s < nscale; s++) {
        int scale_offset = s * size;
        float h1 = h1f_spatial[scale_offset + idx];
        float h2 = h2f_spatial[scale_offset + idx];
        float f = f_spatial[scale_offset + idx];

        // Compute theta (orientation)
        theta[scale_offset + idx] = atan2f(h2, h1);

        // Compute psi (phase)
        float h_mag = sqrtf(h1 * h1 + h2 * h2);
        psi[scale_offset + idx] = atan2f(f, h_mag);

        if (orientWrap) {
            if (theta[scale_offset + idx] < 0) {
                theta[scale_offset + idx] += M_PI;
                psi[scale_offset + idx] = M_PI - psi[scale_offset + idx];
            }
            if (psi[scale_offset + idx] > M_PI) {
                psi[scale_offset + idx] -= 2 * M_PI;
            }
        }
    }
}
''', 'compute_phase_angles')


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

    # Generate frequency coordinates using array operations
    u1 = cp.fft.ifftshift((cp.arange(cols) - (cols // 2 + 1)) / (cols - cols % 2))
    u2 = cp.fft.ifftshift((cp.arange(rows) - (rows // 2 + 1)) / (rows - rows % 2))
    u1, u2 = cp.meshgrid(u1, u2)

    # Compute radius using array operations
    radius = cp.sqrt(u1 ** 2 + u2 ** 2)
    radius[1, 1] = 1  # Avoid division by zero

    # Initialize filter responses
    H1 = 1j * u1 / radius
    H2 = 1j * u2 / radius

    # Pre-allocate arrays for all scales
    f = cp.zeros((nscale, rows, cols), dtype=cp.float32)
    h1f = cp.zeros((nscale, rows, cols), dtype=cp.float32)
    h2f = cp.zeros((nscale, rows, cols), dtype=cp.float32)
    A = cp.zeros((nscale, rows, cols), dtype=cp.float32)

    if thetaPhase:
        theta = cp.zeros((nscale, rows, cols), dtype=cp.float32)
        psi = cp.zeros((nscale, rows, cols), dtype=cp.float32)

    # Compute wavelengths and frequencies for all scales at once
    wavelengths = minWaveLength * (mult ** cp.arange(nscale))
    fo = 1.0 / wavelengths

    # Reshape for broadcasting
    radius_3d = radius[cp.newaxis, :, :]
    fo_3d = fo[:, cp.newaxis, cp.newaxis]
    H1_3d = H1[cp.newaxis, :, :]
    H2_3d = H2[cp.newaxis, :, :]
    IM_3d = IM[cp.newaxis, :, :]

    # Create Log-Gabor filters for all scales at once
    logGabor = cp.exp(-((cp.log(radius_3d / fo_3d)) ** 2) / (2 * cp.log(sigmaOnf) ** 2))
    logGabor[:, 0, 0] = 0

    # Apply filters in frequency domain for all scales
    H1s = H1_3d * logGabor
    H2s = H2_3d * logGabor

    # Convert back to spatial domain for all scales
    f = cp.real(cp.fft.ifft2(IM_3d * logGabor, axes=(1, 2)))
    h1f = cp.real(cp.fft.ifft2(IM_3d * H1s, axes=(1, 2)))
    h2f = cp.real(cp.fft.ifft2(IM_3d * H2s, axes=(1, 2)))

    # Compute amplitude for all scales
    A = cp.sqrt(f ** 2 + h1f ** 2 + h2f ** 2)

    if thetaPhase:
        # Compute phase angles using raw kernel for all scales at once
        size = rows * cols
        # Get the current CUDA device
        device = cp.cuda.Device()
        # Get the maximum threads per block for the device
        max_threads_per_block = device.attributes['MaxThreadsPerBlock']
        # Calculate the optimal block size (using a square block for simplicity)
        block_size = int(cp.sqrt(max_threads_per_block))
        # Ensure block_size is a multiple of 16 for better memory alignment
        block_size = (block_size // 16) * 16

        # Calculate grid dimensions
        threads_per_block = (block_size, block_size)
        blocks_per_grid = ((cols + block_size - 1) // block_size, (rows + block_size - 1) // block_size)

        _phase_kernel(
            blocks_per_grid, threads_per_block,
            (f.ravel(), h1f.ravel(), h2f.ravel(),
             theta.ravel(), psi.ravel(),
             orientWrap, nscale, size)
        )

    if thetaPhase:
        return f, h1f, h2f, A, theta, psi
    else:
        return f, h1f, h2f, A