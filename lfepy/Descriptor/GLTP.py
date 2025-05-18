import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Descriptor.LTeP import LTeP
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_DGLP

# Define the raw kernel for computing DGLP pattern
dglp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_dglp_pattern(
    const float* Gx,
    const float* Gy,
    float* pattern,
    int rows,
    int cols,
    float epsilon
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    float gx = Gx[(y + 1) * (cols + 2) + (x + 1)];
    float gy = Gy[(y + 1) * (cols + 2) + (x + 1)];

    float angle = atan2(gy, gx + epsilon);
    angle = degrees(angle);

    if (gx < 0) {
        angle += 180;
    } else if (gy < 0) {
        angle += 360;
    }

    pattern[idx] = floor(angle / 22.5f);
}
''', 'compute_dglp_pattern')


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

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
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
    maskA = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
    maskB = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)

    # Compute gradients using 2D convolution
    Gx = convolve2d(image, maskA, mode='same')
    Gy = convolve2d(image, maskB, mode='same')

    # Compute gradient magnitude
    img_gradient = cp.abs(Gx) + cp.abs(Gy)

    # Compute Local Ternary Pattern (LTeP) on gradient image
    # Note: Assuming LTeP is already optimized with its own kernel
    _, imgDesc = LTeP(img_gradient, t=options.get('t', 10))
    options['binVec'] = [cp.arange(256) for _ in range(2)]

    # If DGLP flag is set, include directional gradient pattern
    if options['DGLP'] == 1:
        r, c = Gx.shape
        img_angle = cp.zeros((r - 2, c - 2), dtype=cp.float32)

        # Launch DGLP kernel
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
        blocks_per_grid = ((cSize + block_size - 1) // block_size, (rSize + block_size - 1) // block_size)

        dglp_kernel(
            blocks_per_grid,
            threads_per_block,
            (Gx.ravel(), Gy.ravel(), img_angle.ravel(), r - 2, c - 2, EPSILON)
        )

        img_angle = img_angle.astype(cp.int32)
        imgDesc.append({'fea': img_angle})
        options['binVec'].append(cp.arange(16))

    # Compute GLTP histogram
    GLTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        GLTP_hist.extend(hist)
    GLTP_hist = cp.array(GLTP_hist)

    if 'mode' in options and options['mode'] == 'nh':
        GLTP_hist = GLTP_hist / cp.sum(GLTP_hist)

    return GLTP_hist, imgDesc