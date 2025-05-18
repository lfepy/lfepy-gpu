import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing IWBC pattern
iwbc_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_iwbc_pattern(
    const double* image,
    double* DEx,
    double* DEy,
    const int* link,
    int rows,
    int cols,
    int scale,
    int numNeigh,
    double ANGLE,
    double ANGLEDiff
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    double x_c = image[(y + scale) * (cols + 2 * scale) + (x + scale)];
    double angle = ANGLE;

    for (int n = 0; n < numNeigh; n++) {
        int y_pos = y + link[n * 2] - 1;
        int x_pos = x + link[n * 2 + 1] - 1;
        double x_i = image[y_pos * (cols + 2 * scale) + x_pos];
        DEx[idx] += (x_i - x_c) * cos(angle);
        DEy[idx] += (x_i - x_c) * sin(angle);
        angle -= ANGLEDiff;
    }
}
''', 'compute_iwbc_pattern')

# Define the raw kernel for computing LBMP and LXOP
pattern_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_pattern(
    const double* input,
    double* output,
    const int* link,
    int rows,
    int cols,
    int scale,
    int numNeigh,
    int is_magnitude
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    double x_c = input[(y + scale) * (cols + 2 * scale) + (x + scale)];
    double pattern = 0.0;

    for (int i = 0; i < numNeigh; i++) {
        int y_pos = y + link[i * 2] - 1;
        int x_pos = x + link[i * 2 + 1] - 1;
        double x_i = input[y_pos * (cols + 2 * scale) + x_pos];

        if (is_magnitude) {
            double diff = x_i - x_c;
            if (diff >= 0.0) {
                pattern += exp2(static_cast<double>(numNeigh - i - 1));
            }
        } else {
            if (x_i != x_c) {
                pattern += exp2(static_cast<double>(numNeigh - i - 1));
            }
        }
    }
    output[idx] = pattern;
}
''', 'compute_pattern')


def IWBC(image, **kwargs):
    """
    Compute Improved Weber Contrast (IWBC) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing IWBC extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            scale (int): Scale factor for IWBC computation. Default is 1.

    Returns:
        tuple: A tuple containing:
            IWBC_hist (cupy.ndarray): Histogram(s) of IWBC descriptors.
            imgDesc (list): List of dictionaries containing IWBC descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = IWBC(image, mode='nh', scale=1)

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        B.-Q. Yang, T. Zhang, C.-C. Gu, K.-J. Wu, and X.-P. Guan,
        A novel face recognition method based on IWLD and IWBC,
        Multimedia Tools and Applications,
        vol. 75, pp. 6979, 2016.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define scale-specific cell configurations
    scaleCell = {
        1: cp.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1]]),
        2: cp.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5],
                     [5, 5], [5, 4], [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [2, 1]]),
        3: cp.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7],
                     [7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [7, 2], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1]])}

    # Extract scale factor or use default
    scale = options.get('scale', 1)

    # Define constants and angles for IWBC computation
    BELTA = 5
    ALPHA = 3
    EPSILON = 0.0000001
    ANGLE = 5 * cp.pi / 4
    ANGLEDiff = 2 * cp.pi / (scale * 8)

    # Extract central region of the image
    numNeigh = scale * 8
    x_c = image[scale:-scale, scale:-scale]
    rSize, cSize = x_c.shape
    DEx = cp.zeros((rSize, cSize))
    DEy = cp.zeros((rSize, cSize))
    link = scaleCell[scale]

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

    iwbc_kernel(blocks_per_grid, threads_per_block, (image, DEx, DEy, link, rSize, cSize, scale, numNeigh, ANGLE, ANGLEDiff))

    # Compute EPSx and EPSy
    EPSx = cp.arctan((ALPHA * DEx) / (x_c + BELTA))
    EPSy = cp.arctan((ALPHA * DEy) / (x_c + BELTA))
    signEPSx = cp.sign(EPSx)
    signEPSy = cp.sign(EPSy)

    # Convert EPSx and EPSy to degrees
    EPSxDeg = EPSx * 180 / cp.pi
    EPSyDeg = EPSy * 180 / cp.pi
    # Compute NWM (Normalized Weber Magnitude)
    NWM = cp.sqrt(EPSxDeg ** 2 + EPSyDeg ** 2)
    EPSx[EPSx == 0] = EPSILON
    # Compute NWO (Normalized Weber Orientation)
    NWO = cp.arctan(EPSy / EPSx) * 180 / cp.pi
    NWO[EPSx < 0] += 180
    NWO[(EPSx > 0) & (EPSy < 0)] += 360

    # Define binary maps B_x and B_y
    B_x = cp.ones_like(signEPSx)
    B_x[signEPSx == 1] = 0
    B_y = cp.ones_like(signEPSy)
    B_y[signEPSy == 1] = 0

    # Initialize variables for scale 2 computation
    scale2 = 1
    numNeigh = scale2 * 8
    link = scaleCell[scale2]

    # Compute LBMP (Local Binary Magnitude Pattern)
    x_c = NWM[scale2:-scale2, scale2:-scale2]
    rSize, cSize = x_c.shape
    LBMP = cp.zeros((rSize, cSize))

    # Launch kernel for LBMP computation
    pattern_kernel(blocks_per_grid, threads_per_block, (NWM, LBMP, link, rSize, cSize, scale2, numNeigh, 1))

    # Compute IWBC_M (Magnitude Component of Improved Weber Contrast)
    IWBC_M = LBMP + B_y[scale2:-scale2, scale2:-scale2] * 2 ** numNeigh
    IWBC_M += B_x[scale2:-scale2, scale2:-scale2] * 2 ** (numNeigh + 1)

    NWO[NWO == 360] = 0
    NWO[(NWO >= 0) & (NWO < 90)] = 0
    NWO[(NWO >= 90) & (NWO < 180)] = 1
    NWO[(NWO >= 180) & (NWO < 270)] = 2
    NWO[(NWO >= 270) & (NWO < 360)] = 3

    # Convert NWO to discrete orientation bins
    x_c = NWO[scale2:-scale2, scale2:-scale2]
    LXOP = cp.zeros((rSize, cSize))

    # Launch kernel for LXOP computation
    pattern_kernel(blocks_per_grid, threads_per_block, (NWO, LXOP, link, rSize, cSize, scale2, numNeigh, 0))

    IWBC_O = LXOP + B_y[scale2:-scale2, scale2:-scale2] * 2 ** numNeigh
    IWBC_O += B_x[scale2:-scale2, scale2:-scale2] * 2 ** (numNeigh + 1)

    imgDesc = [{'fea': IWBC_M}, {'fea': IWBC_O}]

    # Set bin vectors
    binVec = [cp.arange(0, 2 ** (numNeigh + 2)), cp.arange(0, 2 ** (numNeigh + 2))]
    options['binVec'] = binVec

    # Compute IWBC histogram
    IWBC_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        IWBC_hist.extend(hist)
    IWBC_hist = cp.array(IWBC_hist)

    if 'mode' in options and options['mode'] == 'nh':
        IWBC_hist = IWBC_hist / cp.sum(IWBC_hist)

    return IWBC_hist, imgDesc