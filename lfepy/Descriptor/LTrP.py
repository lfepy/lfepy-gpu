import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing LTrP pattern
ltrp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_ltrp_pattern(
    const double* image, double* pattern,
    int rows, int cols, int img_cols
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int linkList[8][4] = {
        {4, 4, 5, 5},
        {4, 3, 5, 3},
        {4, 2, 5, 1},
        {3, 2, 3, 1},
        {2, 2, 1, 1},
        {2, 3, 1, 3},
        {2, 4, 1, 5},
        {3, 4, 3, 5}
    };

    int idx = y * cols + x;
    double center_val = image[(y + 2) * img_cols + (x + 2)];
    double pattern_val = 0.0;

    for (int n = 0; n < 8; ++n) {
        int y1 = y + linkList[n][0] - 1;
        int x1 = x + linkList[n][1] - 1;
        int y2 = y + linkList[n][2] - 1;
        int x2 = x + linkList[n][3] - 1;

        double val1 = image[y1 * img_cols + x1];
        double val2 = image[y2 * img_cols + x2];

        bool diff1 = (val1 - center_val) >= 0;
        bool diff2 = (val2 - center_val) >= 0;

        if (diff1 != diff2) {
            pattern_val += (1 << (7 - n));
        }
    }

    pattern[idx] = pattern_val;
}
''', 'compute_ltrp_pattern')


def LTrP(image, **kwargs):
    """
    Compute Local Transitional Pattern (LTrP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LTrP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LTrP_hist (cupy.ndarray): Histogram(s) of LTrP descriptors.
            imgDesc (cupy.ndarray): LTrP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LTrP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Jabid, and O. Chae,
        Local Transitional Pattern: A Robust Facial Image Descriptor for Automatic Facial Expression Recognition,
        Proc. International Conference on Computer Convergence Technology, Seoul, Korea,
        2011, pp. 333-44.

        T. Jabid, and O. Chae,
        Facial Expression Recognition Based on Local Transitional Pattern,
        International Information Institute (Tokyo), Information,
        15 (2012) 2007.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Convert image to CuPy array and ensure float64
    image = cp.asarray(image, dtype=cp.float64)

    # Initialize variables
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape
    pattern = cp.zeros((rSize, cSize), dtype=cp.float64)

    # Launch kernel
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

    ltrp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            pattern.ravel(),
            rSize,
            cSize,
            image.shape[1]
        )
    )

    imgDesc = pattern.astype(cp.uint8)

    # Set bin vectors
    options['binVec'] = cp.arange(256)

    # Compute LTrP histogram
    LTrP_hist = cp.zeros(len(options['binVec']))
    LTrP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LTrP_hist = LTrP_hist / cp.sum(LTrP_hist)

    return LTrP_hist, imgDesc