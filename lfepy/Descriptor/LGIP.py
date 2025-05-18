import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing LGIP pattern
compute_lgip_pattern = cp.RawKernel(r'''
extern "C" __global__
void compute_lgip_pattern(
    const double* image,
    int* pattern,
    const int rows,
    const int cols
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < (rows-4) * (cols-4); i += stride) {
        int r = i / (cols-4);
        int c = i % (cols-4);

        // Compute gradient patterns exactly as in NumPy version
        double v000 = (double)(-image[(r+1)*cols + (c+2)] + image[(r+1)*cols + (c+3)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+2)*cols + (c+3)] - 
                       image[(r+3)*cols + (c+2)] + image[(r+3)*cols + (c+3)] > 0);
        double v001 = (double)(-image[(r+1)*cols + (c+1)] + image[r*cols + (c+2)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+1)*cols + (c+3)] - 
                       image[(r+3)*cols + (c+3)] + image[(r+2)*cols + (c+4)] > 0);
        double v010 = (double)(-image[(r+2)*cols + (c+1)] + image[(r+1)*cols + (c+1)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+1)*cols + (c+2)] - 
                       image[(r+2)*cols + (c+3)] + image[(r+1)*cols + (c+3)] > 0);
        double v011 = (double)(-image[(r+3)*cols + (c+1)] + image[(r+2)*cols + c] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+1)*cols + (c+1)] - 
                       image[(r+1)*cols + (c+3)] + image[r*cols + (c+2)] > 0);
        double v100 = (double)(-image[(r+1)*cols + (c+2)] + image[(r+1)*cols + (c+1)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+2)*cols + (c+1)] - 
                       image[(r+3)*cols + (c+2)] + image[(r+3)*cols + (c+1)] > 0);
        double v101 = (double)(-image[(r+1)*cols + (c+1)] + image[(r+2)*cols + c] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+3)*cols + (c+1)] - 
                       image[(r+3)*cols + (c+3)] + image[(r+4)*cols + (c+2)] > 0);
        double v110 = (double)(-image[(r+2)*cols + (c+1)] + image[(r+3)*cols + (c+1)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+3)*cols + (c+2)] - 
                       image[(r+2)*cols + (c+3)] + image[(r+3)*cols + (c+3)] > 0);
        double v111 = (double)(-image[(r+3)*cols + (c+1)] + image[(r+4)*cols + (c+2)] - 
                       2*image[(r+2)*cols + (c+2)] + 2*image[(r+3)*cols + (c+3)] - 
                       image[(r+1)*cols + (c+3)] + image[(r+2)*cols + (c+4)] > 0);

        // Compute OTVx and OTVy exactly as in NumPy version
        double OTVx = v000 + v001 + v111 - v011 - v100 - v101;
        double OTVy = v001 + v010 + v011 - v101 - v110 - v111;

        // Clip OTV values to be within the pattern mask range and convert to integers
        int OTVx_clipped = min(max(int(OTVx + 4), 0), 6);
        int OTVy_clipped = min(max(int(OTVy + 4), 0), 6);

        // Define pattern mask values exactly as in NumPy version
        const int pattern_mask[7][7] = {
            {-1, -1, 30, 29, 28, -1, -1},
            {-1, 16, 15, 14, 13, 12, -1},
            {31, 17, 4, 3, 2, 11, 27},
            {32, 18, 5, 0, 1, 10, 26},
            {33, 19, 6, 7, 8, 9, 25},
            {-1, 20, 21, 22, 23, 24, -1},
            {-1, -1, 34, 35, 36, -1, -1}
        };

        // Get pattern value using the same indexing as NumPy version
        pattern[i] = pattern_mask[OTVx_clipped][OTVy_clipped];
    }
}
''', 'compute_lgip_pattern')


def LGIP(image, **kwargs):
    """
    Compute Local Gradient Increasing Pattern (LGIP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGIP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGIP_hist (cupy.ndarray): Histogram(s) of LGIP descriptors.
            imgDesc (cupy.ndarray): LGIP descriptors themselves.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGIP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        Z. Lubing, and W. Han,
        Local Gradient Increasing Pattern for Facial Expression Recognition,
        Image Processing (ICIP), 2012 19th IEEE International Conference on, IEEE,
        2012, pp. 2601-2604.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Convert image to float64 and move to GPU
    image = cp.asarray(image, dtype=cp.float64)
    r, c = image.shape

    # Allocate memory for output pattern
    pattern = cp.zeros((r - 4, c - 4), dtype=cp.int32)

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
    blocks_per_grid = ((c + block_size - 1) // block_size, (r + block_size - 1) // block_size)

    # Launch kernel
    compute_lgip_pattern(blocks_per_grid, threads_per_block, (image, pattern, r, c))
    imgDesc = pattern

    # Set bin vectors
    options['binVec'] = cp.arange(37)  # 0 to 36 inclusive

    # Compute LGIP histogram
    LGIP_hist = cp.zeros(len(options['binVec']))
    LGIP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LGIP_hist = LGIP_hist / cp.sum(LGIP_hist)

    return LGIP_hist, imgDesc