import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing LGP pattern
lgp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_lgp_pattern(
    const double* image,
    double* pattern,
    int rows,
    int cols
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    double val = 0.0;

    // Path 1: Diagonal comparisons
    double a1 = image[y * (cols + 2) + x];
    double a3 = image[(y + 2) * (cols + 2) + (x + 2)];
    double a2 = image[y * (cols + 2) + (x + 2)];
    double a4 = image[(y + 2) * (cols + 2) + x];

    if (a1 - a3 > 0) val += 2;
    if (a2 - a4 > 0) val += 1;

    // Path 2: Horizontal and vertical comparisons
    double b1 = image[y * (cols + 2) + (x + 1)];
    double b3 = image[(y + 2) * (cols + 2) + (x + 1)];
    double b2 = image[(y + 1) * (cols + 2) + (x + 2)];
    double b4 = image[(y + 1) * (cols + 2) + x];

    if (b1 - b3 > 0) val += 2;
    if (b2 - b4 > 0) val += 1;
    val += 4;  // Add 4 to match the NumPy version's path2

    pattern[idx] = val;
}
''', 'compute_lgp_pattern')


def LGP(image, **kwargs):
    """
    Compute Local Gradient Pattern (LGP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGP_hist (numpy.ndarray): Histogram(s) of LGP descriptors.
            imgDesc (numpy.ndarray): LGP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Local Gradient Pattern-A Novel Feature Representation for Facial Expression Recognition,
        Journal of AI and Data Mining 2,
        (2014), pp. 33-38.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Ensure image is float64
    image = image.astype(cp.float64)

    # Get dimensions for output
    rSize, cSize = image.shape[0] - 2, image.shape[1] - 2

    # Allocate pattern output
    pattern = cp.zeros((rSize, cSize), dtype=cp.float64)

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

    # Launch kernel
    lgp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            pattern.ravel(),
            rSize,
            cSize
        )
    )

    imgDesc = pattern

    # Set bin vectors
    options['binVec'] = cp.arange(4, 11)

    # Compute LGP histogram
    LGP_hist = cp.zeros(len(options['binVec']))
    LGP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LGP_hist = LGP_hist / cp.sum(LGP_hist)

    return LGP_hist, imgDesc