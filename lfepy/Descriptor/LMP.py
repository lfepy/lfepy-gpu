import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing LMP
lmp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_lmp(
    const double* image,
    double* imgDesc,
    int rows,
    int cols
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    // Define link list for LMP computation
    int link[8][2][2] = {
        {{3, 4}, {3, 5}},
        {{2, 4}, {1, 5}},
        {{2, 3}, {1, 3}},
        {{2, 2}, {1, 1}},
        {{3, 2}, {3, 1}},
        {{4, 2}, {5, 1}},
        {{4, 3}, {5, 3}},
        {{4, 4}, {5, 5}}
    };

    // Central pixel
    double x_c = image[(y + 2) * (cols + 4) + (x + 2)];
    double lmp_value = 0.0;

    // Pre-compute powers of 2
    double powers[8] = {128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0};

    for (int n = 0; n < 8; n++) {
        // First neighbor
        int y1 = y + link[n][0][0] - 1;
        int x1 = x + link[n][0][1] - 1;
        double x_i1 = image[y1 * (cols + 4) + x1];

        // Second neighbor
        int y2 = y + link[n][1][0] - 1;
        int x2 = x + link[n][1][1] - 1;
        double x_i2 = image[y2 * (cols + 4) + x2];

        // Compute LMP condition
        if ((x_i1 - x_c) >= 0 && (x_i2 - x_i1) >= 0) {
            lmp_value += powers[n];
        }
    }

    imgDesc[y * cols + x] = lmp_value;
}
''', 'compute_lmp')


def LMP(image, **kwargs):
    """
    Compute Local Monotonic Pattern (LMP) descriptors and histograms from an input image using CUDA.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LMP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LMP_hist (cupy.ndarray): Histogram(s) of LMP descriptors.
            imgDesc (cupy.ndarray): LMP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.
    """
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Extract central region dimensions
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape

    # Allocate output array
    imgDesc = cp.zeros((rSize, cSize), dtype=cp.float64)

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

    lmp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            imgDesc.ravel(),
            rSize,
            cSize
        )
    )

    # Set bin vectors
    options['binVec'] = cp.arange(256)

    # Compute LMP histogram
    LMP_hist = cp.zeros(len(options['binVec']), dtype=cp.float64)
    LMP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LMP_hist = LMP_hist / cp.sum(LMP_hist)

    return LMP_hist, imgDesc