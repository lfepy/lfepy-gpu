import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing neighbor intensities
mbp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_neighbor_intensities(
    const double* image, double* intensities,
    int rows, int cols, int img_cols
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int linkList[8][2] = {
        {2, 1}, {1, 1}, {1, 2}, {1, 3},
        {2, 3}, {3, 3}, {3, 2}, {3, 1}
    };

    int idx = y * cols + x;

    // Get all neighbors
    for (int n = 0; n < 8; ++n) {
        int y1 = y + linkList[n][0] - 1;
        int x1 = x + linkList[n][1] - 1;
        intensities[idx * 8 + n] = image[y1 * img_cols + x1];
    }
}
''', 'compute_neighbor_intensities')


def MBP(image, **kwargs):
    """
    Compute Median Binary Pattern (MBP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MBP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            MBP_hist (cupy.ndarray): Histogram(s) of MBP descriptors.
            imgDesc (cupy.ndarray): MBP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MBP(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust facial expression recognition based on median ternary pattern (MTP),
        Electrical Information and Communication Technology (EICT), 2013 International Conference on, IEEE,
        2014, pp. 1-5.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Convert image to CuPy array and ensure float64
    image = cp.asarray(image, dtype=cp.float64)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2
    intensities = cp.zeros((rSize * cSize, 8), dtype=cp.float64)

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

    mbp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            intensities.ravel(),
            rSize,
            cSize,
            image.shape[1]
        )
    )

    # Compute median and create binary pattern
    medianMat = cp.median(intensities, axis=1)
    MBP = (intensities > medianMat.reshape(-1, 1))

    # Create pattern using matrix multiplication
    imgDesc = cp.dot(MBP.astype(cp.uint8), 1 << cp.arange(MBP.shape[1] - 1, -1, -1)).reshape(rSize, cSize)

    # Set bin vectors
    options['binVec'] = cp.arange(256)

    # Compute MBP histogram
    MBP_hist = cp.zeros(len(options['binVec']))
    MBP_hist = cp.bincount(cp.ravel(imgDesc), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        MBP_hist = MBP_hist / cp.sum(MBP_hist)

    return MBP_hist, imgDesc