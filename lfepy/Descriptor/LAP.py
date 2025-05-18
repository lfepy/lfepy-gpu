import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing LAP patterns
lap_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_lap_patterns(
    const float* image, 
    float* pattern1, 
    float* pattern2, 
    int rows, 
    int cols,
    int image_width
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    // Link lists for pattern1 (4 pairs)
    int linkList1[4][4] = {
        {2, 2, 4, 4},
        {2, 3, 4, 3},
        {2, 4, 4, 2},
        {3, 4, 3, 2}
    };

    // Link lists for pattern2 (8 pairs)
    int linkList2[8][4] = {
        {1, 1, 5, 5},
        {1, 2, 5, 4},
        {1, 3, 5, 3},
        {1, 4, 5, 2},
        {1, 5, 5, 1},
        {2, 5, 4, 1},
        {3, 5, 3, 1},
        {4, 5, 2, 1}
    };

    int idx = y * cols + x;
    float val1 = 0;
    float val2 = 0;

    // Compute pattern1
    for (int n = 0; n < 4; ++n) {
        int y1 = y + linkList1[n][0] - 1;
        int x1 = x + linkList1[n][1] - 1;
        int y2 = y + linkList1[n][2] - 1;
        int x2 = x + linkList1[n][3] - 1;

        float diff = image[y1 * image_width + x1] - image[y2 * image_width + x2];
        if (diff > 0) {
            val1 += (1 << (3 - n));
        }
    }

    // Compute pattern2
    for (int n = 0; n < 8; ++n) {
        int y1 = y + linkList2[n][0] - 1;
        int x1 = x + linkList2[n][1] - 1;
        int y2 = y + linkList2[n][2] - 1;
        int x2 = x + linkList2[n][3] - 1;

        float diff = image[y1 * image_width + x1] - image[y2 * image_width + x2];
        if (diff > 0) {
            val2 += (1 << (7 - n));
        }
    }

    pattern1[idx] = val1;
    pattern2[idx] = val2;
}
''', 'compute_lap_patterns')


def LAP(image, **kwargs):
    """
    Compute Local Arc Pattern (LAP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LAP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LAP_hist (cupy.ndarray): Histogram(s) of LAP descriptors.
            imgDesc (list): List of dictionaries containing LAP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LAP(image, mode='nh')

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam, and S. Auwatanamongkol,
        Facial Expression Recognition using Local Arc Pattern,
        Trends in Applied Sciences Research,
        vol. 9, pp. 113, 2014.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Ensure image is float32
    image = image.astype(cp.float32)

    # Crop center region for output size
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape

    # Allocate pattern outputs
    pattern1 = cp.zeros((rSize, cSize), dtype=cp.float32)
    pattern2 = cp.zeros((rSize, cSize), dtype=cp.float32)

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

    lap_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            pattern1.ravel(),
            pattern2.ravel(),
            rSize,
            cSize,
            image.shape[1]  # Original image width
        )
    )

    imgDesc = [{'fea': pattern1.astype(cp.uint8)}, {'fea': pattern2.astype(cp.uint8)}]

    # Set bin vectors
    binVec = [cp.arange(0, 2 ** 4), cp.arange(0, 2 ** 8)]  # 4 bits for pattern1, 8 bits for pattern2
    options['binVec'] = binVec

    # Compute LAP histogram
    LAP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        binVec = options['binVec'][s]
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        LAP_hist.extend(hist)
    LAP_hist = cp.array(LAP_hist)

    if 'mode' in options and options['mode'] == 'nh':
        LAP_hist = LAP_hist / cp.sum(LAP_hist)

    return LAP_hist, imgDesc