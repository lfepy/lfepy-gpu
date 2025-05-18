import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx import jit
from lfepy.Validator import validate_image, validate_kwargs, validate_mode

# Define the raw kernel for computing GDP2 pattern
gdp2_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_gdp2_pattern(
    const float* image, float* pattern, int rows, int cols
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int linkList[8][4] = {
        {1, 1, 3, 3},
        {1, 2, 3, 2},
        {1, 3, 3, 1},
        {2, 3, 2, 1}
    };

    int idx = y * cols + x;
    float val = 0;

    for (int n = 0; n < 4; ++n) {
        int y1 = y + linkList[n][0] - 1;
        int x1 = x + linkList[n][1] - 1;
        int y2 = y + linkList[n][2] - 1;
        int x2 = x + linkList[n][3] - 1;

        float diff = image[y1 * (cols + 2) + x1] - image[y2 * (cols + 2) + x2];
        if (diff >= 0) {
            val += (1 << (3 - n));
        }
    }

    pattern[idx] = val;
}
''', 'compute_gdp2_pattern')


def GDP2(image, **kwargs):
    """
    Compute Gradient Direction Pattern (GDP2) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing GDP2 extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            GDP2_hist (cupy.ndarray): Histogram(s) of GDP2 descriptors.
            imgDesc (cupy.ndarray): GDP2 descriptors.

    Raises:
        TypeError: If the `image` is not a valid `cupy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option ('nh' or 'h').

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = GDP2(image, mode='nh')

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Gender Classification using Gradient Direction Pattern,
        in Science International,
        vol. 25, 2013.
    """
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Ensure image is float32
    image = image.astype(cp.float32)

    # Crop center region for output size
    x_c = image[1:-1, 1:-1]
    rSize, cSize = x_c.shape

    # Allocate pattern output
    pattern = cp.zeros((rSize, cSize), dtype=cp.float32)

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

    gdp2_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            pattern.ravel(),
            rSize,
            cSize
        )
    )

    imgDesc = pattern.astype(cp.uint8)

    binNum = cp.array(2 ** 4)
    transitionSelected = cp.array([0, 1, 3, 7, 8, 12, 14, 15])
    options['selected'] = transitionSelected
    options['binVec'] = cp.arange(binNum)

    # Histogram computation
    GDP2_hist = cp.zeros(len(options['binVec']))
    GDP2_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))
    GDP2_hist = GDP2_hist[transitionSelected]

    if 'mode' in options and options['mode'] == 'nh':
        GDP2_hist = GDP2_hist / cp.sum(GDP2_hist)

    return GDP2_hist, imgDesc