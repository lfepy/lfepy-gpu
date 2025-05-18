import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_t_MTP

# Define the raw kernel for computing MTP pattern
mtp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_mtp_pattern(
    const float* image, 
    float* Pmtp_pattern,
    float* Nmtp_pattern,
    int rows, 
    int cols,
    float t
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int link[8][2] = {
        {2, 1}, {1, 1}, {1, 2}, {1, 3},
        {2, 3}, {3, 3}, {3, 2}, {3, 1}
    };

    int idx = y * cols + x;
    float intensities[8];
    float median = 0.0;

    // Load intensities from neighboring pixels
    for (int n = 0; n < 8; ++n) {
        int y_pos = y + link[n][0] - 1;
        int x_pos = x + link[n][1] - 1;
        intensities[n] = image[y_pos * (cols + 2) + x_pos];
    }

    // Compute median (simple sort for 8 elements)
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7 - i; ++j) {
            if (intensities[j] > intensities[j+1]) {
                float temp = intensities[j];
                intensities[j] = intensities[j+1];
                intensities[j+1] = temp;
            }
        }
    }
    median = (intensities[3] + intensities[4]) / 2.0;

    // Compute Pmtp and Nmtp patterns
    unsigned char Pmtp_val = 0;
    unsigned char Nmtp_val = 0;

    for (int n = 0; n < 8; ++n) {
        int y_pos = y + link[n][0] - 1;
        int x_pos = x + link[n][1] - 1;
        float val = image[y_pos * (cols + 2) + x_pos];

        if (val > (median + t)) {
            Pmtp_val |= (1 << (7 - n));
        }
        if (val < (median - t)) {
            Nmtp_val |= (1 << (7 - n));
        }
    }

    Pmtp_pattern[idx] = Pmtp_val;
    Nmtp_pattern[idx] = Nmtp_val;
}
''', 'compute_mtp_pattern')


def MTP(image, **kwargs):
    """
    Compute Median Ternary Pattern (MTP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            t (float): Threshold value for MTP computation. Default is 10.

    Returns:
        tuple: A tuple containing:
            MTP_hist (cupy.ndarray): Histogram(s) of MTP descriptors.
            imgDesc (list of dicts): List of dictionaries containing MTP descriptors for positive and negative thresholds.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MTP(image, mode='nh', t=10)

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
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
    t = validate_t_MTP(options)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Ensure image is float32
    image = image.astype(cp.float32)

    # Allocate pattern outputs
    Pmtp_pattern = cp.zeros((rSize, cSize), dtype=cp.float32)
    Nmtp_pattern = cp.zeros((rSize, cSize), dtype=cp.float32)

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

    mtp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            Pmtp_pattern.ravel(),
            Nmtp_pattern.ravel(),
            rSize,
            cSize,
            cp.float32(t)
        )
    )

    imgDesc = [
        {'fea': Pmtp_pattern.astype(cp.uint8)},
        {'fea': Nmtp_pattern.astype(cp.uint8)}
    ]

    options['binVec'] = [cp.arange(256), cp.arange(256)]

    # Compute MTP histogram
    MTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        MTP_hist.extend(hist)
    MTP_hist = cp.array(MTP_hist)

    if 'mode' in options and options['mode'] == 'nh':
        MTP_hist = MTP_hist / cp.sum(MTP_hist)

    return MTP_hist, imgDesc