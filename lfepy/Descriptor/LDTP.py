import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_epsi

# Define the raw kernel for computing LDTP pattern
ldtp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_ldtp_pattern(
    const double* image,
    const int* prin1,
    const int* prin2,
    double* imgDesc,
    int rows,
    int cols,
    int epsi
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    // Define link list for computing intensity differences
    int linkList[8][2][2] = {
        {{2, 3}, {2, 1}}, {{1, 3}, {3, 1}}, {{1, 2}, {3, 2}}, {{1, 1}, {3, 3}},
        {{2, 1}, {2, 3}}, {{3, 1}, {1, 3}}, {{3, 2}, {1, 2}}, {{3, 3}, {1, 1}}
    };

    // Get principal directions
    int p1 = prin1[y * cols + x];
    int p2 = prin2[y * cols + x];

    // Calculate intensity differences for principal directions
    int y1_p1 = y + linkList[p1][0][0] - 1;
    int x1_p1 = x + linkList[p1][0][1] - 1;
    int y2_p1 = y + linkList[p1][1][0] - 1;
    int x2_p1 = x + linkList[p1][1][1] - 1;
    double diffResP = image[y1_p1 * (cols + 2) + x1_p1] - image[y2_p1 * (cols + 2) + x2_p1];

    int y1_p2 = y + linkList[p2][0][0] - 1;
    int x1_p2 = x + linkList[p2][0][1] - 1;
    int y2_p2 = y + linkList[p2][1][0] - 1;
    int x2_p2 = x + linkList[p2][1][1] - 1;
    double diffResN = image[y1_p2 * (cols + 2) + x1_p2] - image[y2_p2 * (cols + 2) + x2_p2];

    // Apply threshold for texture difference
    if (diffResP <= epsi && diffResP >= -epsi) {
        diffResP = 0.0;
    } else if (diffResP < -epsi) {
        diffResP = 1.0;
    } else {
        diffResP = 2.0;
    }

    if (diffResN <= epsi && diffResN >= -epsi) {
        diffResN = 0.0;
    } else if (diffResN < -epsi) {
        diffResN = 1.0;
    } else {
        diffResN = 2.0;
    }

    // Generate LDTP descriptor
    imgDesc[y * cols + x] = 16.0 * p1 + 4.0 * diffResP + diffResN;
}
''', 'compute_ldtp_pattern')


def LDTP(image, **kwargs):
    """
    Compute Local Directional Texture Pattern (LDTP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LDTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            epsi (int): Threshold value for texture difference. Default is 15.

    Returns:
        tuple: A tuple containing:
            LDTP_hist (cupy.ndarray): Histogram(s) of LDTP descriptors.
            imgDesc (cupy.ndarray): LDTP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LDTP(image, mode='nh', epsi=15)

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        A.R. Rivera, J.R. Castillo, and O. Chae,
        Local Directional Texture Pattern Image Descriptor,
        Pattern Recognition Letters,
        vol. 51, 2015, pp. 94-100.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    epsi = validate_epsi(options)

    # Define Kirsch Masks
    Kirsch = [cp.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              cp.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              cp.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              cp.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              cp.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              cp.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              cp.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              cp.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute mask responses
    maskResponses = cp.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch):
        maskResponses[:, :, i] = convolve2d(image, kirsch_mask, mode='same')

    maskResponsesAbs = cp.abs(maskResponses) / 8

    # Sorting to get principal directions
    ind = cp.argsort(maskResponsesAbs[1:-1, 1:-1, :], axis=2)
    prin1 = ind[:, :, 0].astype(cp.int32)
    prin2 = ind[:, :, 1].astype(cp.int32)

    # Initialize output arrays
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2
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

    ldtp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            prin1.ravel(),
            prin2.ravel(),
            imgDesc.ravel(),
            rSize,
            cSize,
            epsi
        )
    )

    # Define unique bins for histogram
    uniqueBin = cp.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 32, 33,
                          34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58, 64, 65,
                          66, 68, 69, 70, 72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90, 96, 97, 98,
                          100, 101, 102, 104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122])

    # Set binVec option
    options['binVec'] = uniqueBin

    # Compute LDTP histogram
    LDTP_hist = cp.zeros(len(options['binVec']))
    LDTP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        LDTP_hist = LDTP_hist / cp.sum(LDTP_hist)

    return LDTP_hist, imgDesc