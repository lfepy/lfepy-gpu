import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from cupyx.scipy.signal import convolve2d
from lfepy.Helper import get_mapping
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_mask_GDP

# Define the CUDA kernel for GDP computation
gdp_kernel = cp.RawKernel(r'''
extern "C" __global__
void gdp_kernel(
    const float* angles,
    float* GDPdecimal,
    const int* link,
    const float t,
    const int rSize,
    const int cSize,
    const int linkSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < cSize && idy < rSize) {
        float x_c = angles[(idy + 1) * (cSize + 2) + (idx + 1)];
        float sum = 0.0f;

        for (int n = 0; n < linkSize; n++) {
            int corner_y = link[n * 2] - 1;
            int corner_x = link[n * 2 + 1] - 1;
            float x_i = angles[(idy + corner_y) * (cSize + 2) + (idx + corner_x)];
            float diff = x_i - x_c;

            if (diff <= t && diff >= -t) {
                sum += powf(2.0f, 8.0f - n - 1.0f);
            }
        }

        GDPdecimal[idy * cSize + idx] = sum;
    }
}
''', 'gdp_kernel')


def GDP(image, **kwargs):
    """
    Compute Gradient Directional Pattern (GDP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing GDP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            mask (str): Mask type for gradient computation. Options: 'sobel', 'prewitt'. Default is 'sobel'.
            t (float): Threshold value for gradient angle difference. Default is 22.5.

    Returns:
        tuple: A tuple containing:
            GDP_hist (cupy.ndarray): Histogram(s) of GDP descriptors.
            imgDesc (cupy.ndarray): GDP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` or `mask` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = GDP(image, mode='nh', mask='sobel', t=22.5)

        >>> plt.imshow(imgDesc.get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Ahmed,
        "Gradient directional pattern: a robust feature descriptor for facial expression recognition",
        in *Electronics letters*,
        vol. 48, no. 23, pp. 1203-1204, 2012.

        W. Chu,
        Facial expression recognition based on local binary pattern and gradient directional pattern,
        in Green Computing and Communications (GreenCom), 2013 IEEE and Internet of Things (iThings/CPSCom), IEEE,
        2013, pp. 1458-1462.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    options, t = validate_mask_GDP(options)

    EPSILON = 1e-7

    # Define masks for Sobel or Prewitt
    if options['mask'] == 'sobel':
        maskA = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        maskB = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        link = cp.array([[1, 2], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3], [2, 3], [1, 3]])
    elif options['mask'] == 'prewitt':
        maskA = cp.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        maskB = cp.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        link = cp.array([[3, 1], [3, 2], [3, 3], [2, 3], [1, 3], [1, 2], [1, 1], [2, 1]])

    Gx = convolve2d(image, maskA, mode='same')
    Gy = convolve2d(image, maskB, mode='same')
    angles = cp.arctan2(Gy, Gx + EPSILON)
    angles = cp.degrees(angles) + 90

    x_c = angles[1:-1, 1:-1]
    rSize, cSize = x_c.shape
    GDPdecimal = cp.zeros((rSize, cSize), dtype=cp.float32)

    # Prepare data for kernel
    angles = cp.ascontiguousarray(angles, dtype=cp.float32)
    link = cp.ascontiguousarray(link, dtype=cp.int32)
    t = cp.float32(t)

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
    gdp_kernel(threads_per_block, blocks_per_grid, (
        angles,
        GDPdecimal,
        link,
        t,
        rSize,
        cSize,
        link.shape[0]
    ))

    if options['mask'] == 'prewitt':
        mapping = get_mapping(8, 'u2')
        GDPdecimal = cp.array(mapping['table'])[GDPdecimal.astype(int)]
        binNum = mapping['num']
    else:
        binNum = 256

    imgDesc = GDPdecimal

    # Set bin vectors
    options['binVec'] = cp.arange(binNum)

    # Compute GDP histogram
    GDP_hist = cp.zeros(len(options['binVec']))
    GDP_hist = cp.bincount(cp.searchsorted(options['binVec'], cp.ravel(imgDesc)), minlength=len(options['binVec']))

    if 'mode' in options and options['mode'] == 'nh':
        GDP_hist = GDP_hist / cp.sum(GDP_hist)

    return GDP_hist, imgDesc