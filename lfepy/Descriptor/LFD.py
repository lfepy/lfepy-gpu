import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Helper import descriptor_LBP, descriptor_LPQ
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


# Define the raw kernel for pattern computation
pattern_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_pattern(
    const unsigned char* quadrantMat,
    unsigned char* pattern,
    const int height,
    const int width,
    const int rSize,
    const int cSize
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Define the 8 neighbors
    const int link[8][2] = {
        {1, 1}, {1, 2}, {1, 3},
        {2, 3}, {3, 3}, {3, 2},
        {3, 1}, {2, 1}
    };
    
    // Get center pixel value
    unsigned char x_c = quadrantMat[(y + 1) * (width + 2) + (x + 1)];
    unsigned char pattern_val = 0;
    
    // Process each neighbor
    for (int n = 0; n < 8; n++) {
        int i = link[n][0];
        int j = link[n][1];
        unsigned char x_i = quadrantMat[(y + i - 1) * (width + 2) + (x + j - 1)];
        
        if (x_c == x_i) {
            pattern_val += (1 << (7 - n));  // 2^(7-n) using bit shift
        }
    }
    
    pattern[y * width + x] = pattern_val;
}
''', 'compute_pattern')


def LFD(image, **kwargs):
    """
    Compute Local Frequency Descriptor (LFD) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LFD extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LFD_hist (cupy.ndarray): Histogram(s) of LFD descriptors.
            imgDesc (list): List of dictionaries containing LFD descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LFD(image, mode='nh')

        >>> plt.imshow(imgDesc[0]['fea'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        Z. Lei, T. Ahonen, M. Pietikäinen, and S.Z. Li,
        Local Frequency Descriptor for Low-Resolution Face Recognition,
        Automatic Face & Gesture Recognition and Workshops (FG 2011), IEEE,
        2011, pp. 161-166.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    _, filterResp = descriptor_LPQ(image, 5)
    magn = cp.abs(filterResp)

    imgDesc = [{'fea': descriptor_LBP(magn, 1, 8)[1]}]

    CoorX = cp.sign(cp.real(filterResp))
    CoorY = cp.sign(cp.imag(filterResp))

    quadrantMat = cp.ones_like(filterResp, dtype=cp.uint8)
    quadrantMat[(CoorX == -1) & (CoorY == 1)] = 2
    quadrantMat[(CoorX == -1) & (CoorY == -1)] = 3
    quadrantMat[(CoorX == 1) & (CoorY == -1)] = 4

    rSize, cSize = quadrantMat.shape[0] - 2, quadrantMat.shape[1] - 2
    x_c = quadrantMat[1:-1, 1:-1]
    pattern = cp.zeros_like(x_c, dtype=cp.uint8)

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

    pattern_kernel(blocks_per_grid, threads_per_block, (
        quadrantMat,
        pattern,
        rSize,
        cSize,
        rSize,
        cSize
    ))

    imgDesc.append({'fea': pattern.astype(cp.float64)})

    options['binVec'] = [cp.arange(256)] * 2

    # Compute LFD histogram
    LFD_hist = []
    for s in range(len(imgDesc)):
        imgReg = cp.array(imgDesc[s]['fea'])
        binVec = cp.array(options['binVec'][s])
        # Vectorized counting for each bin value
        hist, _ = cp.histogram(imgReg, bins=cp.append(binVec, cp.inf))
        LFD_hist.extend(hist)
    LFD_hist = cp.array(LFD_hist)

    if 'mode' in options and options['mode'] == 'nh':
        LFD_hist = LFD_hist / cp.sum(LFD_hist)

    return LFD_hist, imgDesc