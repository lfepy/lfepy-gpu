import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cupyx.jit.rawkernel is experimental.*")
import cupy as cp
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_T, validate_N, validate_scaleTop

# Define the raw kernels for computing WLD patterns
wld_kernels = cp.RawKernel(r'''
#define M_PI 3.14159265358979323846

extern "C" __global__
void compute_wld_pattern(
    const double* image,
    double* imgGO,
    double* imgDE,
    int rows,
    int cols,
    int scale,
    double BELTA,
    double ALPHA,
    double EPSILON
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    double x_c = image[(y + scale) * (cols + 2 * scale) + (x + scale)];
    double V00 = 0.0;
    double V03 = 0.0;
    double V04 = 0.0;

    // Define scale-specific cell configurations
    if (scale == 1) {
        // Link1 for scale 1
        int link1[8][2] = {
            {1, 1}, {1, 2}, {1, 3}, {2, 3},
            {3, 3}, {3, 2}, {3, 1}, {2, 1}
        };
        // Compute V00
        for (int i = 0; i < 8; i++) {
            int y_pos = y + link1[i][0] - 1;
            int x_pos = x + link1[i][1] - 1;
            V00 += image[y_pos * (cols + 2 * scale) + x_pos];
        }
        V00 -= 8 * x_c;

        // Link2 for scale 1
        int link2[4][2] = {
            {3, 2}, {1, 2}, {2, 1}, {2, 3}
        };
        V03 = image[(y + link2[0][0] - 1) * (cols + 2 * scale) + (x + link2[0][1] - 1)] - 
              image[(y + link2[1][0] - 1) * (cols + 2 * scale) + (x + link2[1][1] - 1)];
        V04 = image[(y + link2[2][0] - 1) * (cols + 2 * scale) + (x + link2[2][1] - 1)] - 
              image[(y + link2[3][0] - 1) * (cols + 2 * scale) + (x + link2[3][1] - 1)];
    }
    else if (scale == 2) {
        // Link1 for scale 2
        int link1[16][2] = {
            {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5},
            {2, 5}, {3, 5}, {4, 5}, {5, 5}, {5, 4},
            {5, 3}, {5, 2}, {5, 1}, {4, 1}, {3, 1}, {2, 1}
        };
        // Compute V00
        for (int i = 0; i < 16; i++) {
            int y_pos = y + link1[i][0] - 1;
            int x_pos = x + link1[i][1] - 1;
            V00 += image[y_pos * (cols + 2 * scale) + x_pos];
        }
        V00 -= 16 * x_c;

        // Link2 for scale 2
        int link2[4][2] = {
            {5, 3}, {1, 3}, {3, 1}, {3, 5}
        };
        V03 = image[(y + link2[0][0] - 1) * (cols + 2 * scale) + (x + link2[0][1] - 1)] - 
              image[(y + link2[1][0] - 1) * (cols + 2 * scale) + (x + link2[1][1] - 1)];
        V04 = image[(y + link2[2][0] - 1) * (cols + 2 * scale) + (x + link2[2][1] - 1)] - 
              image[(y + link2[3][0] - 1) * (cols + 2 * scale) + (x + link2[3][1] - 1)];
    }
    else if (scale == 3) {
        // Link1 for scale 3
        int link1[24][2] = {
            {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},
            {2, 7}, {3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7}, {7, 6},
            {7, 5}, {7, 4}, {7, 3}, {7, 2}, {7, 1}, {6, 1}, {5, 1},
            {4, 1}, {3, 1}, {2, 1}
        };
        // Compute V00
        for (int i = 0; i < 24; i++) {
            int y_pos = y + link1[i][0] - 1;
            int x_pos = x + link1[i][1] - 1;
            V00 += image[y_pos * (cols + 2 * scale) + x_pos];
        }
        V00 -= 24 * x_c;

        // Link2 for scale 3
        int link2[4][2] = {
            {7, 4}, {1, 4}, {4, 1}, {4, 7}
        };
        V03 = image[(y + link2[0][0] - 1) * (cols + 2 * scale) + (x + link2[0][1] - 1)] - 
              image[(y + link2[1][0] - 1) * (cols + 2 * scale) + (x + link2[1][1] - 1)];
        V04 = image[(y + link2[2][0] - 1) * (cols + 2 * scale) + (x + link2[2][1] - 1)] - 
              image[(y + link2[3][0] - 1) * (cols + 2 * scale) + (x + link2[3][1] - 1)];
    }

    // Compute DE (Differential Excitation)
    double de = atan((ALPHA * V00) / (x_c + BELTA)) * 180.0 / M_PI + 90.0;
    imgDE[idx] = de;

    // Compute GO (Gradient Orientation)
    if (V03 == 0.0) V03 = EPSILON;
    double go = atan(V04 / V03) * 180.0 / M_PI;
    if (V03 < 0.0) go += 180.0;
    else if (V04 < 0.0) go += 360.0;
    imgGO[idx] = go;
}
''', 'compute_wld_pattern')

# Create kernel object
compute_wld_kernel = cp.RawKernel(wld_kernels.code, 'compute_wld_pattern')


def WLD(image, **kwargs):
    """
    Compute Weber Local Descriptor (WLD) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing WLD extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            T (int): Number of bins for gradient orientation. Default is 8.
            N (int): Number of bins for differential excitation. Default is 4.
            scaleTop (int): Number of scales to consider for WLD computation. Default is 1.

    Returns:
        tuple: A tuple containing:
            WLD_hist (cupy.ndarray): Histogram of WLD descriptors.
            imgDesc (list of dicts): List of dictionaries containing WLD descriptors for each scale.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = WLD(image, mode='nh', T=8, N=4, scaleTop=1)

        >>> plt.imshow(imgDesc[0]['fea']['GO'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()
        >>> plt.imshow(imgDesc[1]['fea']['DE'].get(), cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        S. Li, D. Gong, and Y. Yuan,
        Face recognition using Weber local descriptors.,
        Neurocomputing,
        122 (2013) 272-283.

        S. Liu, Y. Zhang, and K. Liu,
        Facial expression recognition under partial occlusion based on Weber Local Descriptor histogram and decision fusion,
        Control Conference (CCC), 2014 33rd Chinese, IEEE,
        2014, pp. 4664-4668.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    T = validate_T(options)
    N = validate_N(options)
    scaleTop = validate_scaleTop(options)

    BELTA = 5.0
    ALPHA = 3.0
    EPSILON = 1e-7

    imgDescs = []

    # Compute WLD descriptors
    for scale in range(1, scaleTop + 1):
        # Extract central region of the image
        x_c = image[scale:-scale, scale:-scale]
        rSize, cSize = x_c.shape

        # Allocate output arrays
        imgGO = cp.zeros((rSize, cSize), dtype=cp.float64)
        imgDE = cp.zeros((rSize, cSize), dtype=cp.float64)

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

        compute_wld_kernel(
            blocks_per_grid,
            threads_per_block,
            (
                image.ravel(),
                imgGO.ravel(),
                imgDE.ravel(),
                rSize,
                cSize,
                scale,
                BELTA,
                ALPHA,
                EPSILON
            )
        )

        imgDesc = [{'fea': {'GO': imgGO}}, {'fea': {'DE': imgDE}}]
        imgDescs.append(imgDesc)

    options['binVec'] = []
    options['wldHist'] = 1

    # Compute WLD histogram
    WLD_hist = []
    for desc in imgDescs:
        imgGO = desc[0]['fea']['GO']
        imgDE = desc[1]['fea']['DE']

        # Quantize GO and DE
        range_GO = 360.0 / T
        imgGO_quant = cp.floor(imgGO / range_GO)

        range_DE = 180.0 / N
        imgDE_quant = cp.floor(imgDE / range_DE)

        # Compute histogram
        hh = []
        for t in range(T):
            orien = imgDE_quant[imgGO_quant == t]
            orienHist, _ = cp.histogram(orien, bins=cp.arange(N + 1))
            hh.extend(orienHist.get())
        WLD_hist.extend(hh)

    WLD_hist = cp.array(WLD_hist)

    if 'mode' in options and options['mode'] == 'nh':
        WLD_hist = WLD_hist / cp.sum(WLD_hist)

    return WLD_hist, imgDesc