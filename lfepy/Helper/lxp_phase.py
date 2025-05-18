import cupy as cp

# CUDA kernel for LXP descriptor computation
lxp_kernel = cp.RawKernel(r'''
extern "C" __global__
void lxp_kernel(
    const float* image, const float* center_patch,
    const float* spoints, const float* bin_edges,
    float* result,
    int height, int width, int dy, int dx,
    int neighbors, int n_bins,
    int origy, int origx
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= neighbors) return;

    // Get the sampling point coordinates
    float y = spoints[tid * 2] + origy;
    float x = spoints[tid * 2 + 1] + origx;

    // Get integer coordinates for interpolation
    int fy = int(floorf(y));
    int cy = int(ceilf(y));
    int ry = int(roundf(y));
    int fx = int(floorf(x));
    int cx = int(ceilf(x));
    int rx = int(roundf(x));

    float D = 0.0f;

    if (fabsf(x - rx) < 1e-6f && fabsf(y - ry) < 1e-6f) {
        // No interpolation needed
        for (int i = 0; i < dy; i++) {
            for (int j = 0; j < dx; j++) {
                float n_val = image[(ry + i) * width + (rx + j)];
                float c_val = center_patch[i * dx + j];

                // Find bin indices
                int n_bin = 0;
                int c_bin = 0;
                for (int b = 1; b < n_bins; b++) {
                    if (n_val >= bin_edges[b-1] && n_val < bin_edges[b]) {
                        n_bin = b - 1;
                        break;
                    }
                }
                for (int b = 1; b < n_bins; b++) {
                    if (c_val >= bin_edges[b-1] && c_val < bin_edges[b]) {
                        c_bin = b - 1;
                        break;
                    }
                }

                D += (n_bin != c_bin) ? 1.0f : 0.0f;
            }
        }
    } else {
        // Bilinear interpolation
        float ty = y - fy;
        float tx = x - fx;
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;

        for (int i = 0; i < dy; i++) {
            for (int j = 0; j < dx; j++) {
                float n_val = w1 * image[(fy + i) * width + (fx + j)] +
                            w2 * image[(fy + i) * width + (cx + j)] +
                            w3 * image[(cy + i) * width + (fx + j)] +
                            w4 * image[(cy + i) * width + (cx + j)];
                float c_val = center_patch[i * dx + j];

                // Find bin indices
                int n_bin = 0;
                int c_bin = 0;
                for (int b = 1; b < n_bins; b++) {
                    if (n_val >= bin_edges[b-1] && n_val < bin_edges[b]) {
                        n_bin = b - 1;
                        break;
                    }
                }
                for (int b = 1; b < n_bins; b++) {
                    if (c_val >= bin_edges[b-1] && c_val < bin_edges[b]) {
                        c_bin = b - 1;
                        break;
                    }
                }

                D += (n_bin != c_bin) ? 1.0f : 0.0f;
            }
        }
    }

    // Compute the LXP pattern value and accumulate
    float v = powf(2.0f, tid);
    result[tid] = v * D;
}
''', 'lxp_kernel')


def lxp_phase(image, radius=1, neighbors=8, mapping=None, mode='h'):
    """
    Compute the Local X-Y Pattern (LXP) descriptor for a 2D grayscale image based on local phase information.

    Args:
        image (numpy.ndarray): 2D grayscale image.
        radius (int, optional): Radius of the circular neighborhood for computing the pattern. Default is 1.
        neighbors (int, optional): Number of directions or neighbors to consider. Default is 8.
        mapping (numpy.ndarray or None, optional): Coordinates of neighbors relative to each pixel. If None, uses a default circular pattern. If a single digit, computes neighbors in a circular pattern based on the digit. Default is None.
        mode (str, optional): Mode for output. 'h' or 'hist' for histogram of the LXP, 'nh' for normalized histogram. Default is 'h'.

    Returns:
        numpy.ndarray: LXP descriptor, either as a histogram or image depending on the `mode` parameter.

    Raises:
        ValueError: If the input image is too small for the specified radius or the coordinates are invalid.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()
        >>> lxp_desc = lxp_phase(image, radius=1, neighbors=8, mode='nh')
        >>> print(lxp_desc.shape)
        (256,)
    """
    # Convert input to CuPy array if it's not already
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image, dtype=cp.float32)

    # Define bin edges for quantizing phase values
    bin = cp.array([0, 90, 180, 270, 360], dtype=cp.float32)

    # Determine the pattern of neighbors
    if mapping is None:
        # Default 8-neighborhood pattern
        spoints = cp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=cp.float32)
    elif len(str(mapping)) == 1:
        # Compute circular pattern based on neighbors
        a = 2 * cp.pi / neighbors
        angles = cp.arange(neighbors) * a
        spoints = cp.zeros((neighbors, 2), dtype=cp.float32)
        spoints[:, 0] = -radius * cp.sin(angles)
        spoints[:, 1] = radius * cp.cos(angles)
    else:
        # Use user-defined mapping
        spoints = cp.asarray(mapping, dtype=cp.float32)

    # Get the size of the image
    ysize, xsize = image.shape

    # Determine the size of the boundary box needed for the pattern
    miny, maxy = cp.min(spoints[:, 0]), cp.max(spoints[:, 0])
    minx, maxx = cp.min(spoints[:, 1]), cp.max(spoints[:, 1])

    # Calculate size of the boundary box
    bsizey = int(cp.ceil(cp.max(cp.array([maxy.get(), 0]))) - cp.floor(cp.min(cp.array([miny.get(), 0])))) + 1
    bsizex = int(cp.ceil(cp.max(cp.array([maxx.get(), 0]))) - cp.floor(cp.min(cp.array([minx.get(), 0])))) + 1

    if xsize < bsizex or ysize < bsizey:
        raise ValueError('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')

    # Calculate offsets for cropping the image
    origy = 1 - int(cp.floor(cp.min(cp.array([miny.get(), 0]))))
    origx = 1 - int(cp.floor(cp.min(cp.array([minx.get(), 0]))))

    # Calculate sizes for the cropped image
    dy, dx = ysize - bsizey, xsize - bsizex

    # Prepare data for kernel
    center_patch = image[origy:origy + dy, origx:origx + dx].flatten()
    spoints_gpu = spoints.flatten()  # Already a CuPy array
    bin_edges_gpu = bin  # Already a CuPy array
    result_gpu = cp.zeros(neighbors, dtype=cp.float32)

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

    lxp_kernel(
        blocks_per_grid, threads_per_block,
        (image, center_patch, spoints_gpu, bin_edges_gpu, result_gpu,
         image.shape[0], image.shape[1], dy, dx,
         neighbors, len(bin), origy, origx)
    )

    # Sum up the results
    result = cp.sum(result_gpu)

    # Normalize the result or return as histogram
    if mode in ['h', 'hist', 'nh']:
        bins = 2 ** neighbors
        result = cp.histogram(result.ravel(), bins=cp.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / cp.sum(result)
    else:
        # Convert result to appropriate type based on the number of bins
        bins = 2 ** neighbors
        if bins - 1 <= cp.iinfo(cp.uint8).max:
            result = result.astype(cp.uint8)
        elif bins - 1 <= cp.iinfo(cp.uint16).max:
            result = result.astype(cp.uint16)
        else:
            result = result.astype(cp.uint32)

    return result