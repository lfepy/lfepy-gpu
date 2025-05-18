import cupy as cp
from lfepy.Helper.get_mapping import get_mapping

# Define the raw kernel for computing LBP
lbp_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_lbp(
    const double* image,
    double* result,
    const double* spoints,
    const int* mapping_table,
    const int neighbors,
    const int ysize,
    const int xsize,
    const int dy,
    const int dx,
    const int origy,
    const int origx,
    const int bins
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= dx || y >= dy) return;

    // Get center pixel value
    double center = image[(y + origy - 1) * xsize + (x + origx - 1)];
    double lbp_value = 0.0;

    // Compute LBP for each neighbor
    for (int i = 0; i < neighbors; i++) {
        double sp_y = spoints[i * 2];
        double sp_x = spoints[i * 2 + 1];

        // Calculate neighbor coordinates
        double y_coord = y + origy + sp_y;
        double x_coord = x + origx + sp_x;

        // Get floor, ceil and round coordinates
        int fy = (int)floor(y_coord);
        int cy = (int)ceil(y_coord);
        int ry = (int)round(y_coord);
        int fx = (int)floor(x_coord);
        int cx = (int)ceil(x_coord);
        int rx = (int)round(x_coord);

        // Check if interpolation is needed
        if (fabs(x_coord - rx) < 1e-6 && fabs(y_coord - ry) < 1e-6) {
            // No interpolation needed
            double neighbor = image[(ry - 1) * xsize + (rx - 1)];
            if (neighbor >= center) {
                lbp_value += pow(2.0, (double)i);
            }
        } else {
            // Interpolation needed
            double ty = y_coord - fy;
            double tx = x_coord - fx;

            // Calculate interpolation weights using roundn equivalent
            double w1 = round((1.0 - tx) * (1.0 - ty) * 1e6) / 1e6;
            double w2 = round(tx * (1.0 - ty) * 1e6) / 1e6;
            double w3 = round((1.0 - tx) * ty * 1e6) / 1e6;
            double w4 = round((1.0 - w1 - w2 - w3) * 1e6) / 1e6;

            // Get interpolated value
            double neighbor = w1 * image[(fy - 1) * xsize + (fx - 1)] +
                            w2 * image[(fy - 1) * xsize + (cx - 1)] +
                            w3 * image[(cy - 1) * xsize + (fx - 1)] +
                            w4 * image[(cy - 1) * xsize + (cx - 1)];

            // Round to 4 decimal places (equivalent to roundn with -4)
            neighbor = round(neighbor * 1e4) / 1e4;

            if (neighbor >= center) {
                lbp_value += pow(2.0, (double)i);
            }
        }
    }

    // Apply mapping if provided
    if (mapping_table != NULL) {
        lbp_value = mapping_table[(int)lbp_value];
    }

    result[y * dx + x] = lbp_value;
}
''', 'compute_lbp')


def descriptor_LBP(*varargin):
    """
    Compute the Local Binary Pattern (LBP) of an image with various options for radius, neighbors, mapping, and mode.
    Optimized for GPU using CuPy raw kernels.

    Args:
        image (numpy.ndarray): The input image, expected to be a 2D numpy array (grayscale).
        radius (int, optional): The radius of the LBP. Determines the distance of the sampling points from the center pixel.
        neighbors (int, optional): The number of sampling points in the LBP.
        mapping (dict or None, optional): The mapping information for LBP codes. Should contain 'samples' and 'table' if provided. If `None`, no mapping is applied.
        mode (str, optional): The mode for LBP calculation. Options are:
            'h' (histogram): Returns LBP histogram.
            'hist' (histogram): Same as 'h', returns LBP histogram.
            'nh' (normalized histogram): Returns normalized LBP histogram. Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            result (numpy.ndarray): The LBP histogram or LBP image based on the `mode` parameter.
            codeImage (numpy.ndarray): The LBP code image, which contains the LBP codes for each pixel.

    Raises:
        ValueError: If the number of input arguments is incorrect or if the provided `mapping` is incompatible with the number of `neighbors`.
        ValueError: If the input image is too small for the given `radius`.
        ValueError: If the dimensions of `spoints` are not valid.
    """
    # Check the number of input arguments
    if len(varargin) < 1 or len(varargin) > 5:
        raise ValueError("Wrong number of input arguments")

    image = cp.asarray(varargin[0])  # Ensure input is a CuPy array

    if len(varargin) == 1:
        # Default parameters
        spoints = cp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors = 8
        mapping = get_mapping(8, 'riu2')
        mode = 'nh'

    if (len(varargin) == 2) and (len(str(varargin[1])) == 1):
        raise ValueError('Input arguments')

    if (len(varargin) > 2) and (len(str(varargin[1])) == 1):
        radius = varargin[1]
        neighbors = varargin[2]

        spoints = cp.zeros((neighbors, 2))
        a = 2 * cp.pi / neighbors

        # Vectorized calculation of spoints
        indices = cp.arange(neighbors)
        spoints[:, 0] = -radius * cp.sin((indices - 1) * a)
        spoints[:, 1] = radius * cp.cos((indices - 1) * a)

        if len(varargin) >= 4:
            mapping = varargin[3]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 5:
            mode = varargin[4]
        else:
            mode = 'h'

    if (len(varargin) > 1) and (len(str(varargin[1])) > 1):
        spoints = cp.asarray(varargin[1])
        neighbors = spoints.shape[0]

        if len(varargin) >= 3:
            mapping = varargin[2]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 4:
            mode = varargin[3]
        else:
            mode = 'nh'

    # Determine the dimensions of the input image
    ysize, xsize = image.shape

    miny = cp.min(spoints[:, 0])
    maxy = cp.max(spoints[:, 0])
    minx = cp.min(spoints[:, 1])
    maxx = cp.max(spoints[:, 1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey = cp.ceil(cp.maximum(maxy, 0)) - cp.floor(cp.minimum(miny, 0))
    bsizex = cp.ceil(cp.maximum(maxx, 0)) - cp.floor(cp.minimum(minx, 0))

    # Coordinates of origin (0,0) in the block
    origy = int(1 - cp.floor(cp.minimum(miny, 0)))
    origx = int(1 - cp.floor(cp.minimum(minx, 0)))

    # Minimum allowed size for the input image depends on the radius of the used LBP operator
    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")

    # Calculate dx and dy
    dx = int(xsize - bsizex)
    dy = int(ysize - bsizey)

    # Initialize result array
    result = cp.zeros((dy, dx), dtype=cp.float64)

    # Prepare mapping table if provided
    mapping_table = None
    bins = 2 ** neighbors
    if isinstance(mapping, dict):
        mapping_table = mapping['table']
        bins = mapping['num']

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
    blocks_per_grid = ((dx + block_size - 1) // block_size, (dy + block_size - 1) // block_size)

    # Launch the kernel
    lbp_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            image.ravel(),
            result.ravel(),
            spoints.ravel(),
            mapping_table if mapping_table is not None else cp.array([], dtype=cp.int32),
            neighbors,
            ysize,
            xsize,
            dy,
            dx,
            origy,
            origx,
            bins
        )
    )

    codeImage = result

    if mode in ['h', 'hist', 'nh']:
        # Return with LBP histogram if mode equals 'hist'
        result = cp.histogram(result, bins=cp.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / cp.sum(result)
    else:
        # Otherwise return a matrix of unsigned integers
        if bins - 1 <= cp.iinfo(cp.uint8).max:
            result = result.astype(cp.uint8)
        elif bins - 1 <= cp.iinfo(cp.uint16).max:
            result = result.astype(cp.uint16)
        else:
            result = result.astype(cp.uint32)

    return result, codeImage