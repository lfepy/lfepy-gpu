import cupy as cp
from lfepy.Helper.roundn import roundn
from lfepy.Helper.get_mapping import get_mapping


def descriptor_LBP(*varargin):
    """
    Compute the Local Binary Pattern (LBP) of an image with various options for radius, neighbors, mapping, and mode.
    Optimized for GPU using CuPy.
    """
    # Check the number of input arguments
    if len(varargin) < 1 or len(varargin) > 5:
        raise ValueError("Wrong number of input arguments")

    image = cp.asarray(varargin[0])  # Ensure input is a CuPy array

    if len(varargin) == 1:
        spoints = cp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors = 8
        mapping = get_mapping(8, 'riu2')
        mode = 'nh'

    if len(varargin) == 2 and len(str(varargin[1])) == 1:
        raise ValueError('Input arguments')

    if len(varargin) > 2 and len(str(varargin[1])) == 1:
        radius = varargin[1]
        neighbors = varargin[2]

        spoints = cp.zeros((neighbors, 2))
        a = 2 * cp.pi / neighbors

        # Vectorized approach for computing spoints
        angles = (cp.arange(neighbors) - 1) * a
        spoints[:, 0] = -radius * cp.sin(angles)
        spoints[:, 1] = radius * cp.cos(angles)

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

    if len(varargin) > 1 and len(str(varargin[1])) > 1:
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

    ysize, xsize = image.shape
    miny, maxy = cp.min(spoints[:, 0]), cp.max(spoints[:, 0])
    minx, maxx = cp.min(spoints[:, 1]), cp.max(spoints[:, 1])

    bsizey = cp.ceil(cp.maximum(maxy, 0)) - cp.floor(cp.minimum(miny, 0))
    bsizex = cp.ceil(cp.maximum(maxx, 0)) - cp.floor(cp.minimum(minx, 0))

    origy = int(1 - cp.floor(cp.minimum(miny, 0)))
    origx = int(1 - cp.floor(cp.minimum(minx, 0)))

    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")

    dx = int(xsize - bsizex)
    dy = int(ysize - bsizey)

    C = image[origy - 1:origy + dy - 1, origx - 1:origx + dx - 1]
    d_C = C.astype(cp.float64)

    bins = 2 ** neighbors
    result = cp.zeros((dy, dx))

    # Vectorized processing for each neighbor
    y_coords = spoints[:, 0] + origy
    x_coords = spoints[:, 1] + origx

    fy = cp.floor(y_coords)
    cy = cp.ceil(y_coords)
    ry = cp.round(y_coords)
    fx = cp.floor(x_coords)
    cx = cp.ceil(x_coords)
    rx = cp.round(x_coords)

    # Using meshgrid for indexing
    fy_fx = cp.meshgrid(fy, fx, indexing='ij')
    cy_cx = cp.meshgrid(cy, cx, indexing='ij')

    for i in range(neighbors):
        # Check if values are equal, else apply bilinear interpolation
        if (cp.abs(x_coords[i] - rx[i]) < 1e-6) and (cp.abs(y_coords[i] - ry[i]) < 1e-6):
            N = image[ry[i] - 1:ry[i] + dy - 1, rx[i] - 1:rx[i] + dx - 1]
            D = N >= C
        else:
            ty = y_coords[i] - fy[i]
            tx = x_coords[i] - fx[i]

            w1 = roundn((1 - tx) * (1 - ty), -6)
            w2 = roundn(tx * (1 - ty), -6)
            w3 = roundn((1 - tx) * ty, -6)
            w4 = roundn(1 - w1 - w2 - w3, -6)

            N = (w1 * image[fy[i] - 1:fy[i] + dy - 1, fx[i] - 1:fx[i] + dx - 1] +
                 w2 * image[fy[i] - 1:fy[i] + dy - 1, cx[i] - 1:cx[i] + dx - 1] +
                 w3 * image[cy[i] - 1:cy[i] + dy - 1, fx[i] - 1:fx[i] + dx - 1] +
                 w4 * image[cy[i] - 1:cy[i] + dy - 1, cx[i] - 1:cx[i] + dx - 1])
            N = roundn(N, -4)
            D = N >= d_C

        v = 2 ** i
        result += v * D

    if isinstance(mapping, dict):
        bins = mapping['num']
        result = cp.take(mapping['table'], result.flatten().astype(cp.int16))
        result = result.reshape(dy, dx)
    codeImage = result

    if mode in ['h', 'hist', 'nh']:
        result = cp.histogram(result, bins=cp.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / cp.sum(result)
    else:
        if bins - 1 <= cp.iinfo(cp.uint8).max:
            result = result.astype(cp.uint8)
        elif bins - 1 <= cp.iinfo(cp.uint16).max:
            result = result.astype(cp.uint16)
        else:
            result = result.astype(cp.uint32)

    return result, codeImage
