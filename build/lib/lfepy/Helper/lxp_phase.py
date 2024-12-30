import cupy as cp


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
    # Define bin edges for quantizing phase values
    bin = cp.array([0, 90, 180, 270, 360])

    # Determine the pattern of neighbors
    if mapping is None:
        # Default 8-neighborhood pattern
        spoints = cp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    elif len(str(mapping)) == 1:
        # Compute circular pattern based on neighbors
        spoints = cp.zeros((neighbors, 2))
        a = 2 * cp.pi / neighbors
        for i in range(neighbors):
            spoints[i, 0] = -radius * cp.sin((i - 1) * a)
            spoints[i, 1] = radius * cp.cos((i - 1) * a)
    else:
        # Use user-defined mapping
        spoints = cp.array(mapping)

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

    # Crop the image to match the boundary box size
    C = image[origy:origy + dy, origx:origx + dx]

    # Initialize the result array
    bins = 2 ** neighbors
    result = cp.zeros((dy, dx), dtype=cp.float32)

    # Compute the LXP descriptor
    for i in range(neighbors):
        y, x = spoints[i, 0] + origy, spoints[i, 1] + origx
        fy, cy, ry = int(cp.floor(y)), int(cp.ceil(y)), int(cp.round(y))
        fx, cx, rx = int(cp.floor(x)), int(cp.ceil(x)), int(cp.round(x))

        if abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6:
            # No interpolation needed
            N = image[ry:ry + dy, rx:rx + dx]
            tem_N, tem_C = N.copy(), C.copy()
            for tem_i in range(1, bin.shape[0]):
                tem_N[(N >= bin[tem_i - 1]) & (N < bin[tem_i])] = tem_i - 1
                tem_C[(C >= bin[tem_i - 1]) & (C < bin[tem_i])] = tem_i - 1
            D = (tem_N != tem_C).astype(cp.float32)
        else:
            # Bilinear interpolation
            ty, tx = y - fy, x - fx
            w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

            N = (w1 * image[fy:fy + dy, fx:fx + dx] +
                 w2 * image[fy:fy + dy, cx:cx + dx] +
                 w3 * image[cy:cy + dy, fx:fx + dx] +
                 w4 * image[cy:cy + dy, cx:cx + dx])
            tem_N, tem_C = N.copy(), C.copy()
            for tem_i in range(1, bin.shape[0]):
                tem_N[(N >= bin[tem_i - 1]) & (N < bin[tem_i])] = tem_i - 1
                tem_C[(C >= bin[tem_i - 1]) & (C < bin[tem_i])] = tem_i - 1
            D = (tem_N != tem_C).astype(cp.float32)

        # Compute the LXP pattern value and accumulate
        v = 2 ** (i - 1)
        result += v * D

    # Normalize the result or return as histogram
    if mode in ['h', 'hist', 'nh']:
        result = cp.histogram(result.ravel(), bins=cp.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / cp.sum(result)
    else:
        # Convert result to appropriate type based on the number of bins
        if bins - 1 <= cp.iinfo(cp.uint8).max:
            result = result.astype(cp.uint8)
        elif bins - 1 <= cp.iinfo(cp.uint16).max:
            result = result.astype(cp.uint16)
        else:
            result = result.astype(cp.uint32)

    return result