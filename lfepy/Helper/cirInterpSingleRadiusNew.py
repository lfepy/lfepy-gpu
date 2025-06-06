import cupy as cp


def cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius):
    """
    Extract circularly interpolated image blocks around a specified radius and number of points.

    Args:
        img (numpy.ndarray): The input grayscale image.
        lbpPoints (int): The number of points used in the LBP pattern.
        lbpRadius (int): The radius of the circular neighborhood.

    Returns:
        tuple:
            blocks (numpy.ndarray): A 2D array where each row represents a circularly interpolated block.
            dx (int): The width of the output blocks.
            dy (int): The height of the output blocks.

    Raises:
        ValueError: If the input image is too small. The image should be at least (2*radius + 1) x (2*radius + 1).

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> blocks, dx, dy = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
        >>> print(blocks.shape)
    """
    imgH, imgW = img.shape

    # Dimensions of the new image after considering the radius
    imgNewH = imgH - 2 * lbpRadius
    imgNewW = imgW - 2 * lbpRadius

    # Initialize the blocks array
    blocks = cp.zeros((lbpPoints, imgNewH * imgNewW))

    radius = lbpRadius
    neighbors = lbpPoints
    spoints = cp.zeros((neighbors, 2))
    angleStep = 2 * cp.pi / neighbors

    # Vectorized calculation of sampling points
    angles = cp.arange(neighbors) * angleStep
    spoints[:, 0] = -radius * cp.sin(angles)
    spoints[:, 1] = radius * cp.cos(angles)

    miny, maxy = cp.min(spoints[:, 0]), cp.max(spoints[:, 0])
    minx, maxx = cp.min(spoints[:, 1]), cp.max(spoints[:, 1])

    bsizey = int(cp.ceil(max(maxy, 0)) - cp.floor(min(miny, 0)) + 1)
    bsizex = int(cp.ceil(max(maxx, 0)) - cp.floor(min(minx, 0)) + 1)

    origy = 1 - cp.floor(min(miny, 0)).astype(int)
    origx = 1 - cp.floor(min(minx, 0)).astype(int)

    # Check if the image is large enough
    if imgW < bsizex or imgH < bsizey:
        raise ValueError('Input image is too small. Should be at least (2*radius+1) x (2*radius+1)')

    dx = imgW - bsizex
    dy = imgH - bsizey

    # Extract circularly interpolated blocks
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx

        fy, cy, ry = cp.floor(y).astype(int), cp.ceil(y).astype(int), cp.round(y).astype(int)
        fx, cx, rx = cp.floor(x).astype(int), cp.ceil(x).astype(int), cp.round(x).astype(int)

        if cp.abs(x - rx) < 1e-6 and cp.abs(y - ry) < 1e-6:
            # No interpolation needed if exact coordinates are integer
            imgNew = img[ry - 1:ry + dy, rx - 1:rx + dx]
            blocks[i, :] = imgNew.ravel()
        else:
            # Perform bilinear interpolation
            ty, tx = y - fy, x - fx
            w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

            imgNew = (w1 * img[fy - 1:fy + dy, fx - 1:fx + dx] +
                      w2 * img[fy - 1:fy + dy, cx - 1:cx + dx] +
                      w3 * img[cy - 1:cy + dy, fx - 1:fx + dx] +
                      w4 * img[cy - 1:cy + dy, cx - 1:cx + dx])
            blocks[i, :] = imgNew.ravel()

    return blocks, dx, dy