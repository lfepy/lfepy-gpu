import cupy as cp


def cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius):
    """
    Perform circular interpolation for a single radius in the LBP (Local Binary Pattern) computation.

    Args:
        img (numpy.ndarray): 2D grayscale image.
        lbpPoints (int): Number of points used in the LBP pattern.
        lbpRadius (int): Radius of the circular neighborhood for computing LBP.

    Returns:
        tuple:
            blocks (numpy.ndarray): Array of size (lbpPoints, imgNewH * imgNewW) containing the interpolated pixel values.
            dx (int): Width of the output blocks.
            dy (int): Height of the output blocks.

    Raises:
        ValueError: If the input image is smaller than the required size of (2*radius + 1) x (2*radius + 1).

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> blocks, dx, dy = cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius)
        >>> print(blocks.shape)
        (8, 9216)
    """
    # Get image dimensions
    imgH, imgW = img.shape

    # Compute dimensions of the output blocks
    imgNewH = imgH - 2 * lbpRadius
    imgNewW = imgW - 2 * lbpRadius

    # Initialize the blocks array to store interpolated values
    blocks = cp.zeros((lbpPoints, imgNewH * imgNewW))

    # Create circular pattern points
    radius = lbpRadius
    neighbors = lbpPoints
    spoints = cp.zeros((neighbors, 2))
    angleStep = 2 * cp.pi / neighbors

    for i in range(neighbors):
        spoints[i, 0] = -radius * cp.sin(i * angleStep)
        spoints[i, 1] = radius * cp.cos(i * angleStep)

    # Calculate the size of the blocks considering boundary effects
    miny, maxy = cp.min(spoints[:, 0]), cp.max(spoints[:, 0])
    minx, maxx = cp.min(spoints[:, 1]), cp.max(spoints[:, 1])

    bsizey = int(cp.ceil(max(maxy, 0)) - cp.floor(min(miny, 0)) + 1)
    bsizex = int(cp.ceil(max(maxx, 0)) - cp.floor(min(minx, 0)) + 1)

    origy = 1 - cp.floor(min(miny, 0)).astype(int)
    origx = 1 - cp.floor(min(minx, 0)).astype(int)

    # Check if image size is sufficient
    if imgW < bsizex or imgH < bsizey:
        raise ValueError('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')

    # Compute block dimensions
    dx = imgW - bsizex
    dy = imgH - bsizey

    # Perform circular interpolation
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx

        fy, cy, ry = cp.floor(y).astype(int), cp.ceil(y).astype(int), cp.round(y).astype(int)
        fx, cx, rx = cp.floor(x).astype(int), cp.ceil(x).astype(int), cp.round(x).astype(int)

        if cp.abs(x - rx) < 1e-6 and cp.abs(y - ry) < 1e-6:
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