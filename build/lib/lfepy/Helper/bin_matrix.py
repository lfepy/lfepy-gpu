import cupy as cp
from cupyx.scipy.ndimage import label


def bin_matrix(A, E, G, angle, bin):
    """
    Compute the bin matrix for a given angle map and gradient magnitude.

    Args:
        A (cupy.ndarray): Angle map of the gradient directions.
        E (cupy.ndarray): Binary edge map where edges are marked.
        G (cupy.ndarray): Gradient magnitude map.
        angle (float): Total range of angles in degrees (e.g., 360 for full circle).
        bin (int): Number of bins to divide the angle range into.

    Returns:
        tuple:
            bm (cupy.ndarray): Bin matrix with assigned bins for each pixel.
            bv (cupy.ndarray): Gradient magnitude values corresponding to the bin matrix.

    Example:
        >>> import numpy as np
        >>> A = np.array([[0, 45], [90, 135]])
        >>> E = np.array([[1, 1], [1, 1]])
        >>> G = np.array([[1, 2], [3, 4]])
        >>> angle = 360
        >>> bin = 8
        >>> bm, bv = bin_matrix(A, E, G, angle, bin)
        >>> print(bm)
        [[1 2]
         [3 4]]
        >>> print(bv)
        [[1. 2.]
         [3. 4.]]
    """
    # Label connected components in the edge map
    contorns, n = label(cp.array(E))

    # Get the angle range per bin
    nAngle = angle / bin

    # Vectorized bin index calculation based on the angle
    binned_angle = cp.ceil(A / nAngle).astype(cp.int32)
    binned_angle[binned_angle == 0] = 1  # Ensure bin index starts from 1

    # Initialize bin matrix and gradient magnitude matrix
    bm = cp.zeros_like(E, dtype=cp.int32)
    bv = cp.zeros_like(E, dtype=cp.float32)

    # Mask for valid gradient values
    valid_gradients = G > 0

    unique_labels = cp.unique(contorns)[1:]

    # Create a mask for all regions and valid gradients
    region_mask = cp.isin(contorns, unique_labels)
    region_valid = region_mask & valid_gradients

    # Assign the binned angle and gradient values for the valid regions
    bm = cp.where(region_valid, binned_angle, bm)
    bv = cp.where(region_valid, G, bv)

    return bm, bv