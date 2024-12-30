import cupy as cp
from lfepy.Helper.gauss import gauss
from lfepy.Helper.dgauss import dgauss


def gauss_gradient(sigma):
    """
    Generate a set of 2-D Gaussian derivative kernels for gradient computation at multiple orientations.

    Args:
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        cupy.ndarray: A 3D array where each 2D slice represents a Gaussian derivative kernel at a specific orientation.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> sigma = 1.0
        >>> kernels = gauss_gradient(sigma)
        >>> fig, axes = plt.subplots(1, 8, figsize=(20, 5))
        >>> for i in range(8):
        ...     axes[i].imshow(kernels[:, :, i], cmap='gray')
        ...     axes[i].set_title(f'{i*45} degrees')
        ...     axes[i].axis('off')
        >>> plt.tight_layout()
        >>> plt.show()
    """
    epsilon = 1e-2
    halfsize = cp.ceil(sigma * cp.sqrt(-2 * cp.log(cp.sqrt(2 * cp.pi) * sigma * epsilon)))
    size = int(2 * halfsize + 1)

    # Generate a 2-D Gaussian kernel along x direction
    hx = cp.zeros((size, size))
    for i in range(size):
        for j in range(size):
            u = [i - halfsize - 1, j - halfsize - 1]
            hx[i, j] = gauss(u[0] - halfsize + 1, sigma) * dgauss(u[1], sigma)

    hx = hx / cp.sqrt(cp.sum(cp.abs(hx) * cp.abs(hx)))

    # Generate a 2-D Gaussian kernel along y direction
    D = cp.zeros((hx.shape[0], hx.shape[1], 8))
    D[:, :, 0] = hx

    # Rotations using CuPy
    def rotate_image(image, angle):
        angle_rad = cp.deg2rad(angle)
        cos_angle = cp.cos(angle_rad)
        sin_angle = cp.sin(angle_rad)
        center = (cp.array(image.shape) - 1) / 2
        coords = cp.meshgrid(cp.arange(image.shape[0]), cp.arange(image.shape[1]), indexing='ij')
        coords = cp.stack(coords, axis=-1).astype(cp.float32) - center
        new_coords = cp.empty_like(coords)
        new_coords[..., 0] = coords[..., 0] * cos_angle - coords[..., 1] * sin_angle
        new_coords[..., 1] = coords[..., 0] * sin_angle + coords[..., 1] * cos_angle
        new_coords += center
        return cp.clip(new_coords, 0, image.shape[0] - 1).astype(cp.int32)

    for idx, angle in enumerate(range(45, 360, 45)):
        rotated_indices = rotate_image(hx, angle)
        rotated_image = cp.zeros_like(hx)
        rotated_image[rotated_indices[..., 0], rotated_indices[..., 1]] = hx
        D[:, :, idx + 1] = rotated_image

    return D