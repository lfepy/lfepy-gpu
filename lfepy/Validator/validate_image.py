import cupy as cp
import numpy as np


def validate_image(image):
    # Input validation
    if image is None or not isinstance(image, np.ndarray) and not isinstance(image, cp.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray or cupy.ndarray.")

    if image.dtype != cp.ndarray:
        image = cp.array(image)

    # Convert the input image to double precision if needed
    if image.dtype != cp.float64:
        image = cp.asarray(image, dtype=cp.float64)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Convert the weights list to a CuPy array
        weights = cp.array([0.2989, 0.5870, 0.1140])
        image = cp.dot(image[..., :3], weights)

    return image
