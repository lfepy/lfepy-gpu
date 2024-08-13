from lfepy.Helper.helper import np, descriptor_LBP, get_mapping
from PIL import Image
import matplotlib.pyplot as plt


def LBP(image, **kwargs):
    """
        Compute Local Binary Patterns (LBP) descriptors and histograms from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing LBP extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
                - radius (int): Radius for LBP computation. Default: 1.
                - mappingType (str): Type of mapping for LBP computation. Options: 'full', 'ri', 'u2', 'riu2'. Default: 'full'.

        Returns:
            - LBP_hist (numpy.ndarray): Histogram(s) of LBP descriptors.
            - imgDesc (numpy.ndarray): LBP descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = LBP(image, mode='nh', radius=1, mappingType='full')
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - T. Ojala, M. Pietikainen, and T. Maenpaa, Multi-resolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on pattern analysis and machine intelligence 24 (2002) 971-987.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision
    image = np.double(image)

    # Handle keyword arguments
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    # Extract radius and compute number of neighbors or use defaults
    if 'radius' in options:
        radius = options['radius']
        neighbors = 8 * radius
    else:
        radius = 1
        neighbors = 8

    # Extract histogram mode
    if 'mode' not in options:
        options['mode'] = 'nh'

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    mode = options['mode']

    # Handle mapping type and adjust bin vector accordingly
    if 'mappingType' in options and options['mappingType'] != 'full':
        mappingType = options['mappingType']
        mapping = get_mapping(neighbors, mappingType)
        if mappingType == 'u2':
            if radius == 1:
                options['binVec'] = np.arange(0, 59)
            elif radius == 2:
                options['binVec'] = np.arange(0, 243)
        elif mappingType == 'ri':
            if radius == 1:
                options['binVec'] = np.arange(0, 36)
            elif radius == 2:
                options['binVec'] = np.arange(0, 4117)
        else:
            if radius == 1:
                options['binVec'] = np.arange(0, 10)
            elif radius == 2:
                options['binVec'] = np.arange(0, 16)
    else:
        mapping = 0
        options['binVec'] = np.arange(0, 256)

    # Extract LBP descriptors
    _, imgDesc = descriptor_LBP(image, radius, neighbors, mapping, mode)

    # Compute LBP histogram
    LBP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LBP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LBP_hist = LBP_hist / np.sum(LBP_hist)

    return LBP_hist, imgDesc