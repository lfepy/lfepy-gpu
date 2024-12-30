import cupy as cp


def get_mapping(samples, mappingtype):
    """
    Generate a mapping table for Local Binary Patterns (LBP) codes.

    Args:
        samples (int): The number of sampling points in the LBP.
        mappingtype (str): The type of LBP mapping. Options are:
            'u2' (uniform 2)
            'ri' (rotation invariant)
            'riu2' (uniform and rotation invariant)

    Returns:
        dict: A dictionary with the following keys:
            'table' (cupy.ndarray): The mapping table.
            'samples' (int): The number of sampling points.
            'num' (int): The number of patterns in the resulting LBP code.

    Raises:
        ValueError: If an unsupported mapping type is provided.

    Example:
        >>> get_mapping(8, 'u2')
        {'table': array([...]), 'samples': 8, 'num': 59}
    """
    table = cp.arange(2 ** samples, dtype=cp.int32)
    newMax = 0  # Number of patterns in the resulting LBP code
    index = 0

    if mappingtype == 'u2':  # Uniform 2
        # The maximum number of uniform patterns for given samples
        newMax = samples * (samples - 1) + 3
        # Vectorized computation of binary numbers and their rotations
        bin_table = cp.unpackbits(cp.arange(2 ** samples, dtype=cp.uint8))  # Unpack bits
        # Reshape the result to 2D (each row represents a number, each column a bit)
        bin_table = bin_table.reshape(-1, 8)[:, :samples]  # Only keep the 'samples' bits
        shifted_bin_table = cp.roll(bin_table, -1, axis=1)

        # Compute transitions for each pattern in parallel
        transitions = cp.sum(bin_table != shifted_bin_table, axis=1)
        uniform_mask = transitions <= 2

        # Apply uniform mapping in parallel
        table[uniform_mask] = cp.arange(index, index + cp.sum(uniform_mask))
        table[~uniform_mask] = newMax - 1
        index += cp.sum(uniform_mask)

    elif mappingtype == 'ri':  # Rotation invariant
        tmpMap = cp.full(2 ** samples, -1, dtype=cp.int32)

        # Vectorized calculation for all rotations
        bin_table = cp.unpackbits(cp.arange(2 ** samples, dtype=cp.uint8))  # Unpack bits
        # Reshape the result to 2D (each row represents a number, each column a bit)
        bin_table = bin_table.reshape(-1, 8)[:, :samples]  # Only keep the 'samples' bits
        rotated_bin_table = cp.roll(bin_table[:, :, None], -cp.arange(samples), axis=1)

        # Convert binary patterns to integers for each rotation (using the appropriate shape)
        powers_of_two = 1 << cp.arange(samples)  # Array of powers of 2
        rotated_integers = cp.array([
            cp.dot(rotated_bin_table[i].reshape(-1), powers_of_two) for i in range(2 ** samples)
        ])

        # Ensure rotated_integers is a 2D array with each row representing the rotation results
        rotated_integers = rotated_integers.reshape(2 ** samples, -1)

        # Now apply min across axis 1 (which is valid as rotated_integers is 2D)
        min_rotations = cp.min(rotated_integers, axis=1)

        # Assign the minimal rotation to the table and handle mappings
        unique_rotations, unique_idx = cp.unique(min_rotations, return_inverse=True)
        tmpMap[unique_rotations] = cp.arange(newMax, newMax + len(unique_rotations))
        newMax += len(unique_rotations)

        table = tmpMap[min_rotations]

    elif mappingtype == 'riu2':  # Uniform & Rotation invariant
        newMax = samples + 2
        # Vectorized computation for rotations and transitions
        bin_table = cp.unpackbits(cp.arange(2 ** samples, dtype=cp.uint8))  # Unpack bits
        # Reshape the result to 2D (each row represents a number, each column a bit)
        bin_table = bin_table.reshape(-1, 8)[:, :samples]  # Only keep the 'samples' bits
        shifted_bin_table = cp.roll(bin_table, -1, axis=1)

        # Count transitions for each pattern in parallel
        transitions = cp.sum(bin_table != shifted_bin_table, axis=1)
        uniform_mask = transitions <= 2
        table[uniform_mask] = cp.sum(bin_table[uniform_mask], axis=1)
        table[~uniform_mask] = samples + 1

    else:
        raise ValueError("Unsupported mapping type. Supported types: 'u2', 'ri', 'riu2'.")

    mapping = {'table': table, 'samples': samples, 'num': newMax}
    return mapping
