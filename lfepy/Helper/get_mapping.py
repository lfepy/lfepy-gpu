import cupy as cp


def get_mapping(samples, mappingtype):
    """
    Generate a mapping table for Local Binary Patterns (LBP) codes using raw kernels.

    Args and Returns are the same as the original function.
    """
    table = cp.arange(2 ** samples, dtype=cp.int32)
    newMax = 0  # Number of patterns in the resulting LBP code

    if mappingtype == 'u2':  # Uniform 2
        newMax = samples * (samples - 1) + 3

        # Raw kernel for uniform mapping
        uniform_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void uniform_mapping(int* table, int samples, int newMax) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int max_pattern = 1 << samples;
            if (i >= max_pattern) return;

            int num_transitions = 0;
            int prev = (i >> (samples - 1)) & 1;
            int first = prev;

            for (int j = 0; j < samples; j++) {
                int current = (i >> (samples - 1 - j)) & 1;
                if (current != prev) {
                    num_transitions++;
                }
                prev = current;
            }

            if (first != prev) {
                num_transitions++;
            }

            if (num_transitions <= 2) {
                // Count number of uniform patterns up to this point
                // This part is tricky in parallel - we'll do a prefix sum later
                table[i] = i;  // Temporary value
            } else {
                table[i] = newMax - 1;
            }
        }
        ''', 'uniform_mapping')

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
        threads_per_block = block_size
        grid_size = (2 ** samples + block_size - 1) // block_size
        uniform_kernel((grid_size,), (block_size,), (table, samples, newMax))

        # Second pass: assign sequential indices to uniform patterns
        # This replaces the prefix sum that would be needed for exact numbering
        # For exact numbering, we'd need a more complex approach with atomic operations
        is_uniform = (table != (newMax - 1))
        uniform_indices = cp.arange(cp.sum(is_uniform), dtype=cp.int32)
        table[is_uniform] = uniform_indices

    elif mappingtype == 'ri':  # Rotation invariant
        tmpMap = cp.full(2 ** samples, -1, dtype=cp.int32)

        # Raw kernel for rotation invariant mapping
        ri_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void rotation_invariant_mapping(int* table, int* tmpMap, int samples) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int max_pattern = 1 << samples;
            if (i >= max_pattern) return;

            int min_pattern = i;
            int current = i;

            for (int j = 1; j < samples; j++) {
                // Rotate left
                int msb = (current >> (samples - 1)) & 1;
                current = ((current << 1) | msb) & ((1 << samples) - 1);

                if (current < min_pattern) {
                    min_pattern = current;
                }
            }

            table[i] = min_pattern;

            // The unique mapping would need atomic operations or a separate pass
            // We'll handle that in Python after the kernel
        }
        ''', 'rotation_invariant_mapping')

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
        threads_per_block = block_size
        grid_size = (2 ** samples + block_size - 1) // block_size
        ri_kernel((grid_size,), (block_size,), (table, tmpMap, samples))

        # Post-processing in Python for the unique mapping
        min_rotations = table.copy()
        unique_rotations = cp.unique(min_rotations)
        newMax = len(unique_rotations)

        # Create mapping from unique rotations to sequential indices
        tmpMap[unique_rotations.get()] = cp.arange(newMax, dtype=cp.int32)
        table = tmpMap[min_rotations.get()]

    elif mappingtype == 'riu2':  # Uniform & Rotation invariant
        newMax = samples + 2

        # Raw kernel for uniform & rotation invariant mapping
        riu2_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void riu2_mapping(int* table, int samples, int newMax) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int max_pattern = 1 << samples;
            if (i >= max_pattern) return;

            // Check if uniform
            int num_transitions = 0;
            int prev = (i >> (samples - 1)) & 1;
            int first = prev;

            for (int j = 0; j < samples; j++) {
                int current = (i >> (samples - 1 - j)) & 1;
                if (current != prev) {
                    num_transitions++;
                }
                prev = current;
            }

            if (first != prev) {
                num_transitions++;
            }

            if (num_transitions <= 2) {
                // Count set bits
                int sum = 0;
                for (int j = 0; j < samples; j++) {
                    sum += (i >> j) & 1;
                }
                table[i] = sum;
            } else {
                table[i] = samples + 1;
            }
        }
        ''', 'riu2_mapping')

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
        threads_per_block = block_size
        grid_size = (2 ** samples + block_size - 1) // block_size
        riu2_kernel((grid_size,), (block_size,), (table, samples, newMax))

    else:
        raise ValueError("Unsupported mapping type. Supported types: 'u2', 'ri', 'riu2'.")

    mapping = {'table': table, 'samples': samples, 'num': newMax}
    return mapping