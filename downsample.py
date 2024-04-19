import numpy as np

def downsample_grid(grid):
    """
    Downsamples a given fluid grid by merging every four cells into one.
    If any of the four cells is liquid, the resulting cell is marked as liquid.
    Otherwise, it is marked as solid. The divergence is updated using a specific 
    weighted sum method for 4x4 neighborhood.

    Parameters:
    grid (numpy.ndarray): The grid to be downsampled, which contains 'type' and 'divergence' information.

    Returns:
    numpy.ndarray: The downsampled grid.
    """
    n = grid.shape[0] // 2
    dt = np.dtype([('type', 'U10'), ('divergence', float)])
    downsampled_grid = np.zeros((n, n), dtype=dt)

    # Define the 4x4 weights matrix for updating the divergence
    weights = np.array([
        [1/64, 3/64, 3/64, 1/64],
        [3/64, 9/64, 9/64, 3/64],
        [3/64, 9/64, 9/64, 3/64],
        [1/64, 3/64, 3/64, 1/64]
    ])

    for i in range(n):
        for j in range(n):
            subgrid = grid[2*i:2*i+2, 2*j:2*j+2]  # Extract the 2x2 block
            if np.any(subgrid['type'] == 'liquid'):
                downsampled_grid[i, j]['type'] = 'liquid'
            else:
                downsampled_grid[i, j]['type'] = 'solid'
            
            # Calculate the weighted sum for the divergence in the 4x4 neighborhood
            weighted_divergence_sum = 0
            for di in range(-1, 3):  # Adjusted for a 4x4 neighborhood
                for dj in range(-1, 3):
                    ni, nj = 2*i + di, 2*j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        weight_index_i = di + 1  # Adjust index to start from 0
                        weight_index_j = dj + 1
                        weighted_divergence_sum += weights[weight_index_i, weight_index_j] * grid[ni, nj]['divergence']
            downsampled_grid[i, j]['divergence'] = weighted_divergence_sum

    return downsampled_grid