import numpy as np

def restrict(x, N):
    """
    Restricts a solution vector from a fine grid (N x N) to a coarse grid (N/2 x N/2),
    using a specified 4x4 weighting matrix.
    
    Parameters:
    x (np.ndarray): The fine grid solution vector of length N*N.
    N (int): The number of rows/columns in the fine grid.
    
    Returns:
    np.ndarray: The restricted solution vector of length (N/2)*(N/2).
    """
    assert N % 2 == 0, "N must be even."
    
    # Reshape the flat array to a 2D array for easier manipulation
    fine = x.reshape(N, N)
    coarse = np.zeros((N//2, N//2))

    # Define the 4x4 weights matrix
    weights = np.array([
        [1/64, 3/64, 3/64, 1/64],
        [3/64, 9/64, 9/64, 3/64],
        [3/64, 9/64, 9/64, 3/64],
        [1/64, 3/64, 3/64, 1/64]
    ])
    
    # Apply the 4x4 weighted average to restrict the fine grid to the coarse grid
    for i in range(N//2):
        for j in range(N//2):
            weighted_sum = 0
            for di in range(-1, 3):
                for dj in range(-1, 3):
                    ni = 2 * i + di
                    nj = 2 * j + dj
                    if 0 <= ni < N and 0 <= nj < N:
                        weight = weights[di + 1, dj + 1]
                        weighted_sum += weight * fine[ni, nj]
            coarse[i, j] = weighted_sum
    
    # Flatten the coarse grid array to a 1D array before returning
    return coarse.flatten()


