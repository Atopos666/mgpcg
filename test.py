import numpy as np

def trivial_test_case(n):
    """
    Generates a trivial test case for the Poisson solver with known solution.
    Parameters:
    n (int): Grid size (n x n).
    Returns:
    numpy.ndarray: Structured array grid with divergence set to zero and all cells as liquid.
    """
    dt = np.dtype([('type', 'U10'), ('divergence', float)])
    grid = np.zeros((n, n), dtype=dt)
    u_exact = np.ones((n, n))  # Known solution

    for i in range(n):
        for j in range(n):
            grid[i, j]['type'] = 'liquid'
            grid[i, j]['divergence'] = 0.0  # Set divergence to zero

    return grid, u_exact


import numpy as np
import matplotlib.pyplot as plt


def add_velocity_divergence(grid, liquid_cells):
    """
    Adds a random velocity divergence to the liquid cells in the grid.

    Parameters:
    grid (numpy.ndarray): The grid where the liquid cells are defined.
    liquid_cells (list of tuples): List of coordinates for the liquid cells.

    Returns:
    numpy.ndarray: The grid with the velocity divergence added to the liquid cells.
    """
    # Define a new structured data type for the grid with an additional 'divergence' field
    dt = np.dtype([('type', 'U10'), ('divergence', float)])
    grid_with_divergence = np.zeros(grid.shape, dtype=dt)

    # Copy the 'type' from the original grid and initialize divergence
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid_with_divergence[i, j]['type'] = grid[i, j]
            grid_with_divergence[i, j]['divergence'] = 0.0  # Initialize divergence to zero

    # Add random divergence to liquid cells
    for cell in liquid_cells:
        grid_with_divergence[cell]['divergence'] = np.random.randn()

    return grid_with_divergence


# Assume 'grid' is the original grid and 'liquid_cells' are the liquid cell coordinates from previous code
# Initialize the grid with 'solid' for all cells
grid_size = (15, 15)
grid = np.full(grid_size, 'solid', dtype='<U10')

# Define the liquid cells based on the shape provided earlier
liquid_cells = [
    (5, 4), (5, 5), (5, 6), (5, 7),
    (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
    (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
    (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
    (9, 3), (9, 4), (9, 5), (9, 6), (9, 7),
    (10, 4), (10, 5), (10, 6), (10, 7),
    (11, 5), (11, 6)
]

# Set the liquid cells in the original grid
for cell in liquid_cells:
    grid[cell] = 'liquid'

# Now add velocity divergence to the grid
# 水滴
grid_with_velocity_divergence = add_velocity_divergence(grid, liquid_cells)
print(grid_with_velocity_divergence)

# trivial测试
n = 10
grid, u_exact = trivial_test_case(n)