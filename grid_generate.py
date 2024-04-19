import numpy as np
import matplotlib.pyplot as plt
import downsample

def generate_MAC_velocity_grid(n, liquid_probability=0.5):
    """
    Generates a random 2D MAC grid for fluid simulation where velocities are stored on the edges.
    Parameters:
    n (int): The number of cells along one dimension of the grid.
    liquid_probability (float): Probability of a cell being liquid.
    Returns:
    tuple: A tuple containing the horizontal velocity grid u, vertical velocity grid v, and the cell type grid.
    """
    u = np.zeros((n, n+1))  # Horizontal velocities on vertical faces
    v = np.zeros((n+1, n))  # Vertical velocities on horizontal faces
    cell_type = np.full((n, n), 'solid', dtype='U10')

    for i in range(n):
        for j in range(n):
            if np.random.rand() <= liquid_probability:
                cell_type[i, j] = 'liquid'
                # u[i, j] = np.random.randn()
                # u[i, j+1] = np.random.randn()
                # v[i, j] = np.random.randn()
                # v[i+1, j] = np.random.randn()
                u[i, j] = 0
                u[i, j+1] = 0
                v[i, j] = 0
                v[i+1, j] = 0
    return u, v, cell_type


def calculate_divergence(u, v):
    """
    Calculates the divergence of the velocity field stored on the MAC grid.
    Parameters:
    u (numpy.ndarray): The horizontal velocity grid.
    v (numpy.ndarray): The vertical velocity grid.
    Returns:
    numpy.ndarray: The divergence field.
    """
    n, m = u.shape[0], v.shape[1]
    divergence = np.zeros((n, m))
    for i in range(1, n-1):
        for j in range(1, m-1):
            du_dx = (u[i, j+1] - u[i, j])
            dv_dy = (v[i+1, j] - v[i, j])
            divergence[i, j] = du_dx + dv_dy

    return divergence


def generate_fluid_grid(n, liquid_probability=0.5):
    """
    Generates a random 2D matrix simulating fluid and solid cells.
    The velocity divergence of each liquid cell will be calculated and stored.
    Parameters:
    n (int): Number of rows and columns in the matrix.
    liquid_probability (float): Probability that each cell is liquid, default is 0.5.
    Returns:
    numpy.ndarray: A two-dimensional array with a complex structure representing the fluid simulation grid.
    """
    dt = np.dtype([('type', 'U10'), ('divergence', float)])
    grid = np.zeros((n, n), dtype=dt)
    u, v, cell_type = generate_MAC_velocity_grid(n, liquid_probability)
    divergence = calculate_divergence(u, v)

    for i in range(n):
        for j in range(n):
            grid[i, j]['type'] = cell_type[i, j]
            if cell_type[i, j] == 'liquid':
                grid[i, j]['divergence'] = divergence[i, j]
            else:
                grid[i, j]['divergence'] = 0  # Solid cells have no divergence

    return grid


def plot_fluid_grid(grid):
    """
    Visualize the grid using matplotlib, where liquid cells are blue and solid cells are white.

    Parameters:
    grid (numpy.ndarray): The grid to be visualized, which contains 'type' and 'divergence' information.
    """
    n = grid.shape[0]
    color_map = np.zeros((n, n, 3))  # Initialize an RGB color map

    for i in range(n):
        for j in range(n):
            if grid[i, j]['type'] == 'liquid':
                color_map[i, j] = [0, 0, 1]  # Blue for liquid
            else:
                color_map[i, j] = [1, 1, 1]  # White for solid

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.imshow(color_map, interpolation='nearest')
    plt.title('Fluid Grid Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()


# Example: Generate a 10x10 grid with a liquid probability of 0.6
grid = generate_fluid_grid(32, 0.3)

# Print the generated grid
print(grid)
plot_fluid_grid(grid)
downsampled_grid = downsample.downsample_grid(grid)
plot_fluid_grid(downsampled_grid)