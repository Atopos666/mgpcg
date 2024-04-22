import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt


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
                u[i, j] = np.random.randn()
                u[i, j+1] = np.random.randn()
                v[i, j] = np.random.randn()
                v[i+1, j] = np.random.randn()

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
                divergence[i, j] = 0

    return grid, divergence


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


def solve_pressure(divergence, n, max_iter=10000, tol=1e-6):
    p = np.zeros((n, n))  # Initial pressure field
    dx = 1.0  # Assuming grid spacing is 1 for simplicity

    for _ in range(max_iter):
        p_old = p.copy()

        # Update pressure based on the finite difference approximation of Laplacian
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p[i, j] = 0.25 * (p_old[i + 1, j] + p_old[i - 1, j] + p_old[i, j + 1] + p_old[i, j - 1] - dx ** 2 *
                                  divergence[i, j])

        # Check for convergence
        if np.linalg.norm(p - p_old, ord=np.inf) < tol:
            break

    return p

def showsol(sol):
    plt.imshow(sol.T,cmap=ml.cm.Blues,interpolation='none',origin='lower')

def solve_pressure_visual(divergence, n, max_iter=10000, tol=1e-6):
    p = np.zeros((n, n))  # Initial pressure field
    dx = 1.0  # Assuming grid spacing is 1 for simplicity

    plt.figure(figsize=(12, 12))  # 设置绘图窗口大小

    for iter_count in range(max_iter):
        p_old = p.copy()

        # Update pressure
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p[i, j] = 0.25 * (p_old[i + 1, j] + p_old[i - 1, j] +
                                  p_old[i, j + 1] + p_old[i, j - 1] -
                                  dx ** 2 * divergence[i, j])

        # Visualize at specific iterations
        if iter_count % 100 == 0:
            plt.subplot(4, 4, int(iter_count / 100 + 1))
            showsol(p)
            plt.title('Iter = %s' % iter_count)

        # Check for convergence
        if np.linalg.norm(p - p_old, ord=np.inf) < tol:
            break

    plt.tight_layout()
    plt.show()

    return p


# generate sample
n = 100
grid, divergence = generate_fluid_grid(n, 0.3)

# Print the generated grid
plot_fluid_grid(grid)

# p = solve_pressure(divergence, n)
p = pressure_field = solve_pressure_visual(divergence, n, max_iter=1500, tol=1e-6)
print(p)