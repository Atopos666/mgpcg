import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def setup_poisson_equation(grid, h=1.0):
    n = grid.shape[0]
    A = lil_matrix((n**2, n**2))
    b = np.zeros(n**2)

    for i in range(n):
        for j in range(n):
            index = i * n + j
            if grid[i, j]['type'] == 'liquid':
                A[index, index] = -4
                if i > 0:
                    A[index, index - n] = 1
                if i < n - 1:
                    A[index, index + n] = 1
                if j > 0:
                    A[index, index - 1] = 1
                if j < n - 1:
                    A[index, index + 1] = 1
                b[index] = -h**2 * grid[i, j]['divergence']
            else:  # Apply Neumann boundary condition (simplest form)
                A[index, index] = 1
                b[index] = 0

    return A, b