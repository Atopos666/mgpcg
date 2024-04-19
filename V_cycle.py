import numpy as np
from scipy.sparse.linalg import spsolve

def multigrid_vcycle(level, A_list, R_list, b, x0, direct_n, PR_coef, smoother, pre_steps=1, pos_steps=1):
    """
    Multigrid V-cycle algorithm for solving A * x = b

    Parameters:
        level (int): Current multigrid level, starting with 1
        A_list (list of ndarray): Coefficient matrices on each level
        R_list (list of ndarray): Restriction operators on each level
        b (ndarray): Right-hand side of the equation
        x0 (ndarray): Initial guess
        direct_n (int): Threshold for directly solving A*x = b
        PR_coef (float): Coefficient constant between restriction and prolongation
        smoother (callable): Function handle for an iterative method as a smoother
        pre_steps (int): Number of iterations in the pre-smoothing (default 1)
        pos_steps (int): Number of iterations in the post-smoothing (default 1)
    Returns:
        ndarray: The solution vector x
    """
    # Load coefficient matrix
    A = A_list[level]

    # If the problem is small enough, solve it directly
    if b.size <= direct_n:
        return np.linalg.solve(A, b)

    # Pre-smoothing
    x = smoother(A, b, x0, pre_steps)

    # Load restriction operator and construct interpolation operator
    R = R_list[level]
    P = R.T * PR_coef

    # Compute residual and transfer to coarse grid
    r = b - A.dot(x)
    r_H = R.dot(r)

    # Solve coarse grid problem recursively
    x0_coarse = np.zeros(r_H.shape)
    if level + 1 < len(A_list):
        e_H = multigrid_vcycle(level + 1, A_list, R_list, r_H, x0_coarse, direct_n, PR_coef, smoother, pre_steps, pos_steps)
    else:
        e_H = spsolve(A_list[level + 1], r_H)

    # Transfer error to fine grid and correct
    x += P.dot(e_H)

    # Post-smoothing
    x = smoother(A, b, x, pos_steps)

    return x
