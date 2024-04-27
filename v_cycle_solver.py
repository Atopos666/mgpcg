import numpy as np
from scipy.sparse.linalg import spsolve  
import restrict
import prolongation
import grid_generate
import downsample
import generate_equation
import math
import jacobi
import gauss_seidel




def generate_multigrid_matrices(coarse_level):
    gird = grid_generate.generate_fluid_grid(128, 0.3)
    A_list = []
    A0, b0 = generate_equation.setup_poisson_equation(gird)
    A_list.append(A0)

    for i in range(coarse_level-1):
        gird = downsample.downsample_grid(gird)
        A, b = generate_equation.setup_poisson_equation(gird)
        A_list.append(A)
    return A_list, b0

    




def v_cycle(u, b, levels, restrict, prolongation):
    """
    V-Cycle Multigrid method to solve linear equations.
    
    Parameters:
    u (list): Initial guesses at each level, typically zeros at the start.
    b (list): Right-hand side (RHS) of the equations at each level.
    levels (list): List of matrix operators at each level.
    smoother (function): Function to apply smoothing, e.g., Gauss-Seidel.
    restrict (function): Function to restrict residual to a coarser grid.
    prolongate (function): Function to interpolate correction to a finer grid.
    solve_coarsest (function): Function to directly solve the equation at the coarsest level.
    
    Returns:
    list: Updated solutions at each level after one V-cycle.
    """
    L = len(levels) - 1  # index of the last level
    # Downward cycle
    for l in range(L):
        gauss_seidel.gauss_seidel_smooth(L=levels[l], u=u[l],b=b[l], iterations=10)
        r = b[l] - levels[l] @ u[l]  # compute the residual
        b[l+1] = restrict.restrict(r,int(math.sqrt(len(r))))  # restrict residual to the next coarser level
        u[l+1] = np.zeros_like(b[l+1])  # initial guess for the next level
    
    # # Solve at the coarsest level
    # A_csr = levels[L].tocsr()
    # # 检查矩阵是否为方阵
    # if A_csr.shape[0] != A_csr.shape[1]:
    #     raise ValueError("The coefficient matrix A must be square!")
    # # 使用 spsolve 函数求解 Ax = b
    # u[L] = spsolve(A_csr, b[L])

    jacobi.damped_jacobi_smooth(L=levels[L], u=u[L],b=b[L],omega=2/3, iterations=10)
    
    # Upward cycle
    for l in range(L-1, -1, -1):
        u[l] += prolongation.prolongate(u[l+1],int(math.sqrt(len(u[l]))))  # interpolate correction to the finer grid
        jacobi.damped_jacobi_smooth(L=levels[l], u=u[l],b=b[l],omega=2/3, iterations=10)

    return u[0]







