import numpy as np
from numpy.linalg import norm, solve
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve  
import restrict
import prolongation
import grid_generate
import downsample
import generate_equation
import math
import jacobi
import gauss_seidel
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator


def generate_multigrid_matrices(coarse_level):
    gird = grid_generate.generate_fluid_grid(64, 0.3)
    A_list = []
    A0, b0 = generate_equation.setup_poisson_equation(gird)
    A_list.append(A0)

    for i in range(coarse_level-1):
        gird = downsample.downsample_grid(gird)
        A, b = generate_equation.setup_poisson_equation(gird)
        A_list.append(A)
    return A_list, b0

    




def v_cycle(u, b, levels, restrict, prolongation, solve_coarsest):
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
    print(L)
    # Downward cycle
    for l in range(L):
        gauss_seidel.gauss_seidel_smooth(L=levels[l], u=u[l],b=b[l], iterations=1)
        r = b[l] - levels[l] @ u[l]  # compute the residual
        b[l+1] = restrict.restrict(r,int(math.sqrt(len(r))))  # restrict residual to the next coarser level
        u[l+1] = np.zeros_like(b[l+1])  # initial guess for the next level
    
    # Solve at the coarsest level
    A_csr = levels[L].tocsr()
    # 检查矩阵是否为方阵
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError("The coefficient matrix A must be square!")
    # 使用 spsolve 函数求解 Ax = b
    u[L] = spsolve(A_csr, b[L])
    
    # Upward cycle
    for l in range(L-1, -1, -1):
        u[l] += prolongation.prolongate(u[l+1],int(math.sqrt(len(u[l]))))  # interpolate correction to the finer grid
        jacobi.damped_jacobi_smooth(L=levels[l], u=u[l],b=b[l],omega=2/3, iterations=1)
    
    print(u[0][0])
    return u[0]

# def Multigrid_Vcycle(level, A_list, R_list, b, x0, direct_n, PR_coef, smoother, pre_steps, pos_steps):
#     A = A_list[level]
#     if A.shape[0] <= direct_n or level == len(A_list) - 1:
#         x = solve(A, b)
#         return x
#     x = smoother(A, b, 1e-14, pre_steps, x0)[0]
#     R = R_list[level]
#     P = PR_coef * R.T
#     r = b - A.dot(x)
#     r_H = R.dot(r)
#     x0 = np.zeros(r_H.shape[0])
#     e_H = Multigrid_Vcycle(level + 1, A_list, R_list, r_H, x0, direct_n, PR_coef, smoother, pre_steps, pos_steps)
#     x += P.dot(e_H)
#     x = smoother(A, b, 1e-14, pos_steps, x)[0]
#     return x

level = 5
u = [np.zeros(64 * 64 // (2 ** (2 * i))) for i in range(level)]
b = [np.zeros(64 * 64 // (2 ** (2 * i))) for i in range(level)]
levels, b[0] = generate_multigrid_matrices(level)
solution = v_cycle(u, b, levels, restrict, prolongation, level)
print(solution)
print(len(solution))





