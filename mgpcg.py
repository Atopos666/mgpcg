import numpy as np
import v_cycle_solver as solver

class TestMGPCG:
    def __init__(self, tolerance=1e-8):
        self.tolerance = tolerance
        self.max_iter = 1000
        self.matrix_size = 256

    def test_mgpcg(self):
        level = 5
        u = [np.zeros(self.matrix_size * self.matrix_size // (2 ** (2 * i))) for i in range(level)]
        b = [np.zeros(self.matrix_size * self.matrix_size // (2 ** (2 * i))) for i in range(level)]
        levels, b[0] = solver.generate_multigrid_matrices(level)
        raw_matrix = levels[0]
        x = u[0]
        r = b[0] 
        r = r - raw_matrix @ x
        mu = np.average(r)
        v = np.linalg.norm(r, np.inf)
        if v < self.tolerance:
            return x, 0
        r = r - mu
        p = solver.v_cycle(u, b, levels, solver.restrict, solver.prolongation)
        rho = np.dot(p, r)
        for i in range(self.max_iter):
            z = raw_matrix @ p
            alpha = rho / np.dot(p, z)
            r -= alpha * z
            mu = np.average(r)
            v = np.linalg.norm(r, np.inf)
            print(v)
            if v < self.tolerance or i == self.max_iter - 1:
                x = x + alpha * p
                return x, i
            r = r - mu
            b[0] = r
            z = solver.v_cycle(u, b, levels, solver.restrict, solver.prolongation)
            rho_new = np.dot(z, r)
            beta = np.dot(rho_new, r)
            beta = beta / rho
            rho = rho_new
            x = x + alpha * p
            p = z + beta * p
        return x, self.max_iter
    
def main():
    test = TestMGPCG()
    solution = test.test_mgpcg()[0]
    iter = test.test_mgpcg()[1]
    print(solution, iter)

if __name__ == "__main__":
    main()