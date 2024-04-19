import numpy as np

def damped_jacobi_smooth(L, u, b, omega=2/3, iterations=1):
    n = len(u)  
    for _ in range(iterations):
        u_new = np.copy(u)  # 创建一个新的u数组，用于存储更新
        for i in range(n):
            sum_Lu = np.dot(L[i].toarray(), u)  # 将 L[i] 转换为稠密数组再进行点积
            if isinstance(sum_Lu, np.ndarray) and sum_Lu.size == 1:
                sum_Lu = sum_Lu.item()  # 提取标量值
            u_new[i] = u[i] + omega * (b[i] - sum_Lu) / L[i, i]
        u[:] = u_new  