import numpy as np

def gauss_seidel_smooth(L, u, b, iterations=1):
    n = len(u)  
    for _ in range(iterations):
        for i in range(n):
            
            sum_Lu = np.dot(L[i].toarray(), u)  # 将 L[i] 转换为稠密数组再进行点积
            if isinstance(sum_Lu, np.ndarray) and sum_Lu.size == 1:
                sum_Lu = sum_Lu.item()  # 提取标量值
                u[i] += (b[i] - sum_Lu) / L[i, i]