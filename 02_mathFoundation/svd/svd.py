
# svd 奇异值分解 适用于任何矩阵  
# 不同于特征值分解只能用于方阵，SVD 可以用于任意矩阵ß

# 任何一个mn矩阵 都可以转变成 m*m   m * n  n * n 的奇异值分解形式 对应着旋转缩放再旋转的操作

# 奇异值就是各个轴的长度  Sigma 的各个值
# A = U * Sigma * V^T
# 矩阵 U 的列向量称为左奇异向量。           矩阵 V 的列向量称为右奇异向量 

# 使用幂迭代从头开始进行svd分解
# 为了找到最大奇异值及其向量，对A^TA（或 AA^T）进行幂迭代 
# 然后对矩阵进行降维，并对下一个奇异值及其向量重复该过程

import numpy as np 

'''
A
↓
AᵀA
↓
找到最大特征值 λ₁、特征向量 v₁
↓
σ₁ = √λ₁
↓
u₁ = Av₁ / σ₁
↓
A ≈ σ₁u₁v₁ᵀ + ...
↓
从 A 中减掉第一部分
↓
继续找 σ₂、u₂、v₂
'''

# 矩阵 M 的最大特征值和对应的特征向量。 M 应该是方阵 = AᵀA
def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n) # 不知道最大特征向量到底在哪，所以先扔一个随机向量进去。 随机初始化
    v = v/np.linalg.norm(v) # 归一化，使其长度为1
    for _ in range(num_iters): # 进行幂迭代，逐步逼近最大特征向量
        """
        假设M 有三个特征向量  对应特征值：λ₁ > λ₂ > λ₃  
        随机向量 v = c_1q_1+c_2q_2+c_3q_3  
        两边同时乘以 M
        Mv = Mv = c_1λ₁q_1 + c_2λ₂q_2 + c_3λ₃q_3
        由于 λ₁ > λ₂ > λ₃，经过多次迭代后，c_1λ₁^kq_1 项会占主导，v 会逐渐逼近 q_1
        (λ 的k次方增长会特别快，所以会占主导地位)
        经过多次迭代后，v 会逐渐逼近 q_1，即最大特征值对应的特征向量
        """
        Mv = M @ v  
        v = Mv / np.linalg.norm(Mv) # 
    eigenvalue = v @ M @ v
    return eigenvalue, v

# 使用幂迭代进行 SVD 分解
def svd_from_scratch(A, k = None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):

        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA)

        if eigenvalue <= 1e-10:
            break

        # 计算当前的奇异值和对应的左奇异向量
        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual -= sigma * np.outer(u, v)
    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.diag(sigmas) if sigmas else np.empty((0, 0))
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V

# 测试并与np比较
np.random.seed(42)

A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, V_np = np.linalg.svd(A, full_matrices=False)

print("Our SVD:")
print("U:\n", U_ours)
print("S:\n", S_ours)
print("V:\n", V_ours)

print("\nNumPy SVD:")
print("U:\n", U_np)
print("S:\n", S_np)
print("V:\n", V_np)

# 图像压缩演示
# 使用 SVD 对图像进行压缩
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    # 保留前 k 列， 前 k 行 的奇异向量和奇异值
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed_image = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed_image) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1) 
    ratio = compressed_size / original_size
    print(f"k={k}, error={error:.4f}, compression ratio={ratio:.4f}")

# 降噪
np.random.seed(42)
# 构造干净的信号矩阵（没有噪声）
clean = np.outer(np.sin(np.linspace(0, 4 * np.pi, 100)), 
                 np.sin(np.linspace(0, 2 * np.pi, 80)))

noise = 0.3 * np.random.randn(*clean.shape)
noisy = clean + noise

# 使用 SVD 对噪声矩阵进行降噪
U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
k = 5  # 保留前 k 个奇异值
S_denoised = np.zeros_like(S)
S_denoised[:k] = S[:k]
denoised = U @ np.diag(S_denoised) @ Vt

# 显示降噪效果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Clean Signal")
plt.imshow(clean, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Noisy Signal")
plt.imshow(noisy, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Denoised Signal")
plt.imshow(denoised, cmap='gray')
plt.show()

# 伪逆
A = np.array([[1,1], [2, 1], [3, 1]])
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

# 计算伪逆
S_inv = np.diag(1.0 / S) # 计算奇异值的倒数，构造伪逆的对角矩阵
A_pinv = Vt.T @ S_inv @ U.T  # 构造伪逆矩阵

x_svd = A_pinv @ b  # 使用伪逆求解最小二乘问题
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]  # 使用最小二乘法求解最小二乘问题

x_pinv = np.linalg.pinv(A) @ b  # 使用 NumPy 的伪逆求解最小二乘问题

print("x_svd:", x_svd)
print("x_lstsq:", x_lstsq)
print("x_pinv:", x_pinv)