# 1从零实现 svd
# 通过特征分解得到svd

# 一个 \(1000\times1000\) 的矩阵有 100 万个元素。只保留前 10 个奇异值对应的结构 20010 元素 = 1000*10 + 10 + 1000*10

# 构造 AtA  计算对应的特征向量
import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# Part 1: 使用 Jacobi 旋转法实现对称矩阵的特征值分解
# ============================================================

def jacobi_eigen(A, tol=1e-12, max_iter=100000):
    """
    对实对称矩阵 A 进行特征值分解。

    返回：
        eigenvalues: 特征值
        V: 特征向量矩阵，列向量为特征向量

    满足：
        A @ V ≈ V @ diag(eigenvalues)

    不使用 np.linalg.eigh / np.linalg.eig。
    """

    A = np.array(A, dtype=float, copy=True)

    n = A.shape[0]

    assert A.shape == (n, n), "输入必须是方阵"

    assert np.allclose(A, A.T), "输入必须是对称矩阵"

    # 初始化特征向量矩阵为单位矩阵
    V = np.eye(n)

    for iteration in range(max_iter):

        # 找到非对角线上绝对值最大的元素
        off_diag = np.abs(np.triu(A, k=1))

        p, q = np.unravel_index(
            np.argmax(off_diag),
            off_diag.shape
        )

        # 所有非对角元素足够小，说明矩阵接近对角化
        if off_diag[p, q] < tol:
            break

        # 计算 Jacobi 旋转参数
        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        tau = (aqq - app) / (2 * apq)

        if tau == 0:
            t = 1.0
        else:
            t = np.sign(tau) / (
                abs(tau) + np.sqrt(1 + tau**2)
            )

        c = 1 / np.sqrt(1 + t**2)
        s = t * c

        # 更新对角元素
        A[p, p] = app - t * apq
        A[q, q] = aqq + t * apq

        # 消除非对角元素
        A[p, q] = 0
        A[q, p] = 0

        # 更新其他行列
        for r in range(n):

            if r == p or r == q:
                continue

            arp = A[r, p]
            arq = A[r, q]

            A[r, p] = c * arp - s * arq
            A[p, r] = A[r, p]

            A[r, q] = s * arp + c * arq
            A[q, r] = A[r, q]

        # 累积特征向量
        vp = V[:, p].copy()
        vq = V[:, q].copy()

        V[:, p] = c * vp - s * vq
        V[:, q] = s * vp + c * vq

    else:
        raise RuntimeError("Jacobi 算法未在最大迭代次数内收敛")

    eigenvalues = np.diag(A).copy()

    # 按特征值从大到小排序
    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]

    V = V[:, indices]

    return eigenvalues, V


# ============================================================
# Part 2: 通过 A.T @ A 实现 SVD
# ============================================================

def svd_from_scratch(A):
    """
    从零实现紧凑 SVD：

        1. B = A.T @ A
        2. B 的特征值分解
        3. sigma = sqrt(lambda)
        4. U = A @ V / sigma

    返回：
        U, S, Vt

    这里只返回数值上非零的奇异值及对应向量。
    """

    A = np.asarray(A, dtype=float)

    # Step 1: 构造对称矩阵
    B = A.T @ A

    # Step 2: 自己实现特征值分解
    eigenvalues, V = jacobi_eigen(B)

    # Step 3: 特征值转换成奇异值
    eigenvalues = np.maximum(eigenvalues, 0)

    S = np.sqrt(eigenvalues)

    # Step 4: 排除接近 0 的奇异值
    # 避免除以 0
    threshold = S[0] * 1e-10 if len(S) else 0

    mask = S > threshold

    S = S[mask]
    V = V[:, mask]

    # Step 5: 计算左奇异向量
    U = (A @ V) / S

    return U, S, V.T


# ============================================================
# Part 3: 幂迭代版本 SVD
# ============================================================

def power_iteration_svd(A, max_iter=2000, tol=1e-12):
    """
    使用幂迭代 + Deflation 实现 SVD。

    该实现用于与 Jacobi 版本比较，不属于实验要求的
    非幂迭代主实现。
    """

    A = np.asarray(A, dtype=float)

    B = A.copy()

    m, n = A.shape

    rank = min(m, n)

    rng = np.random.default_rng(123)

    U_list = []
    S_list = []
    V_list = []

    for i in range(rank):

        # 随机初始化右奇异向量
        v = rng.normal(size=n)

        v /= np.linalg.norm(v)

        # 幂迭代
        for _ in range(max_iter):

            new_v = B.T @ (B @ v)

            norm = np.linalg.norm(new_v)

            if norm < 1e-12:
                break

            new_v /= norm

            if np.linalg.norm(new_v - v) < tol:
                v = new_v
                break

            v = new_v

        # 计算奇异值
        sigma = np.linalg.norm(B @ v)

        if sigma < 1e-10:
            break

        # 计算左奇异向量
        u = B @ v / sigma

        U_list.append(u)
        S_list.append(sigma)
        V_list.append(v)

        # Deflation：减去已经提取的秩 1 分量
        B = B - sigma * np.outer(u, v)

    return (
        np.column_stack(U_list),
        np.array(S_list),
        np.vstack(V_list)
    )


# ============================================================
# Part 4: 数值精度测试
# ============================================================

def evaluate_svd(A, U, S, Vt):
    """评估 SVD 分解精度。"""

    # 重建矩阵
    A_reconstructed = (U * S) @ Vt

    # 相对重建误差
    reconstruction_error = (
        np.linalg.norm(A - A_reconstructed)
        / np.linalg.norm(A)
    )

    # U 正交性误差
    U_error = np.linalg.norm(
        U.T @ U - np.eye(U.shape[1])
    )

    # V 正交性误差
    V_error = np.linalg.norm(
        Vt @ Vt.T - np.eye(Vt.shape[0])
    )

    return reconstruction_error, U_error, V_error


def main():

    rng = np.random.default_rng(42)

    # 随机生成一个 20x12 矩阵
    A = rng.normal(size=(20, 12))

    print("原始矩阵形状:", A.shape)

    # 方法 1：Jacobi 特征值分解
    U1, S1, Vt1 = svd_from_scratch(A)

    # 方法 2：幂迭代
    U2, S2, Vt2 = power_iteration_svd(A)

    # 方法 3：NumPy
    U3, S3, Vt3 = np.linalg.svd(
        A,
        full_matrices=False
    )

    methods = {
        "Jacobi SVD": (U1, S1, Vt1),
        "Power SVD": (U2, S2, Vt2),
        "NumPy SVD": (U3, S3, Vt3)
    }

    print("\n数值精度比较")
    print("-" * 95)

    print(
        f"{'Method':<15}"
        f"{'Reconstruction':<20}"
        f"{'Singular Value':<20}"
        f"{'U Orthogonal':<20}"
        f"{'V Orthogonal':<20}"
    )

    for name, (U, S, Vt) in methods.items():

        reconstruction, u_err, v_err = evaluate_svd(
            A, U, S, Vt
        )

        singular_error = (
            np.linalg.norm(S - S3[:len(S)])
            / np.linalg.norm(S3)
        )

        print(
            f"{name:<15}"
            f"{reconstruction:<20.3e}"
            f"{singular_error:<20.3e}"
            f"{u_err:<20.3e}"
            f"{v_err:<20.3e}"
        )

    print("\nNumPy 奇异值:")
    print(S3)

    print("\nJacobi 奇异值:")
    print(S1)


if __name__ == "__main__":
    main()



# 2 图像压缩，为什么少量奇异值就能保留图片


# 3 电影推荐  如何预测用户没有评分的电影



# 4 文档主题聚类  如何从大量词汇中发现隐藏主题


# 5 矩阵去噪   为什么舍弃部分奇异值反而效果更好