import numpy as np
import matplotlib.pyplot as plt

from skimage import data, transform


# ============================================================
# Part 1: 加载真实灰度图像
# ============================================================

def load_image():

    # skimage 自带的真实灰度照片
    image = data.camera()

    # 转换成 256x256
    image = transform.resize(
        image,
        (256, 256),
        anti_aliasing=True
    )

    return image


# ============================================================
# Part 2: SVD 压缩
# ============================================================

def compress_image(image, k, U, S, Vt):
    """
    通过截断 SVD 压缩图像。

    A_k = U_k @ Sigma_k @ V_k.T
    """
# k 表示截断的秩 压缩就是 u 前 k 列，v 前 k 行
    U_k = U[:, :k]

    S_k = S[:k]

    Vt_k = Vt[:k, :]

    compressed = (U_k * S_k) @ Vt_k

    return compressed


# ============================================================
# Part 3: 计算压缩比
# ============================================================

def calculate_compression_ratio(image, k):

    m, n = image.shape

    # 原始矩阵需要存储的元素数量
    original_size = m * n

    # 截断 SVD 存储 U、S、Vt 的元素数量
    compressed_size = m * k + k + k * n

    return original_size / compressed_size


# ============================================================
# Part 4: 计算相对误差
# ============================================================

def relative_error(original, reconstructed):

    return (
        np.linalg.norm(original - reconstructed)
        / np.linalg.norm(original)
    )


# ============================================================
# Part 5: 主程序
# ============================================================

def main():

    image = load_image()

    print("图像大小:", image.shape)

    # 计算完整 SVD
    U, S, Vt = np.linalg.svd(
        image,
        full_matrices=False
    )

    ranks = [1, 5, 10, 25, 50, 100]

    errors = []
    ratios = []

    # 创建可视化画布
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 8)
    )

    axes = axes.flatten()

    # 显示原图
    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Original Image")

    print("\nSVD Image Compression")
    print("-" * 50)

    print(
        f"{'Rank':<10}"
        f"{'Compression':<15}"
        f"{'Relative Error':<20}"
    )

    for i, k in enumerate(ranks):

        compressed = compress_image(
            image, k, U, S, Vt
        )

        # 计算压缩比
        ratio = calculate_compression_ratio(
            image, k
        )

        # 计算误差
        error = relative_error(
            image, compressed
        )

        errors.append(error)
        ratios.append(ratio)

        print(
            f"{k:<10}"
            f"{ratio:<15.3f}"
            f"{error:<20.5f}"
        )

        # 显示压缩结果
        axes[i + 1].imshow(
            compressed,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i + 1].set_title(
            f"Rank={k}\nError={error:.4f}"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        "experiment_02_comparison.png",
        dpi=200
    )

    plt.show()

    # ========================================================
    # 绘制 Rank - Error 曲线
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        ranks,
        errors,
        marker="o"
    )

    plt.xlabel("Truncated Rank k")
    plt.ylabel("Relative Frobenius Error")

    plt.title("SVD Compression Error")

    plt.grid(True)

    plt.savefig(
        "experiment_02_error_curve.png",
        dpi=200
    )

    plt.show()

    # ========================================================
    # 依据误差阈值给出参考阶数
    # ========================================================

    threshold = 0.05

    acceptable = [
        k for k, error in zip(ranks, errors)
        if error <= threshold
    ]

    if acceptable:
        print(
            f"\n相对误差不超过 {threshold:.0%} "
            f"的最小测试阶数: {min(acceptable)}"
        )
    else:
        print("\n测试的阶数均未达到误差阈值")


if __name__ == "__main__":
    main()