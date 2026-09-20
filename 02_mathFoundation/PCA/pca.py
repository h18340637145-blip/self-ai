import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components #  最终我要保留几个主成分。
        self.components = None # 
        self.mean = None # 数据的均值，用于中心化。
        self.eigenvalues = None # 特征值，用于排序主成分。
        self.explained_variance_ratio = None # 每个主成分解释的方差比例。

    # 拟合数据，计算主成分  拟合数据就是根据输入的数据计算出主成分和相关信息 属于一种预处理
    def fit(self, X):
        self.mean = np.mean(X, axis=0) # 计算每一列的均值
        X_centered = X - self.mean # 数据中心化

        cov_matrix = np.cov(X_centered, rowvar=False) # 计算协方差矩阵

        eigenvalues,eigenvectors = np.linalg.eigh(cov_matrix) # 计算协方差矩阵的特征值和特征向量

        # 默认是从小到大排序    [::-1] 把顺序整个反过来
        sorted_idx = np.argsort(eigenvalues)[::-1] # 按特征值从大到小排序

        eigenvalues = eigenvalues[sorted_idx] # 按特征值排序后的特征值

        eigenvectors = eigenvectors[:, sorted_idx] # 按特征值排序后的特征向量

        """
        eigenvectors:
        [特征向量1, 特征向量2, ..., 特征向量n]
        特征向量 1=[特征向量1的各个分量]
        特征向量 2=[特征向量2的各个分量]
        ...
        特征向量 n=[特征向量n的各个分量]

        前 n 个就是取前 n 个特征向量，即主成分。
        保留向量的维度，取前 n_components 个特征向量。 ：n  从 0 到 n_components-1
        """
        self.components = eigenvectors[:, :self.n_components] # 取前 n_components 个特征向量 
        self.eigenvalues = eigenvalues[:self.n_components] # 取前 n_components 个特征值 
        total_var = np.sum(eigenvalues) # 总方差
        # 计算每个主成分解释的方差比例 
        # 总方差是所有特征值的和
        # 每个主成分解释的方差比例 = 对应特征值 / 总方差
        self.explained_variance_ratio = self.eigenvalues / total_var # 每个主成分解释的方差比例

        return self

    # 将数据投影到主成分空间
    def transform(self, X):
        X_centered = X - self.mean # 数据中心化
        # 结果就是投影到主成分空间后的数据 也就是降维 
        return np.dot(X_centered, self.components) # 投影到主成分空间 对应公式：X' = (X - mean) * components

    # 将投影后的数据还原回原始空间
    def inverse_transform(self, X_reduced):
        # 根据低维度数据还原回高维度 原始空间
        X_reconstructed = np.dot(X_reduced, self.components.T) + self.mean # 还原回原始空间
        return X_reconstructed


    # 拟合数据并返回投影到主成分空间后的数据
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) # 先拟合数据，再将数据投影到主成分空间 
        

# 在合成数据上进行测试
np.random.seed(0)
n_samples = 500

# 生成合成数据的参数
t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)
# x1 x2 x3 三个就是生成的合成数据的三个特征
# 每个特征有 n_samples 个数据点 也就是样本 
# X_synthetic 的形状是 (n_samples, 3) 也就是 (样本数, 特征数)
X_synthetic = np.column_stack([x1, x2, x3])

print(f"合成数据的形状: {X_synthetic.shape}")
print(f"合成数据的前5行:\n{X_synthetic[:5]}")

# 使用 pca 降维
pca = PCA(n_components=2) # 保留两个主成分 默认调用 init
X_reduced = pca.fit_transform(X_synthetic) # 先拟合数据 预处理，再将数据投影到主成分空间
print(f"降维后的数据形状: {X_reduced.shape}")
print(f"降维后的数据前5行:\n{X_reduced[:5]}")

# 二维mnist数字】
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X_mnist = mnist.data[:5000].astype(float) # 取前5000个样本并转换为浮点型 [0:5000] [:5000]是一样的意思 
y_mnist = mnist.target[:5000] # 取前5000个样本的标签ß

pca_mnist = PCA(n_components=50) # 保留 50 个特征 主成分
X_pca = pca_mnist.fit_transform(X_mnist) # 先拟合数据，再将数据投影到主成分空间
print(f"降维后的 MNIST 数据形状: {X_pca.shape}")
print(f"降维后的 MNIST 数据前5行:\n{X_pca[:5]}")


# 与sklearn比较
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=50) # 保留 50 个主成分
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist) # 先拟合数据，再将数据投影到主成分空间
print(f"降维后的 sklearn PCA MNIST 数据形状: {X_sklearn_pca.shape}")
print(f"降维后的 sklearn PCA MNIST 数据前5行:\n{X_sklearn_pca[:5]}")

diff = np.abs(np.abs(X_pca) - np.abs(X_sklearn_pca)) # 计算自实现 PCA 与 sklearn PCA 的差异
print(f"自实现 PCA 与 sklearn PCA 的差异前5行:\n{diff[:5]}")

tsne = TSNE(n_components=2, random_state=0) # 将数据降到二维
X_tsne = tsne.fit_transform(X_mnist) # 将 MNIST 数据降到二维
print(f"降维后的 t-SNE MNIST 数据形状: {X_tsne.shape}")
print(f"降维后的 t-SNE MNIST 数据前5行:\n{X_tsne[:5]}")


# umap 比较
from umap import UMAP

reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42) # 将数据降到二维
X_umap = reducer.fit_transform(X_mnist) # 将 MNIST 数据降到二维
print(f"降维后的 UMAP MNIST 数据形状: {X_umap.shape}")
print(f"降维后的 UMAP MNIST 数据前5行:\n{X_umap[:5]}")


# pca 作为分类器之前的预处理步骤：
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42
)

# results = {}
# for k in [10, 30, 50, 100, 200]:  # 添加 200 作为主成分的选择
#     pca_k = SklearnPCA(n_components=k) # 保留 k 个主成分，依次尝试不同的 k 值
#     X_train_pca = pca_k.fit_transform(X_train)
#     X_test_pca = pca_k.transform(X_test)

#     clf = LogisticRegression(max_iter=1000, random_state=42) # 使用逻辑回归分类器
#     clf.fit(X_train_pca, y_train)
#     y_pred = clf.predict(X_test_pca)
#     acc = accuracy_score(y_test, y_pred)
#     results[k] = acc

# # 输出不同主成分数下的测试集准确率
# for k, acc in results.items():
#     print(f"主成分数: {k}, 测试集准确率: {acc:.4f}")

'''
主成分数: 10, 测试集准确率: 0.8160
主成分数: 30, 测试集准确率: 0.8940
主成分数: 50, 测试集准确率: 0.9000
主成分数: 100, 测试集准确率: 0.8580
主成分数: 200, 测试集准确率: 0.8700

50个主成分时，测试集准确率最高。
'''

# test1 
result = {}
for k in [10, 50, 200]:  # 添加 200 作为主成分的选择
    pca_k = SklearnPCA(n_components=k) # 保留 k 个主成分
    X_train_pca = pca_k.fit_transform(X_train)
    X_test_pca = pca_k.transform(X_test)

    X_reconstructed_train = pca_k.inverse_transform(X_train_pca)
    X_reconstructed_test = pca_k.inverse_transform(X_test_pca)

    # 计算重建误差
    train_error = np.mean((X_train - X_reconstructed_train) ** 2)
    test_error = np.mean((X_test - X_reconstructed_test) ** 2)
    result[k] = (train_error, test_error)

for k, (train_error, test_error) in result.items():
    print(f"主成分数: {k}, 训练集重建误差: {train_error:.4f}, 测试集重建误差: {test_error:.4f}")

'''
主成分数: 10, 训练集重建误差: 2198.6139, 测试集重建误差: 2272.5729
主成分数: 50, 训练集重建误差: 742.3631, 测试集重建误差: 776.1702
主成分数: 200, 训练集重建误差: 134.9456, 测试集重建误差: 165.6930

每个像素是 0-255 之间的灰度值
如果原始和重建之后差值是 50，那么平方后就是 2500。
所以重建误差的数量级是可以理解的。
'''


# test2
for k in [5, 30, 100]:  # t-sne 算法 困惑度值分别设置为 5, 30, 100
    tsne = TSNE(n_components=2, perplexity=k, random_state=42)
    X_tsne = tsne.fit_transform(X_mnist)
    print(f"困惑度: {k}, 降维后的 t-SNE MNIST 数据形状: {X_tsne.shape}")
    print(f"困惑度: {k}, 降维后的 t-SNE MNIST 数据前5行:\n{X_tsne[:5]}")


# test3 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X, y = make_classification(
    n_samples=2000,
    n_features=50, # 50个特征
    n_informative=5, # 5个特征对分类有意义
    n_redundant=0, # 不生成冗余特征
    n_repeated=0,
    random_state=42
)

print("XShape:", X.shape)

# 使用 PCA 对数据进行降维
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
print("降维后的 XShape:", X_pca.shape)

# 每个主成分解释的方差比例
explained_var_ratio = pca.explained_variance_ratio_
print("每个主成分解释的方差比例:", explained_var_ratio)

# 累积解释的方差比例
cumulative_var = np.cumsum(explained_var_ratio)
print("累积解释的方差比例:", cumulative_var)

# 可视化累积解释的方差比例
plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), cumulative_var, marker='o', linestyle='--')
plt.xlabel('主成分数')
plt.ylabel('累积解释的方差比例')
plt.title('PCA 累积解释的方差比例')
plt.grid(True)
plt.show()