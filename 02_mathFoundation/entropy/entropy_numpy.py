import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    reulst = np.zeros_like(p)
    reulst[mask] = p[mask] * np.log(p[mask])
    return -reulst.sum()


def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = (p > 0)
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print("Entropy of true distribution:", np_entropy(true))
print("Cross entropy of true vs predicted distribution:", np_cross_entropy(true, pred))
print("KL divergence of true vs predicted distribution:", np_kl_divergence(true, pred))

# 为什么训练过程中损失会下降？
# 模型预测的分布越来越接近真实分布