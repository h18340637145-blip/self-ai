import numpy as np

a = np.array([1,2,3], dtype=float)
b = np.array([4,5,6], dtype=float)

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a)}")
print(f"cosine similarity = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))}")



# rank 投影 qr
A = np.array([[1, 2], [2, 4]])
print(f"rank of A = {np.linalg.matrix_rank(A)}")

a = np.array([3,4])
b = np.array([1, 0])

proj = (np.dot(a, b) / np.dot(b, b)) * b

print(f"projection of a onto b = {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3)) # 原矩阵 = 正交矩阵（长度 1 且正交，就是 smt 正交化得到的正交向量排成的矩阵） × 上三角矩阵（上有数据，原矩阵的列向量，在 Q 这些正交方向上分别有多少成分。）

print(f"Q is orthogonal = {np.allclose(Q.T @ Q, np.eye(3))}")  # np.allclose(Q @ Q.T, np.eye(3))  一定是满足点乘是单位矩阵的
print(f"R is upper triangular = {np.allclose(R, np.triu(R))}")


import torch 
x = torch.randn(3, requires_grad=True) # PyTorch 要记录和 x 有关的计算，之后我要对它求导
y = torch.tensor([1.0, 0.0, 0.0 ])

similarity = torch.dot(x, y)
print(f"cosine similarity = {similarity / (x.norm() * y.norm())}")
similarity.backward() # 计算 similarity 对所有需要梯度的变量的导数。

"""
这个例子演示了如何使用 PyTorch 计算向量的点积及其关于输入向量的梯度。
y = 1 0 0
相似度其实就是 x1  x2 和 x3都是 0，因为 y的另外两个维度都是 0

所以相似度对 x 求梯度，实际就是对 x1 方向的投影
x.grad 就是 1 0 0


x = tensor([ 0.8451,  1.1467, -1.5984])
y = tensor([1., 0., 0.])
dot product = 0.8450527191162109
d(dot)/dx = tensor([1., 0., 0.])
"""


print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {torch.dot(x, y)}")
print(f"d(dot)/dx = {x.grad}")  # 点积关于 x 的梯度就是 y

