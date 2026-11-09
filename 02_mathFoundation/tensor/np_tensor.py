import numpy as np

activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
print(f"activations={activations}")
print(f"bias={bias}")
# 4 行中每一行都加上 bias 的值
result = activations + bias
print(f"result={result}")

# 理解这个图片，就是一个 2 批次、3 个通道、4x4 高宽的图像数据。
# 图片0： 通道 0 1 2 都是 4*4 的矩阵
image = np.random.randn(2,3,4,4)
print(f"image={image}")

# 原本scale是一个一维数组； 改变这个数组的维度结构， 应用到 image 的每个通道上，每个通道应用不同的缩放因子 0.5, 1.0, 1.5
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)  # 广播到 image 的形状 (2,3,4,4)
print(f"scale={scale}")
result = image * scale
print(f"result={result}")

# 
a = np.array([1,2,3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
print(f"outer={outer}")


aa = np.array([1, 2, 3])
bb = np.array([4, 5, 6])
# 4 + 10 + 18 = 32
dot = np.einsum("i,i->", aa, bb) # 计算向量的点积  爱因斯坦求和
print(f"dot={dot}")

A = np.array([[1, 2], [3, 4], [5, 6 ]])
B = np.array([[7, 8 ,9], [10, 11, 12]]) 

matmul = np.einsum("ik,kj->ij", A, B)
print(f"matmul={matmul}")

batch_A = np.random.randn(4,3, 5)
batch_B = np.random.randn(4, 5, 6)
batch_matmul = np.einsum("bij,bjk->bik", batch_A, batch_B)
print(f"batch_matmul={batch_matmul}")


# B-批次大小
# H-头的数量
# T-序列长度
# D-每个头的维度
# E-每个序列的嵌入维度 (等于 H * D)   每个token的完整特征维度
B, H, T, D = 2, 4, 8, 16
E = H * D # 64   注意力头数*每个头的维度

X = np.random.randn(B, T, E) # (2, 8, 64)   两个样本，每个样本 8 个 token  每个 token 的特征维度是 64  64 维度特征被分成 4 个注意力头

# Batch 0
#     Token 0 → 64维向量
#     Token 1 → 64维向量
#     Token 2 → 64维向量
#     ...
#     Token 7 → 64维向量

# Batch 1
#     Token 0 → 64维向量
#     Token 1 → 64维向量
#     ...
#     Token 7 → 64维向量

W_q = np.random.randn(E, E) * 0.02  # 64 * 64总特征维度   Query的投影矩阵
Q = np.einsum("bte,ef->btf", X, W_q) # 计算每个序列的查询向量 (Query)  本质就是X@W_q

# 第一个张量 X：

# b → Batch
# t → Token
# e → 输入特征

# 第二个张量 W_q：

# e → 输入特征
# k → 输出特征

# 输出：

# b → Batch
# t → Token
# k → 输出特征


# 对第一个样本的第一个token X[0, 0, :]  # 64维向量
# 乘 W_q         # 64×64
# 得到 Q[0, 0, :]  # 64维向量
# 本质就是  X[0, 0, :] @ W_q  # 矩阵乘法
print(f"Q={Q}")
print(f"Q shape {Q.shape}")
# 将 Q 拆成多个注意力头  之前是 2 8 64转成 batch token head headDimension  最终形状是 (B, H, T, D)
# 原本每个token有64个特征
# 拆成4个注意力头，每个头有16个特征
"""
拆分成4个Head：

Head 0 → 16维
Head 1 → 16维
Head 2 → 16维
Head 3 → 16维
"""

Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3) # 把头的维度移到第二个位置，最终形状是 (B, H, T, D)  就可以让每个头独立处理每个token的特征
print(f"Q reshaped and transposed={Q}")
# 对于一个 token  当前 token 应该关注序列中的哪些 token --- 分数的意义
# Q 表示当前token用来查找信息的特征
# K 表示其他token用来被匹配的特征
# token 0 的 query  与  token 0 的 key 的分数，token1 的 key的分数，依此类推
score = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)  # 缩放点积注意力，除以每个头的维度的平方根
# Q的形状 (B, H, T, D)  即 (2, 4, 8, 16)
# K的形状 (B, H, T, D)  即 (2, 4, 8, 16)
# S= Q@Kt   8 * 16 @ 16 * 8---> 8 * 8
# score的形状 (B, H, T, S)  即 (2, 4, 8, 8)  每个token与序列中每个token的匹配分数

# D = 每个注意力头的维度  即 16  可以降低 Softmax 过早进入饱和区域的风险，让梯度更加稳定


#                   Key Token
# 注意力分数矩阵 每一行表示某个 q 对所有 keytoken的注意力分数
#                0   1   2   3 ... 7
# Query Token 0  ●   ●   ●   ●     ●
# Query Token 1  ●   ●   ●   ●     ●
# Query Token 2  ●   ●   ●   ●     ●
# Query Token 3  ●   ●   ●   ●     ●
# ...
# Query Token 7  ●   ●   ●   ●     ●


# 第五步：Softmax 转成注意力权重
# score.shape = (B, H, T, S)  即 (2, 4, 8, 8)
weights = softmax(score, axis=-1)  # 对每个 token 的注意力分数进行归一化，得到注意力权重  沿着最后一维进行计算
# weights.shape = (B, H, T, S)  即 (2, 4, 8, 8)  每个 token 对序列中每个 token 的注意力权重
print(f"weights={weights}") # 注意力权重矩阵  是概率分布 和为 1 的
print(f"weights shape {weights.shape}")

attn_output = np.einsum("bhts, bhsd->bhta", weights, V) # 计算加权和，得到每个 token 的注意力输出 形状 (B, H, T, D)
print(f"attn_output={attn_output}")
print(f"attn_output shape {attn_output.shape}")