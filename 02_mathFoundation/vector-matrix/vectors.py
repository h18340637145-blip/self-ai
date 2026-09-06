import random

class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scale):
        return Vector([a * scale for a in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        # 模
        return sum(x ** 2 for x in self.data) ** 0.5

class Matrix:
    '''
    传入参数：
    - data: 二维列表，表示矩阵的元素
    '''
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n ".join(str(row) for row in self.data)
        return f"Matrix({rows_str})"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                # 一行有几个数据，要看 other 的列
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) # 遍历内部维度
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [
                self.data[j][i] for j in range(self.rows) # 将行转列
            ]
            for i in range(self.cols)
        ])

    def determinant(self):
        # 行列式
        if self.shape == (1, 1):
            return self.data[0][0]

        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        det = 0

        for j in range(self.cols):
            # 余子式
            sub_matrix = [
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ]
            det += ((-1) ** j) * self.data[0][j] * Matrix(sub_matrix).determinant()

        return det

    def inverse_22(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        return Matrix([
            [ self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det,  self.data[0][0] / det]
        ])
    """
    计算逆矩阵的过程：
    1. 计算行列式 det(A)
    2. 计算伴随矩阵 adj(A)
        2.1 对于每个元素 a_ij，计算其余子式 M_ij
            2.1.1 余子式 M_ij 是通过删除第 i 行和第 j 列得到的矩阵的行列式
        2.2 计算代数余子式 C_ij = (-1)^(i+j) * M_ij
        2.3 将代数余子式组成矩阵 C，然后转置得到伴随矩阵 adj(A) = C^T
    3. 计算逆矩阵 A^(-1) = adj(A) / det(A)
    其中，伴随矩阵 adj(A) 是由余子式组成的矩阵的转置矩阵。
    对于 2x2 矩阵，伴随矩阵可以直接通过交换对角线元素和改变非对角线元素的符号来计算。
    """
    def minor(self, row, col):
        # 计算余子式
        sub_matrix = [
            [self.data[i][j] for j in range(self.cols) if j != col]
            for i in range(self.rows) if i != row
        ]
        return Matrix(sub_matrix).determinant()

    def cofactor_matrix(self):
        # 计算代数余子式矩阵
        return Matrix([
            [((-1) ** (i + j)) * self.minor(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ])
    def adjugate(self):
        # 计算伴随矩阵
        return self.cofactor_matrix().transpose()

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        return self.adjugate().scalar_multiply(1 / det)

    @staticmethod
    def identity(n):
        # 生成 n*n 单位矩阵
        return Matrix([
            [1 if i == j else 0 for j in range(n)] for i in range(n)
        ])

A = Matrix([
    [1, 2],
    [3, 4]
])

B = Matrix([
    [5, 6],
    [7, 8]
])
print("A + B =", (A + B).data)
print("A - B =", (A - B).data)
print("A * B =", (A.matmul(B)).data)
print("A @ B =", (A.matmul(B)).data)
print("A ^ T =", (A.transpose()).data)
print("det(A) =", A.determinant())
print("inv(A) =", (A.inverse_22()).data)
I = Matrix.identity(2)
print("I =", I.data)


# 连接到神经网络
import random
inputs = Matrix([
    [0.5], [0.8], [0.2]
])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])

bias = Matrix([
    [0.1], [0.1]
])

def relu_matrix(m):
    return Matrix([
        [max(0, m.data[i][j]) for j in range(m.cols)]
        for i in range(m.rows)
    ])


pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)
print(f"Input shape = {inputs.rows}x{inputs.cols}")
print(f"Weights shape = {weights.rows}x{weights.cols}")
print(f"Pre-activation shape = {pre_activation.rows}x{pre_activation.cols}")
print(f"Output shape = {output.rows}x{output.cols}")


# test1
test1_matrix = Matrix([
    [1, 2],
    [3, 4]
])

res = test1_matrix.matmul(test1_matrix.inverse_22())
det = res.determinant()
print("det(res) =", det)

test1_A = Matrix([
    [1, 1],
    [1, 1]
])
print(f"det(test1_A) =", test1_A.determinant())
resA = test1_A.matmul(test1_A.inverse_22())
detA = resA.determinant()
print("det(resA) =", detA)


# test2
# 实现 3 * 3 矩阵的逆矩阵计算
def inverse_33(matrix):
    det = matrix.determinant()
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted")

    # 计算伴随矩阵
    adjugate = []
    for i in range(3):
        row = []
        for j in range(3):
            # 计算余子式
            sub_matrix = [
                [matrix.data[x][y] for y in range(3) if y != j]
                for x in range(3) if x != i
            ]
            cofactor = ((-1) ** (i + j)) * Matrix(sub_matrix).determinant()
            row.append(cofactor)
        adjugate.append(row)

    # 转置伴随矩阵
    adjugate_matrix = Matrix(adjugate).transpose()

    # 计算逆矩阵
    inverse_matrix = adjugate_matrix.scalar_multiply(1 / det)  # 伴随/行列式
    return inverse_matrix

# test3 构建 2层神经网络

inputs = Matrix([
    [0.5], 
    [0.8], 
    [0.2]
])
print(f"Inputs shape = {inputs.rows}x{inputs.cols}")

# 第一层： 4*3 的权重矩阵，4 个神经元，每个神经元有 3 个输入
weights1 = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(4)
])

bias1 = Matrix([
    [random.uniform(-1, 1)] for _ in range(4)
])

# 第二层： 2*4 的权重矩阵，2 个神经元，每个神经元有 4 个输入 接收第一层的输出
weights2 = Matrix([
    [random.uniform(-1, 1) for _ in range(4)]
    for _ in range(2)
])

bias2 = Matrix([
    [random.uniform(-1, 1)] for _ in range(2)
])

def relu_matrix(m):
    return Matrix([
        [max(0, m.data[i][j]) for j in range(m.cols)]
        for i in range(m.rows)
    ])

# 第一层前向传播 4*3 @ 3 * 1 = 4 * 1
output1 = relu_matrix(weights1.matmul(inputs) + bias1)
# 第二层前向传播  
output2 = relu_matrix(weights2.matmul(output1) + bias2)


print(f"Output1 shape = {output1.rows}x{output1.cols}")
print(f"Output2 shape = {output2.rows}x{output2.cols}")

# 神经网络的每一层主要就是在做：y = wx+b，然后再经过激活函数处理。eg Relu, Sigmoid, Tanh 等。
# 本题目需要两层：那就是需要两个权重矩阵和两个偏置矩阵。
# 输入层已知定义为3个神经元，第一层定义为4个神经元，第二层定义为2个神经元。
# 意味着：
# 第一层的权重矩阵是 4*3，偏置矩阵是 4*1
# 第二层的权重矩阵是 2*4，偏置矩阵是 2*1
# 输入层的输入是 3*1，
# 第一层的输出是 4*1，第二层的输出是 2*1