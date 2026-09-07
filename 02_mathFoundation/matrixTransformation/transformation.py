import math

def rotation_2d(theta):
    """
    2D 旋转矩阵
    :param theta: 旋转角度，单位为弧度
    :return: 旋转矩阵
    """
    return [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]

def scaling_2d(sx, sy):
    """
    2D 缩放矩阵
    :param sx: x 方向缩放因子
    :param sy: y 方向缩放因子
    :return: 缩放矩阵
    """
    return [
        [sx, 0],
        [0, sy]
    ]

def shearing_2d(shx, shy):
    """
    2D 剪切矩阵
    :param shx: x 方向剪切因子
    :param shy: y 方向剪切因子
    :return: 剪切矩阵
    """
    return [
        [1, shx],
        [shy, 1]
    ]

def reflection_x():
    """
    关于 x 轴的反射矩阵
    :return: 反射矩阵
    """
    return [
        [1, 0],
        [0, -1]
    ]

def reflection_y():
    """
    关于 y 轴的反射矩阵
    :return: 反射矩阵
    """
    return [
        [-1, 0],
        [0, 1]
    ]

def mat_vec_mul(matrix, vector):
    """
    矩阵与向量相乘
    :param matrix: 矩阵
    :param vector: 向量
    """
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    """
    矩阵相乘
    :param a: 矩阵 a
    :param b: 矩阵 b
    """
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    return [

        [
            sum(a[i][k] * b[k][j] for k in range(cols_a))
            for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]

point = [1, 0]
angle = math.pi / 4  # 旋转角度为 45 度
rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Original point: {point}, Rotated point: {rotated}")

scaled = mat_vec_mul(scaling_2d(2, 3), point)
print(f"Original point: {point}, Scaled point: {scaled}")

shreared = mat_vec_mul(shearing_2d(1, 0), [1, 1])
print(f"Original point: {[1, 1]}, Sheared point: {shreared}")

reflected = mat_vec_mul(reflection_y(), [2, 1])
print(f"Original point: {[2, 1]}, Reflected point: {reflected}")



# 变换的组合
R = rotation_2d(math.pi / 2)  # 旋转 90 度
S = scaling_2d(2, 0.5)  # 缩放

rotate_then_scale = mat_mul(S, R)  # 先旋转再缩放
scale_then_rotate = mat_mul(R, S)  # 先缩放再旋转

point = [1, 0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)
print(f"Original point: {point}, Rotate then scale: {result1}, Scale then rotate: {result2}")
print(f"Are the two results equal? {result1 == result2}")

# 从头开始计算特征值
def eigenvalues_2x2(matrix):
    """
    计算 2x2 矩阵的特征值
    :param matrix: 2x2 矩阵
    :return: 特征值列表
    """
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d # 对角线相加 矩阵的迹
    det = a * d - b * c # 行列式
    # 特征值来源于 det(λI - A) = 0
    # λ² - trace(A) λ + det(A) = 0
    discriminant = trace ** 2 - 4 * det # b² - 4ac<0 无解，是复数根
    if discriminant < 0:
        real = trace / 2
        imag = math.sqrt(-discriminant) / 2
        return [complex(real, imag), complex(real, -imag)]
    else:
        sqrt_disc = math.sqrt(discriminant)
        return [(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2]

# 特征向量计算
def eigenvectors_2x2(matrix, eigenvalue):
    """
    计算 2x2 矩阵的特征向量
    :param matrix: 2x2 矩阵
    :param eigenvalue: 特征值
    :return: 特征向量列表
    """
    a, b = matrix[0]
    c, d = matrix[1]

    # 解方程 (A - λI)v = 0
    # (a - λ)x + by = 0
    # cx + (d - λ)y = 0

    if abs(b) > 1e-10:  # 避免除以零
        v = [b, eigenvalue - a]   # x=b y=eigenvalue - a 代入到第一个方程中，就是一个解，特征向量不需要精确，只要一个方向即可
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        # 如果 b 和 c 都为零，说明矩阵是对角矩阵，特征向量可以是标准基向量
        v = [1, 0] if a == eigenvalue else [0, 1]
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return [v[0] / mag, v[1] / mag]  # 归一化特征向量

A = [[2, 1], [1, 2]]
values = eigenvalues_2x2(A)
print(f"Eigenvalues of A: {values}")

for val in values:
    vec = eigenvectors_2x2(A, val)
    print(f"Eigenvector for eigenvalue {val}: {vec}")
    result = mat_vec_mul(A, vec) # A @ vec
    scaled = [val * vec[0], val * vec[1]] # lambda * vec
    print(f"A * vec: {result}, val * vec: {scaled}")
    print(f" A@vec == val * vec? {result == scaled} ")


# 行列式作为体积缩放因子
def det_2x2(matrix):
    """
    计算 2x2 矩阵的行列式
    :param matrix: 2x2 矩阵
    :return: 行列式值
    """
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c

print(f"Determinant of A: {det_2x2(A)}")
print(f"Determinant of rotation matrix: {det_2x2(rotation_2d(math.pi / 4))}")
print(f"Determinant of scaling matrix: {det_2x2(scaling_2d(2, 3))}")
print(f"Determinant of shearing matrix: {det_2x2(shearing_2d(1, 0))}")
print(f"Determinant of reflection matrix: {det_2x2(reflection_y())}")

singular = [
    [1, 2],
    [2, 4]
]

print(f"Determinant of singular matrix: {det_2x2(singular)}")
print(f"Is singular matrix invertible? {'Yes' if det_2x2(singular) != 0 else 'No'}")

# test1 
square = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]

print(f"shape of square: {len(square)}x{len(square[0])}")
rotated_square = [mat_vec_mul(rotation_2d(math.pi / 4), point) for point in square]
print(f"Original square points: {square}")
print(f"Rotated square points: {rotated_square}")