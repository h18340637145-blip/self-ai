import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

point = np.array([1, 0])
print("Original point:", point)
rotated_point = R @ point
print("Rotated point:", rotated_point)

S = np.diag([2, 3])
scaled_point = S @ point
print("Scaled point:", scaled_point)

A = np.array([[2, 1], [1, 2]], dtype=float)
# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# 计算每个特征值对应的特征向量
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i] # 获取第 i 个特征向量  第 i 列就是特征向量
    print(f"Eigenvector for eigenvalue {eigenvalues[i]}: {v}")
    lam = eigenvalues[i]
    # 验证特征向量是否满足 A @ v = lam * v
    left_side = A @ v
    right_side = lam * v
    print(f"Verification for eigenvalue {lam}: A @ v = {left_side}, lam * v = {right_side}, Equal: {np.allclose(left_side, right_side)}")


print("det(R) =", np.linalg.det(R))
print("det(S) =", np.linalg.det(S))


B = np.array([
    [3,1],
    [0,2]
], dtype=float)
vals, vecs = np.linalg.eig(B) 
print("Eigenvalues of B:", vals)
print("Eigenvectors of B:", vecs)
# 特征分解
D = np.diag(vals)
V = vecs
# 验证 B = V D V^{-1}
B_reconstructed = V @ D @ np.linalg.inv(V)
print("Reconstructed B from eigen decomposition:", B_reconstructed)
print(f"original B\n{B}")
print("\n Eigen decomposition B = V D V^{-1} \n", B_reconstructed)

# numpy 进行3d 旋转
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def rotation_3d_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

point_3d = np.array([1, 0, 0])
theta_3d = np.pi / 4
R_z = rotation_3d_z(theta_3d)
rotated_point_3d_z = R_z @ point_3d
print("Original 3D point:", point_3d)
print("Rotated 3D point around Z-axis:", rotated_point_3d_z)


# test1
square_matrix = np.array([
    [0, 0], 
    [1, 0],
    [1, 1],
    [0, 1]
])
print("Original square matrix:\n", square_matrix)
print("shape of square matrix:", square_matrix.shape)

# 旋转
theta = np.pi / 4
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

'''
square 中 每个顶点是一行
如果按列向量公式： newpoint = R @ oldpoint   所有的 point组成 square 的行，那么批量计算可以写成：(R @ square_matrix.T).T
那批量计算可以写成：
(R @ square_matrix.T).T
'''
rotated_square = (R @ square_matrix.T).T
print("Rotated square matrix:\n", rotated_square)

scale_matrix = np.diag([2, 3])
scaled_square = (scale_matrix @ square_matrix.T).T
print("Scaled square matrix:\n", scaled_square)

shear_matrix = np.array([
    [1, 1],
    [0, 1]
])
sheared_square = (shear_matrix @ square_matrix.T).T
print("Sheared square matrix:\n", sheared_square)

# 简写
rotated_square_short = square_matrix @ R.T
scaled_square_short = square_matrix @ scale_matrix.T
sheared_square_short = square_matrix @ shear_matrix.T
print("Rotated square matrix (short):\n", rotated_square_short)
print("Scaled square matrix (short):\n", scaled_square_short)


# test2
test_matrix = quadratic_matrix = np.array([
    [2, 1],
    [1, 2]
], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(test_matrix)
print("Eigenvalues of test_matrix:", eigenvalues)
print("Eigenvectors of test_matrix:\n", eigenvectors)

# test3
theta3 = np.pi / 6
R3 = np.array([
    [np.cos(theta3), -np.sin(theta3)],
    [np.sin(theta3), np.cos(theta3)]
])
print("Rotation matrix R3:\n", R3)

scale3 = np.array([
    [1.5, 0],
    [0, 0.8]
])

shearing3 = np.array([
    [1, 0.3],
    [0, 1]
])

# 创建8个圆上的点
circle_points = np.array([
    [np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)]
    for i in range(8)
])
print("Original circle points:\n", circle_points)


circle_points_rotated = (R3 @ circle_points.T).T
print("Rotated circle points:\n", circle_points_rotated)
scaled_circle_points = (scale3 @ circle_points_rotated.T).T
print("Scaled circle points:\n", scaled_circle_points)
sheared_circle_points = (shearing3 @ scaled_circle_points.T).T
print("Sheared circle points:\n", sheared_circle_points)

# 复合矩阵
composite_matrix = shearing3 @ scale3 @ R3

result = circle_points @ composite_matrix.T
print("Result of composite transformation:\n", result)
print(f"Equal to sequential transformations: {np.allclose(result, sheared_circle_points)}")

composite_matrix_det = np.linalg.det(composite_matrix)  # 计算复合矩阵的行列式

R3_det = np.linalg.det(R3)
scale3_det = np.linalg.det(scale3)
shearing3_det = np.linalg.det(shearing3)

print("Determinant of composite matrix:", composite_matrix_det)
print("Determinant of R3:", R3_det)
print("Determinant of scale3:", scale3_det)
print("Determinant of shearing3:", shearing3_det)
print(f"Product of determinants: {R3_det * scale3_det * shearing3_det}, Equal to composite determinant: {np.isclose(composite_matrix_det, R3_det * scale3_det * shearing3_det)} ")

