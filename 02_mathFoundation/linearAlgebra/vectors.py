class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for addition")
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for subtraction")
        return Vector([a - b for a, b in zip(self.components, other.components)])
    def __mul__(self, other):
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for multiplication")
        return Vector([a * b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for dot product")
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        """
        计算向量的模（长度）
        """
        return sum(a * a for a in self.components) ** 0.5

    def normalize(self):
        """
        返回单位向量（方向不变，长度为 1）  每个值都除以模（长度）
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector([a / mag for a in self.components])

    def cosine_similarity(self, other):
        """
        计算两个向量的余弦相似度
        返回值在 -1 到 1 之间，表示两个向量的相似程度，1 表示完全相似，-1 表示完全相反。
        """
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for cosine similarity")
        return self.dot(other) / (self.magnitude() * other.magnitude())


    def angle_between(self, other):
        """
        计算两个向量之间的夹角（以弧度为单位）
        """
        if self.dim != other.dim:
            raise ValueError("Dimensions must match for angle calculation")
        import math
        # 防止浮点误差导致 acos() 报错
        cos_theta = max(-1.0, min(1.0, self.cosine_similarity(other)))

        angle_radians = math.acos(cos_theta)

        # 弧度转角度
        return math.degrees(angle_radians)

    

    
    def __repr__(self):
        """
        返回向量的字符串表示，便于调试和打印
        """
        return f"Vector({self.components})"

class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]) if len(self.rows) > 0 else 0)

    def __matmul__(self, other):
        """
        矩阵乘法
        返回一个新的矩阵，表示 self 与 other 的矩阵乘积
        """
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))  # 的每一个元素相乘再求和
                for i in range(self.shape[0])  # 对每一行
            ])
        rows = []
        # 对每一行，都会形成一个行结果
        for i in range(self.shape[0]):
            result_row = []
            for j in range(other.shape[1]):
                sum_product = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1]))
                result_row.append(sum_product)
            rows.append(result_row)
        return Matrix(rows)

    def transpose(self):
        """
        矩阵转置
        返回一个新的矩阵，表示 self 的转置矩阵
        """
        transposed_rows = [[self.rows[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return Matrix(transposed_rows)

    def rank(self):
        """
        计算矩阵的秩
        返回矩阵的秩
        """
        # 复制矩阵，避免修改原矩阵
        row = [
            row[:]
            for row in self.rows
        ]

        row_count = self.shape[0]
        col_count = self.shape[1]

        rank = 0

        for col in range(col_count):
            # 寻找当前主元
            pivot = None
            for row in range(rank, row_count):
                if abs(self.rows[row][col]) > 1e-10:
                    pivot = row
                    break
            # 如果找到了主元，继续进行行交换和消元
            if pivot is None:
                continue
            # 把找到的 pivot 行换到当前应该处理的位置。
            self.rows[rank], self.rows[pivot] = self.rows[pivot], self.rows[rank]
            scale = self.rows[rank][col]
            self.rows[rank] = [x / scale for x in self.rows[rank]]
            for r in range(row_count):
                if r != rank and abs(self.rows[r][col]) > 1e-10:
                    factor = self.rows[r][col]
                    self.rows[r] = [self.rows[r][c] - factor * self.rows[rank][c] for c in range(col_count)]
            rank += 1
        return rank

    def __repr__(self):
        """
        返回矩阵的字符串表示，便于调试和打印
        """
        return f"Matrix({self.rows})"



# 线性无关性和投影
def is_linearly_independent(vectors):
    """
    判断一组向量是否线性无关
    如果向量组线性无关，返回 True；否则返回 False
    """

    n = len(vectors)
    dim = len(vectors[0].components)  # len(list)  一个向量的特征数，维度 ，元素个数
    mat = Matrix([v.components[:] for v in vectors])  # 向量组成矩阵的每一行 做高斯消元
    rows = [row[:] for row in mat.rows] # 复制一个矩阵，高斯消元会大量修改
    rank = 0 # 计算rank 独立方向  只要rank == n 就可以提前返回 True

    for col in range(dim):
        pivot = None # 主元
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:  # 找到主元
                pivot = row
                break
        if pivot is None:
            continue
        # 把找到的 pivot 行换到当前应该处理的位置。
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]] # 主元变成 1

        # 消元
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]  # 这一行的 1的倍数 
                rows[row] = [rows[row][c] - factor * rows[rank][c] for c in range(dim)]  # rows[rank][c] 主元是 1，其他的也要一起算
        rank += 1
    return rank == n


# vector投影
def project(a, b):
    """
    将向量 a 投影到向量 b 上
    返回投影向量
    """
    scaler = a.dot(b) / b.dot(b) # a在 b的方向上应该取多少倍
    return Vector([scaler * x for x in b.components])  # 对b 中的每个值投影放缩即可


def gram_schmidt(vectors):
    """
    对一组向量进行 Gram-Schmidt 正交化
    返回正交化后的向量列表   
    把一组线性独立的向量，变成一组彼此垂直，长度为 1 的向量
    """
    orthogonal_vectors = []
    for v in vectors:
        w = v # 一个向量，减掉 在已有方向上的投影 比如 
        """
        1 0
        1 1

        1 0 ---> 保持  
        1 1 --->减掉已有方向[1 0]方向上的投影，  即 [0 1]
        """
        for u in orthogonal_vectors:  
            proj = project(w, u)
            w = w - proj
        if w.magnitude() > 1e-10:  # 避免数值误差导致的零向量
            orthogonal_vectors.append(w.normalize())  # 归一化，长度为 1
    return orthogonal_vectors

# test4 
def verify_orthonormal(vectors, tol=1e-10):
    """
    检查一组向量是否正交归一化
    vectors: 向量列表
    tol: 容差，用于判断浮点数是否接近 0
    返回 True 如果向量组正交归一化，否则返回 False
    """
    all_valid = True
    print("检查每个向量的模：")
    for i, v in enumerate(vectors):
        mag = v.magnitude()
        print(f"Vector {i+1} magnitude: {mag}")
        if abs(mag - 1) > tol: # 模不是 1
            all_valid = False
    for i, v in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            if abs(v.dot(vectors[j])) > tol: # 点积不是0，不正交
                all_valid = False
    return all_valid
    
if __name__ == "__main__":
    # 测试向量和矩阵的基本操作
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print("v1 + v2 =", v1 + v2)
    print("v1 - v2 =", v1 - v2)
    print("v1 * v2 =", v1 * v2)
    print("v1 · v2 =", v1.dot(v2))
    print("v1 magnitude =", v1.magnitude()) # √14
    print("v1 normalized =", v1.normalize())
    print("cosine similarity =", v1.cosine_similarity(v2))

    # 测试矩阵的基本操作
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[7, 8], [9, 10], [11, 12]])
    print("m1 @ m2 =", m1 @ m2)
    print("m1 transpose =", m1.transpose())
    print("m2 transpose =", m2.transpose())


    import random

    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)]) # 创建2*3 权重矩阵
    input_vector = Vector([1.0, 0.5, -0.3])  # 一个样本有三个特征的输入向量  比如猫的三个特征 长度、体重、毛色

    output = weights @ input_vector  # 实际就是神经网络中的  y = wx + b 中的 wx   weights 是 2*3 的矩阵 2 表示输出的类别 比如猫和狗，数字，3 表示输入特征数   这个权重表示的就是，对于猫和狗在三个特征下，对应的权重值。
    print(f"Input (3D): {input_vector}")
    print(f"Output (2D): {output}")  # 对于三个特征，算出两个类别的分数的分数，分数高就倾向谁
    print("This is what a neural network layer does -- matrix multiplication.")


    # 线性无关和投影
    print("-------" * 10)
    print("线性无关和投影")
    v1 = Vector([1, 0, 0])
    v2 = Vector([1, 1, 0])
    v3 = Vector([1, 1, 1])
    print("正交化之后的向量：")
    basis = gram_schmidt([v1, v2, v3])
    for i, u in enumerate(basis):
        print(f"u{i+1} = {u}")
        print(f"  |u{i+1}| = {u.magnitude():.6f}")

    print("向量两两垂直正交")
    print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
    print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
    print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")


    # test2
    test2_v = Vector([1, 1])
    # 创建权重矩阵，让向量 test2_v 的x坐标加倍 y 坐标加三倍
    test2_matrix = Matrix([[2, 0], [0, 3]]) # 对各个坐标轴分别缩放自己的倍数，就是一个对角权重矩阵
    test2_output = test2_matrix @ test2_v
    print(f"Test2 input: {test2_v}")
    print(f"Test2 output: {test2_output}")


    # test3
    # 给定5个随机的类词向量，维度为 50
    import random
    random.seed(42)
    # 本质就是 5 个样本，每个样本 50 个数，使用高斯随机分布创建每个值
    test3_vectors = [Vector([random.gauss(0, 1) for _ in range(50)]) for _ in range(5)]
    # for i, v in enumerate(test3_vectors):
    #     print(f"Test3 vector {i+1}: {v}")

    # 找出余弦相似度最高的两个向量
    best_similarity = -1
    best_pair = None

    # 两两比较
    for i in range(len(test3_vectors)):
        for j in range(i + 1, len(test3_vectors)):
            similarity = test3_vectors[i].cosine_similarity(test3_vectors[j])
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (i, j)

    print(f"Best cosine similarity: {best_similarity:.6f} between vectors {best_pair[0]+1} and {best_pair[1]+1}")
    for i, v in enumerate(test3_vectors):
        if i == best_pair[0] or i == best_pair[1]:
            print(f"Test3 vector {i+1}: {v}")


    # test4 检查每一对元素的点积是否为 0，以及每个向量的模是否为 1。
    test4_vectors1 = Vector([1, 0, 0])
    test4_results = verify_orthonormal([test4_vectors1, Vector([0, 1, 0]), Vector([0, 1, 1])])

    # Vector 1 magnitude: 1.0
    # Vector 2 magnitude: 1.0
    # Vector 3 magnitude: 1.4142135623730951
    # Test4 results: False

    print(f"Test4 results: {test4_results}")


    # test4
    # 创建一个秩 为 2 的 3 3 矩阵
    test4_matrix = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    print(f"Test4 matrix:\n{test4_matrix}")
    print(f"Test4 matrix rank: {test4_matrix.rank()}")
    print("这个矩阵的列空间 是三维空间中的一个二维平面")
    print("rank 表示 这些列向量能够张成多少维度的空间")

    # test5
    test5_a = Vector([1, 2, 3])
    test5_b = Vector([1, 1, 1])
    proj = project(test5_a, test5_b)
    print(f"a @ b = {test5_a.dot(test5_b)}")
    print(f"|b|^2 = {test5_b.magnitude()**2}")
    # a @ b = 6
    # |b|^2 = 2.9999999999999996
    # scale = 6/3 = 2 所以结果是 2 * b = 【2， 2， 2】 
    # # 向量 [1,2,3] 在 [1,1,1] 这个方向上的“影子 是【2， 2， 2】 
    print(f"计算过程：a * b / |b|^2")  
    print(f"Test5 projection of {test5_a} onto {test5_b}: {proj}")
    