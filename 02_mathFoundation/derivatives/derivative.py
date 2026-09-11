
def numerical_derivative(f, x, h=1e-5):
    """
    使用中心差分法计算函数f在点x处的数值导数。
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    """
    示例函数：f(x) = x^2
    """
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    derivative = numerical_derivative(f, x)
    print(f"f'({x}) ≈ {derivative}")  # 输出数值导数的近似值
    analytical_derivative = 2 * x # # 解析导数 f'(x) = 2x
    print(f"Analytical f'({x}) = {analytical_derivative}") 

print("-------------------------------")
# 偏导数和梯度
def numerical_gradient(f, point, h=1e-5):
    """
    使用中心差分法计算函数 f 在点x 处的数值梯度。
    """
    gradient = []
    for i in range(len(point)):
        point_plus = list(point) # +h
        point_minus = list(point) # -h
        point_plus[i] += h
        point_minus[i] -= h
        # 计算 每个点的偏导数
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    """
    示例函数：f(x, y) = x^2 + 3xy + y^2
    """
    x, y = point
    return x ** 2 + 3*x*y + y ** 2

grad = numerical_gradient(f_multi, [1, 2])
print(f"Gradient at (1, 2) ≈ {grad}")  # 输出数值梯度的近似值
analytical_grad = [2*1 + 3*2, 3*1 + 2*2] # 解析梯度 [2x + 3y, 3x + 2y]
print(f"Analytical Gradient at (1, 2) = {analytical_grad}")

print("-------------------------------")
# 使用梯度下降法求 F(x)= x^2 的最小值
def gradient_descent(f, initial_x, learning_rate=0.1, num_iterations=100):
    """
    使用梯度下降法寻找函数 f 的最小值。
    """
    x = initial_x
    for step in range(num_iterations):
        grad = numerical_derivative(f, x)
        x -= learning_rate * grad
    return x

# x 表示初始点，从 5.0 开始，学习率为 0.1，迭代 100 次 f(x) = x^2 = 25 开始， 沿着梯度下降，最终会接近最小值 0
x = 5.0
lr = 0.1
min_x = gradient_descent(f, x, learning_rate=lr)
print(f"Minimum of f(x) = x^2 found at x ≈ {min_x}, f(x) ≈ {f(min_x)}")
# 基本约等于 0 0
# Minimum of f(x) = x^2 found at x ≈ 1.0185179881726768e-09, f(x) ≈ 1.037378892231317e-18

print("-------------------------------")
# 4 二维函数上的梯度下降法
def f_2d(point):
    """
    示例二维函数：f(x, y) = x^2 + y^2
    """
    x, y = point
    return x ** 2 + y ** 2

def gradient_descent_2d(f, initial_point, learning_rate=0.1, num_iterations=100):
    """
    使用梯度下降法寻找二维函数 f 的最小值。
    """
    point = initial_point
    for step in range(num_iterations):
        # 基于原始点和更新点的位置 计算梯度
        grad = numerical_gradient(f_2d, point)
        # 更新点的位置
        point = [p - learning_rate * g for p, g in zip(point, grad)]
    return point

initial_point = [4.0, 3.0]
learning_rate = 0.1
min_point = gradient_descent_2d(f_2d, initial_point, learning_rate=learning_rate)
print(f"Minimum of f(x, y) = x^2 + y^2 found at point ≈ {min_point}, f(x, y) ≈ {f_2d(min_point)}")
# Minimum of f(x, y) = x^2 + y^2 found at point ≈ [1.0185179881726768e-09, 7.638884911294513e-10], f(x, y) ≈ 1.637378892231317e-18  
# 接近最小值 (0, 0)

print("-------------------------------")
# 5 比较数值倒数和解析导数的结果, 数值和解析导数的误差几乎为零
import math
test_functions = [
    ("x^2", lambda x: x ** 2, lambda x: 2 * x),
    ("x^3", lambda x: x ** 3, lambda x: 3 * x ** 2),
    ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
    ("e^x", lambda x: math.exp(x), lambda x: math.exp(x)),
    ("1/x", lambda x: 1 / x, lambda x: -1 / (x ** 2)),
]
x = 2.0
for name, f, df in test_functions:
    numerical = numerical_derivative(f, x)
    analytical = df(x)
    err = abs(numerical - analytical) # 计算数值导数和解析导数之间的误差
    print(f"{name}: Numerical derivative at x={x} ≈ {numerical}, Analytical derivative = {analytical}, Error ≈ {err}")


print("--------------6-----------------")
# 6 数值计算 hessian 矩阵
def hessian_2d(f, x, y, h=1e-5):
    """
    计算二维函数 f 在点 (x, y) 处的 Hessian 矩阵。
    """
    f_xx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    f_yy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    f_xy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[f_xx, f_xy], [f_xy, f_yy]]

def saddle(x, y):
    """
    示例函数：f(x, y) = x^2 - y^2
    """
    return x ** 2 - y ** 2

def bowl(x, y):
    """
    示例函数：f(x, y) = x^2 + y^2
    """
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)

print(f"Hessian of saddle at (0, 0) ≈ {H_saddle}")  # 特征值为正负，鞍点
print(f"Hessian of bowl at (0, 0) ≈ {H_bowl}")  # 特征值为正，最小值

# 7 泰勒近似
def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_value = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1) # 一阶泰勒展开
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2) # 二阶泰勒展开
    print(f"h = {h}: True value = {true_value}, Taylor order 1 ≈ {t1}, Taylor order 2 ≈ {t2}")

print("Taylor approximation completed.")
print("-------------------------------")

# 8 对神经网络为什么重要
import random
random.seed(42)
w = random.gauss(0, 1)  # 初始化权重为均值为0，标准差为1的高斯随机数
b = random.gauss(0, 1)  # 初始化偏置为均值为0，标准差为1的高斯随机数
lr = 0.01  # 学习率

xs = [ 1, 2, 3, 4, 5]
ys = [3, 5, 7, 9, 11]

# 简单的线性回归训练示例
for epoch in range(100):
    total_loss = 0.0
    dw = 0
    db = 0

    for x_val, y_val in zip(xs, ys):
        y_pred = w * x_val + b
        error = y_pred - y_val
        total_loss += error ** 2 # loss 按照平方误差来计算 (y - y_pred)^2

        # 累积梯度
        dw += 2 *error * x_val  # 对权重的梯度 L 对 w 的偏导 按照链式法则：dL/dw = dL/dy_pred * dy_pred/dw = 2 * error * x_val
        db += 2 *  error  # 对偏置的梯度 L 对 b 的偏导 按照链式法则：dL/db = dL/dy_pred * dy_pred/db = 2 * error
    # 梯度下降更新权重和偏置
    # dw = 5个样本的梯度之和
    # db = 5个样本的梯度之和
    # total_loss = 5个样本的loss之和
    # 取平均梯度和平均损失后再更新权重和偏置
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)  # 计算平均损失
    w -= lr * dw
    b -= lr * db
    if (epoch % 40 == 0 or epoch == 199):
        print(f"Epoch {epoch}: Total loss = {total_loss}, Weight = {w}, Bias = {b}")

print(f"Final trained weight: {w}, Final trained bias: {b}")
print(f"actual values: {ys}")
print(f"predicted values: {[w * x + b for x in xs]}")



def numerical_second_derivative(f, x):
    h = 1e-5
    return numerical_derivative(lambda y: numerical_derivative(f, y), x)
import numpy as np
fx = lambda x: x ** 3
x_val = 2.0
second_derivative = numerical_second_derivative(fx, x_val)
activate_value = 3 * x_val ** 2
print(f"Numerical second derivative of x^3 at x = {x_val} ≈ {second_derivative}")
print(f"Function value of x^3 at x = {x_val} = {activate_value}")
print(f"is equal {np.isclose(second_derivative, activate_value)}")

fxy = lambda x, y: (x-3) ** 2 + (y + 1) ** 2  # 示例二元函数
x_val, y_val = 0, 0
# 梯度下降且收敛
learning_rate = 0.1
for i in range(100):
    # grad_x = 2 * (x_val - 3)
    # grad_y = 2 * (y_val + 1)
    grad_x = numerical_derivative(lambda x: fxy(x, y_val), x_val)
    grad_y = numerical_derivative(lambda y: fxy(x_val, y), y_val)
    x_val -= learning_rate * grad_x
    y_val -= learning_rate * grad_y
print(f"Converged to (x, y) ≈ ({x_val}, {y_val})")
