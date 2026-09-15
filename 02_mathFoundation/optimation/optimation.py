# f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2


def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
# 计算 Rosenbrock 函数的梯度
def rosenbrock_gradient(params):
    x, y = params
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return [dx, dy]

# gd渐变下降
class GradientDescent:
    def __init__(self, lr=0.001, decay=False):
        self.lr = lr
        self.decay = decay
        self.step_count = 0
    # 执行一次梯度下降步骤  参数就是权重 w
    def step(self, params, grads):
        # 增加指数学习率衰减
        if self.decay:
            lr = self.lr * (0.999 ** self.step_count)
        else:
            lr = self.lr
        # 参数更新
        new_params = [
            p - lr * g
            for p, g in zip(params, grads)
        ]
        self.step_count += 1
        return new_params
    
# 动量sgd 使用速度 更新参数
class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    # 执行一次梯度下降
    def step(self, params, grads):
        # 初始化速度向量
        if self.velocity is None:
            self.velocity = [0 for _ in params]
        # 更新速度向量
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]

# Adam 优化器
class Adam:
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)

        self.t += 1
        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * (g ** 2)
            for v, g in zip(self.v, grads)
        ]
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]
        return [
            p - self.lr * m_hat_i / (v_hat_i ** 0.5 + self.epsilon)
            for p, m_hat_i, v_hat_i in zip(params, m_hat, v_hat)
        ]

# 运行并比较
def optimize(optimizer, func, grads_func, start, steps=1000):
    params = start
    history = [params[:]]

    for _ in range(steps):
        grads = grads_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1, 1]
# 使用普通GD优化
gd_history = optimize(GradientDescent(lr=0.001), rosenbrock, rosenbrock_gradient, start)

# 使用动量SGD优化
adam_history = optimize(Adam(lr=0.001), rosenbrock, rosenbrock_gradient, start)
# 使用动量SGD优化
sgd_history = optimize(SGDMomentum(lr=0.001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)

# 
for name, history in [("GD", gd_history), ("Momentum SGD", sgd_history), ("Adam", adam_history)]:
    print(f"{name}: Final parameters: {history[-1]}")
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name}: Final loss: {loss}")

# f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
# 它的最小值位于 (1, 1) 处，位于一个狭窄的弯曲山谷内，这个山谷很容易找到，但很难追踪。

# 对 Rosenbrock 函数运行标准梯度下降法，学习率分别为 [0.0001, 0.0005, 0.001, 0.005, 0.01]
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
gd_histories = {}
for lr in learning_rates:
    try:
        gd_histories[lr] = optimize(GradientDescent(lr=lr), rosenbrock, rosenbrock_gradient, start)
    except Exception as e:
        print(f"Failed to optimize with lr={lr}: {e}")

for lr, history in gd_histories.items():
    print(f"GD with lr={lr}: Final parameters: {history[-1]}")
    final = history[-1]
    loss = rosenbrock(final)
    print(f"GD with lr={lr}: Final loss: {loss}")

# 对 Rosenbrock 函数运行动量值为 [0.0, 0.5, 0.9, 0.99] 的随机梯度下降 (SGD) 算法

momentum_values = [0.0, 0.5, 0.9, 0.99]
sgd_histories = {}
for momentum in momentum_values:
    try:
        history = optimize(
            SGDMomentum(lr=0.001, momentum=momentum), rosenbrock, rosenbrock_gradient, start
        )
        sgd_histories[momentum] = history
    except Exception as e:
        print(f"Failed to optimize with momentum={momentum}: {e}")

for momentum, history in sgd_histories.items():
    print(f"SGD with momentum={momentum}: Final parameters: {history[-1]}")
    final = history[-1]
    loss = rosenbrock(final)
    print(f"SGD with momentum={momentum}: Final loss: {loss}")

# \((1,1)\)  目标

# 希望loss在最后阶段是越小越好，越容易接近目标点 (1, 1) 所以 0.9 的动量表现最好
# | Momentum    | 最终参数      | Final Loss |
# |---:|---|---:|
# | `0.0` | `[0.3636, 0.1292]` | `0.405875` |
# | `0.5` | `[0.7284, 0.5292]` | `0.073955` |
# | `0.9` | `[0.9939, 0.9878]` | **`0.00003744`** |    表现最好
# | `0.99` | `[0.9858, 0.9721]` | `0.00021004` |



# 定义函数f(x, y) = x ^ 2 - y ^ 2  鞍点在 0 0 ， 看三种方法哪种可以逃逸鞍点
def f(x, y):
    return x ** 2 - y ** 2

# 设置初始点 0.01 ,0.01
# 比较原始梯度下降法， 带动量的梯度下降法，以及 Adam 优化器在该函数上的表现
start = [0.01, 0.01]
# 使用普通GD优化

gd_history = optimize(GradientDescent(lr=0.001), f, lambda params: [2*params[0], -2*params[1]], start)
# 使用动量SGD优化
sgd_history = optimize(SGDMomentum(lr=0.001, momentum=0.9), f, lambda params: [2*params[0], -2*params[1]], start)
# 使用 Adam 优化
adam_history = optimize(Adam(lr=0.001), f, lambda params: [2*params[0], -2*params[1]], start)

# print final results for each optimizer
for name, history in [("GD", gd_history), ("Momentum SGD", sgd_history), ("Adam", adam_history)]:
    print(f"{name}: Final parameters: {history[-1]}")
    final = history[-1] 
    loss = f(final[0], final[1])
    print(f"{name}: Final loss: {loss}")


# | 优化器 | 最终参数 | 最终 Loss |
# |---|---|---:|
# | GD | `[0.00135, 0.07374]` | `-0.00544` |      无法有效逃离鞍点 |
# | Momentum SGD | `[≈0, 258687.36]` | `-6.69×10¹⁰` |      快速逃离鞍点，但是没有最小值，就一直向最小方向加速｜
# | Adam | `[≈0, 1.58318]` | 约 `-2.51` |      能够较好地逃离鞍点并找到较小的 Loss |


# 从接近鞍点 (0.01, 0.01) 出发，三种优化器都能够沿负曲率方向逃离鞍点。普通 GD 逃逸最慢；Adam 更快；带 0.9 动量的 SGD 由于持续累积下降方向上的速度，逃逸最快。但由于 \(f(x,y)=x^2-y^2\) 在 \(y\) 方向无下界，Momentum 最终出现了严重发散，因此极大的负 loss 并不代表找到了更好的有限最优解。