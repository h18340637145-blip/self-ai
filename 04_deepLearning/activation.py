import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1.0 - math.tanh(x) ** 2

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def leaky_relu_derivative(x):
    return 1.0 if x > 0 else 0.01

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    tanh_term = math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + tanh_term + x * (1 - tanh_term ** 2) * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x ** 2))

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    # 导数乘法
    return s + x * s * (1 - s)

# 对于-5 到 5 的输入范围，扫描各激活函数的梯度情况 对每个值进行依次梯度计算，如果当前值的梯度过小，接近于 0，表示这个梯度值会导致无法继续学习
def gradient_scan(name, derivative_fn, start = -5, end = 5, n = 100):
    # 步长计算  
    step = (end - start) / n
    near_zero = 0 # 记录接近零的梯度数量
    healthy = 0 # 记录健康梯度数量
    for i in range(n):
        x = start + i * step # 当前扫描的输入值
        g = derivative_fn(x)
        if abs(g) < 1e-5:  # 梯度接近零，认为是“死亡”梯度  会导致神经网络无法继续学习的梯度大小
            near_zero += 1 # 认为是“死亡”梯度
        else:
            healthy += 1 # 认为是健康梯度
    pct_dead = near_zero / n * 100 # 计算死亡梯度的百分比
    print(f"{name}: {pct_dead:.2f}% near zero, {healthy / n * 100:.2f}% healthy")

gradient_scan("sigmoid", sigmoid_derivative)
gradient_scan("relu", relu_derivative)
gradient_scan("tanh", tanh_derivative)
gradient_scan("leaky_relu", leaky_relu_derivative)
gradient_scan("gelu", gelu_derivative)
gradient_scan("swish", swish_derivative)

# 梯度消失实验
import random
def vanishing_gradient_experiment(activation_fn, derivative_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)] # 初始化输入值，服从标准正态分布
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)] # 初始化权重，服从标准正态分布
        z = sum(w * v for w, v in zip(weights, values)) # 计算加权和 # values 是输入值
        activated = activation_fn(z) # 计算当前层的激活值
        grad = derivative_fn(z) # 计算当前层的梯度
        print(f"Layer {layer+1}, z={z:.4f}, activated={activated:.4f}, grad={grad:.4f}")
        values = [activated] * n_inputs # 将当前层的激活值作为下一层的输入

vanishing_gradient_experiment(sigmoid, sigmoid_derivative, "sigmoid") # sigmoid 的梯度容易消失
vanishing_gradient_experiment(relu, relu_derivative, "relu") # relu 的梯度会有0，可能导致部分神经元死亡
vanishing_gradient_experiment(tanh_act, tanh_derivative, "tanh") # tanh 的梯度容易消失
vanishing_gradient_experiment(leaky_relu, leaky_relu_derivative, "leaky_relu") # leaky_relu 的梯度不会完全消失
vanishing_gradient_experiment(gelu, gelu_derivative, "gelu") # gelu 的梯度在大部分区域不会消失
vanishing_gradient_experiment(swish, swish_derivative, "swish") # swish 的梯度在大部分区域不会消失


# 死亡神经元检测
def dead_neuron_detection(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(42)
    weights = [
        [random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)
    ]
    bias = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * n_inputs
    for _ in range(n_samples): # 对每个样本进行实验，统计每个输入神经元的激活情况
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + bias[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx % n_inputs] += 1 # 统计对应输入神经元的激活次数

    dead = sum(1 for c in fire_counts if c == 0) # 统计死亡神经元的数量
    rarely_fire = sum(1 for c in fire_counts if c < n_samples * 0.1) # 统计很少激活的神经元数量
    healthy_fire = sum(1 for c in fire_counts if c >= n_samples * 0.1) # 统计健康激活的神经元数量
    print(f"Dead neurons: {dead}")
    print(f"Rarely firing neurons: {rarely_fire}")
    print(f"Healthy firing neurons: {healthy_fire}")


    for i, count in enumerate(fire_counts):
        status = "dead" if count == 0 else "rarely firing" if count < n_samples * 0.1 else "healthy"
        print(f"Input {i+1} fired {count} times out of {n_samples} ({status})")

dead_neuron_detection()


def make_circle_data(n = 200, seed=42):
    # 训练对比----sigmoid vs relu vs gelu
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if x**2 + y ** 2 < 1.5 else 0
        data.append(([x, y], label))
    return data

class ActivationNetwork:
    # 这是一个简单的两层神经网络，使用自定义激活函数和反向传播进行训练
    # 一共两层，输入层到隐藏层，隐藏层到输出层
    def __init__(self, actiavtion_fn, activation_deriv, hidden_size = 8, lr = 0.1):
        random.seed(42)
        self.act = actiavtion_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        # 初始化网络参数，包括输入到隐藏层的权重和偏置，以及隐藏层到输出的权重和偏置
        self.w1 = [
            [random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)
        ]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0

    # 前向传播函数，计算网络输出
    def forward(self, x):
        self.x = x # 保存输入以便反向传播使用
        self.z1 = []
        self.h = []
        # 计算隐藏层每个神经元的加权和并应用激活函数
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)

        return self.out

    def backward(self, target):
        # l = 1/2(y' - y)^2
        # error = y' - y
        error = self.out - target
        # dL/dy' = error
        # dL/dz2 = dL/dy' * dy'/dz2 = error * sigmoid'(z2)
        d_out = error * self.out * (1 - self.out) # 输出层对损失的梯度

        # 计算隐藏层对损失的梯度
        for i in range(self.hidden_size):
            # 计算隐藏层第i个神经元对损失的梯度
            # z2 = hi * w2[i] + ...
            # dL/dh_i = dL/dz2 * dz2/dh_i = d_out * w2[i]
            # dL/dz1_i = dL/dh_i * dh_i/dz1_i = d_out * w2[i] * act_d(z1[i])
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i]) # 隐藏层对损失的梯度
            # 更新隐藏层到输出层的权重和偏置
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs = 200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x,y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += 0.5 * (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return losses



data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("Gelu", gelu, gelu_derivative)
]
results = {}
for name, act_fn, act_d_fn in configs:
    print(f"Training with activation function: {name}")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr = 0.1)
    losses = net.train(data)
    results[name] = losses

print("final loss comparsion")
for name, losses in results.items():
    print(f"{name}: {losses[-1]:.4f}")


class PReLUNetwork(ActivationNetwork):
    # 参数化 ReLU：f(z) = z (z>0)，alpha * z (z<=0)，alpha 与权重一起通过梯度下降学习
    # 整层共享一个 alpha，与 nn.PReLU() 的默认行为一致
    def __init__(self, alpha_init=0.01, hidden_size=8, lr=0.1):
        super().__init__(self._prelu, self._prelu_d, hidden_size, lr)
        self.alpha = alpha_init
        self.alpha_history = []

    def _prelu(self, z):
        return z if z > 0 else self.alpha * z

    def _prelu_d(self, z):
        return 1.0 if z > 0 else self.alpha

    def backward(self, target):
        # dL/dalpha = sum_i dL/dh_i * dh_i/dalpha，仅 z<=0 的神经元有贡献，此时 dh/dalpha = z
        # 必须在父类更新 w2 之前计算，这里用的是更新前的 w2
        d_out = (self.out - target) * self.out * (1 - self.out)
        grad_alpha = sum(
            d_out * self.w2[i] * self.z1[i] for i in range(self.hidden_size) if self.z1[i] <= 0
        )
        super().backward(target)
        self.alpha -= self.lr * grad_alpha
        self.alpha_history.append(self.alpha)

def evaluate(net, dataset):
    correct = 0
    loss = 0.0
    for x, y in dataset:
        pred = net.forward(x)
        loss += 0.5 * (pred - y) ** 2
        correct += (pred >= 0.5) == (y >= 0.5)
    return loss / len(dataset), correct / len(dataset)

import contextlib, io

def run_prelu_vs_leaky(epochs=200):
    train_data = make_circle_data(seed=42)
    test_data = make_circle_data(seed=7)
    models = {
        "Leaky ReLU (alpha=0.01 fixed)": ActivationNetwork(leaky_relu, leaky_relu_derivative, hidden_size=8, lr=0.1),
        "PReLU (alpha init 0.01)": PReLUNetwork(alpha_init=0.01),
        "PReLU (alpha init 0.25)": PReLUNetwork(alpha_init=0.25),
    }
    print("\n=== PReLU vs Leaky ReLU ===")
    for name, net in models.items():
        with contextlib.redirect_stdout(io.StringIO()):  # 屏蔽 train 里每轮的打印
            net.train(train_data, epochs=epochs)
        tr_loss, tr_acc = evaluate(net, train_data)
        te_loss, te_acc = evaluate(net, test_data)
        extra = f", learned alpha={net.alpha:.4f}" if isinstance(net, PReLUNetwork) else ""
        print(f"{name}: train loss={tr_loss:.4f} acc={tr_acc:.3f} | test loss={te_loss:.4f} acc={te_acc:.3f}{extra}")

run_prelu_vs_leaky()



import torch 
import torch.nn as nn
import torch.nn.functional as Functional

x = torch.randn(4, 10)

relu_out = Functional.relu(x)
gelu_out = Functional.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = Functional.silu(x)

logits = torch.randn(4, 5) # 4个样本在5个类别上的logits分数
probs = Functional.softmax(logits, dim = 1) # 计算每个样本在各个类别上的概率分布

# 输入 10 维度 
# 输出 5 维度 的 分类结果
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5)  # 输出形式的logits 分数
)
predict = model(x)


# Transformer 模型中的隐藏层：GELU。
# CNN 模型中的隐藏层：ReLU。
# 分类输出层：softmax。
# 回归输出层：无（线性）。
# 概率输出层：sigmoid。