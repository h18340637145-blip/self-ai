import math

def sigmoid(x):
    x = max(-500, min(500,x)) # [-500, 500]
    return 1 / (1 + math.exp(-x))

# deeplearning 中最重要的是矩阵乘法
# 每一层，每一个注意力头，每一次前向传播，都离不开矩阵乘法
# 线性层接收一个输入向量，经过权重矩阵乘法和偏置加法，输出一个新的向量  这总操作占据了深度学习计算的大部分时间

# 因此，每一层都包含一个权重矩阵和一个偏置向量
# 他的前向传播方法就是 输入向量乘以权重矩阵，再加上偏置向量，然后通过激活函数输出新的向量
# 这就是深度学习中每一层的基本计算过程

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases = None):
        # 给权重就用给定的权重，否则随机初始化ai
        # 给偏置就用给定的偏置，否则初始化为0
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0 for _ in range(n_neurons)]
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0 for _ in range(n_neurons)]

    def forward(self, inputs):
        self.last_inputs = inputs # x
        self.last_output = []
        for neuron_idx in range(len(self.weights)): # 根据笔记中记录的，每一个输出的神经元，都是由输入向量和对应的权重计算得到的（所有的权重和输入）
            z = sum(
                w * x
                for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z)) # 将每个神经元的输出需要做一次激活函数处理

        return self.last_output

# 网络： 若干个层叠加在一起，形成一个多层感知机（MLP）
# 网络的前向传播，就是将输入向量依次通过每一层（对于层就是调用它的 forward 方法），得到最终的输出向量
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

# 使用手动调整的权重进行异或运算
# 使用layer和network 来解决异或问题
# 构建一个两层的网络，隐藏层有两个神经元，输出层有一个神经元 2 2 1架构
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[
        [20, 20],
        [-20, -20]
    ],
    biases=[-10, 30]
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[
        [20, 20]
    ],
    biases=[-30]
)

xor_network = Network([hidden, output])

xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]
for inputs, expected in xor_data:
    result = xor_network.forward(inputs)
    predicted = [1 if r > 0.5 else 0 for r in result]
    print(f"Input: {inputs}, Expected: {expected}, Predicted: {predicted}")


import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y ) < 0.25 else 0
    data.append(([x, y], [label]))

# 输入 2 个特征，隐藏层 8 个神经元，输出层 1 个神经元表示再或者不在内部
circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1)
])

correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected[0]:
        correct += 1

print(f"Accuracy: {correct / len(data)}")

# 使用 pytorch 实现
import torch
import torch.nn as nn

# Sequential 模型实现一个两层的神经网络，输入 2 个特征，隐藏层 8 个神经元，输出层 1 个神经元
# 与自己实现的 Network 类似，都是两层的神经网络结构
# 权重和偏置 会在模型初始化时自动生成，默认是随机初始化
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
# 元素必须是浮点数（float）类型
# 构建输入张量，表示四种可能的输入组合
x = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
, dtype=torch.float32
)
output = model(x)
print(output)


# 构建一个2 4 2 1 网络 使用随机权重对xor数据进行前向传播
xor_net = Network([
    Layer(n_inputs=2, n_neurons=4),
    Layer(n_inputs=4, n_neurons=2),
    Layer(n_inputs=2, n_neurons=1)
])

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = [1 if r > 0.5 else 0 for r in result]
    print(f"Input: {inputs}, Expected: {expected}, Predicted: {predicted}")
    print()  # 添加空行以便输出更清晰

# 将圆形分类器中的隐藏层神经元从 8 改为 2， 然后再改成 32 每次都是用随机权重前向传播
# 隐藏层神经元数量变换，输出的范围可能会有所不同
for hidden_neurons in [2, 32]:
    circle_net = Network([
        Layer(n_inputs=2, n_neurons=hidden_neurons),
        Layer(n_inputs=hidden_neurons, n_neurons=1)
    ])
    correct = 0
    for inputs, expected in data:
        result = circle_net.forward(inputs)  
        # 根据前向传播的结果判断预测类别 0.5 为阈值
        predicted = 1 if result[0] >= 0.5 else 0
        if predicted == expected[0]: # 如果预测正确，计数加一
            correct += 1
    print(f"Hidden neurons: {hidden_neurons}, Accuracy: {correct / len(data)}")