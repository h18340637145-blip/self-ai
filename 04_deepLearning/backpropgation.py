
import math

class Value:
    '''
    一个表示标量值的类，支持自动求导。
    每个操作会创建一个新的Value对象，并记录如何反向传播梯度。
    data: 当前数值
    grad: 最终结果对这个数的导数
    _backward: 把梯度传回参与计算的输入
    '''
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    # 反向函数运算  每个操作会创建一个新的值，记录定义梯度如何反向流经该值
    def __add__(self, other):
        # eg： out = a + b   self = a, other = b
        # 对a 和对 b 的导数都是 1 所以结束把结果的梯度分别传给两个输入 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # 定义反向传播函数，计算梯度 
        def _backward():
            # 对加法操作，梯度直接传递给两个操作数
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')


        # 对乘法操作，梯度根据乘法法则传递给两个操作数
        def _backward():
            # 乘法法则：对self的梯度是other.data乘以输出的梯度，对other的梯度是self.data乘以输出的梯度
            # a 的梯度是 other.data * out.grad b
            # b 的梯度是 self.data * out.grad a
            # 比如： a = 2 , b = 3, out = 6    out.grad = 1  反向传播之后 a.grad = 3, b.grad = 2
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # 减法操作，类似加法，但对第一个操作数的梯度是1，对第二个操作数的梯度是-1
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward = _backward
        return out

    # 取负操作，等价于乘以-1
    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out
    
    import math
    def sigmoid(self):
        x = self.data
        x = max(-500, min(500, x))
        s = 1 / (1 + math.exp(-x))

        out = Value(s , (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    # relu
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad # ReLU的梯度，只有当输入大于0时才传递梯度, 且为1
        out._backward = _backward
        return out
        
    # 后传
    def backward(self):
        #√ 拓扑排序确保我们按正确的顺序处理节点——在传播到节点之前，节点的梯度会被完全累积。
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    
# 均方误差 预测-目标的平方
def mse_loss(predicted, target):
    diff = predicted + Value(-target)
    return diff * diff

# 分层和网络
import random 
class Neuron:
    def __init__(self, n_inputs):
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
    def __call__(self, x):
        # 计算神经元的输出
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.sigmoid()
    
    # 获取神经元的所有参数，包括权重和偏置
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    #  本层都是由 n_inputs 个输入组成，每个神经元都有自己的权重和偏置 
    # 本层有 n_neurons 个神经元，每个神经元都有自己的权重和偏置ß
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, x):
        # 计算层的输出，逐个神经元计算
        out = [n(x) for n in self.neurons]
        return out
    # 获取本层所有神经元的参数
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class Network:
    # 网络由多层组成，每层都是一个Layer实例
    def __init__(self, sizes): # sizes 是一个列表，表示每层的神经元数量，例如 [2, 3, 1] 表示输入层有 2 个神经元，隐藏层有 3 个神经元，输出层有 1 个神经元
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))
    # 前向传播，依次通过每一层计算输出  call就是计算输出
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if not isinstance(x, list):
            x = [x]
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    # 将网络中所有参数的梯度清零
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# 进行异或运算训练
random.seed(42)
net = Network([2, 4, 1])
# xor data
xor_data = [
    ([0, 0], 0), 
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]
learning_rate = 0.1
for epoch in range(10000):
    total_loss = Value(0)
    for inputs, target in xor_data:
        # forward
        x = [Value(i) for i in inputs]
        pred = net(x)  # 前向传播计算预测值，返回列表
        loss = mse_loss(pred[0], target) # pred 是一个列表，取第一个元素作为输出值 
        # 为什么取第一个元素？因为网络输出层只有一个神经元，输出值在列表的第一个位置
        total_loss += loss
    # 反向传播计算梯度并更新参数
    net.zero_grad()
    total_loss.backward()
    for p in net.parameters(): # 更新每个参数的值，根据梯度下降法
        p.data = p.data - learning_rate * p.grad
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.data}")

# 测试训练后的网络
for inputs, target in xor_data:
    x = [Value(i) for i in inputs]
    pred = net(x)
    print(f"Input: {inputs}, Predicted: {pred[0].data}, Target: {target}")