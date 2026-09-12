

# 每个节点都是 Value 存储其数值数据 梯度 以及计算图中的父节点和操作类型
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data # 保存数值数据
        self.grad = 0.0 # 保存梯度信息 损失函数对节点的导数
        self._backward = lambda: None # 反向传播函数 这个节点的梯度如何影响其父节点，如何传递给前面的节点
        self._prev = set(children) # 父节点集合  记录当前节点是由哪些父节点计算得到的
        self._op = op # 操作类型

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# 使用梯度跟踪进行算术运算
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 输出节点   数据是当前点+其他节点的数据   当前值由 self.data + other.data 得到   是加法操作的结果
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 来源多条路径，所以用+=
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        # 输出节点   数据是当前点*其他节点的数据   当前值由 self.data * other.data 得到   是乘法操作的结果
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 来源多条路径，所以用+=
            '''
            eg: c = a * b
                dc/da = b
                dc/db = a
                dc/dc = 1

                这里是 self 是 a  dL/da = dL/dc * dc/da = out.grad * other.data
                这里是 other 是 b  dL/db = dL/dc * dc/db = out.grad * self.data
            '''
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # 后传  拓扑排序确保每个节点的梯度在传播到其子节点之前都已完全计算处来，种子梯度为 1
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # 这里可以添加其他操作的反向传播逻辑，例如减法、除法、指数等操作
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        l = math.log(self.data)
        out = Value(l, (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out



# 有了Value类，就可以构建神经网络，无需pytorch numpy 只需要Value类和链式法则即可实现自动微分

"""
一个 Neuron 神经元进行计算，A 是一个神经元列表， A 堆叠层， 每个权重都是一个 Value 对象，因此调用会触发链式法则进行梯度传播
"""
import random
# 定义神经网络的基本单元：神经元和层
# 每个神经元包含若干输入权重和一个偏置，激活函数为tanh
class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    # 返回神经元的所有可训练参数（权重和偏置）
    def parameters(self):
        return self.w + [self.b]
    
# 定义神经网络的层，包含多个神经元
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    # 返回层中所有神经元的可训练参数
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
# 定义多层感知机（MLP），由多层神经网络组成
class MLP:
    def __init__(self, size):
        self.layer = []
        for i in range(len(size) - 1):
            self.layer.append(Layer(size[i], size[i + 1]))

    def __call__(self, x):
        for layer in self.layer:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    # 返回多层感知机中所有层的可训练参数
    def parameters(self):
        return [p for layer in self.layer for p in layer.parameters()]



# xor 培训
random.seed(42)  # 设置随机种子以确保可重复性
model = MLP([2, 4, 1])  # 定义一个简单的 XOR 网络，输入层2个神经元，隐藏层4个神经元，输出层1个神经元

xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
ys = [-1, 1, 1, -1]  # XOR 的输出

for step in range(100):
    preds = [model(x) for x in xs] # 计算每个输入的预测值
    # 计算损失 (均方误差)
    loss = sum((pred - y) ** 2 for pred, y in zip(preds, ys))
    # 反向传播
    for p in model.parameters():
        p.grad = 0.0 # 重置梯度
    loss.backward() # 进行反向传播，计算梯度
    # 更新参数（梯度下降）
    learning_rate = 0.01
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.data}")
print(f"predictions: after training:")
for x, y in zip(xs, ys):
    print(f"input = {x}, target = {y}, prediction = {model(x)}")


# 梯度检查
# 如何判断自动微分的结果是否正确  将其与数值导数进行比较
def gradient_check(build_expr, x_val, h = 1e-7):
    x = Value(x_val)
    y = build_expr(x)

    y.backward() # 进行反向传播，计算梯度 
    autodiff_grad = x.grad # 自动微分计算得到的梯度 直接将导数结果获取

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data

    numerical_grad = (y_plus - y_minus) / (2 * h)
    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
# 用复杂表达式测试一下
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)

print(f" Auto diff : {ad: .8f}")
print(f" Numerical grad : {num: .8f}")
print(f" Difference : {diff: .8f}")

# 与手动计算结果进行核对
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2  # a = 6
b = a + Value(1.0)  # b = 7
t = b.relu()  # t = 7

t.backward()  # 进行反向传播，计算梯度
print(f"x1.grad = {x1.grad}")  # 应该是 3.0
print(f"x2.grad = {x2.grad}")  # 应该是 2.0


aa = Value(2.0)
bb = Value(-3)
cc = Value(10)
f = (aa * bb + cc).relu()
f.backward()  # 进行反向传播，计算梯度
print(f"aa.grad = {aa.grad}")  # 应该是 -3.0
print(f"bb.grad = {bb.grad}")  # 应该是 2.0
print(f"cc.grad = {cc.grad}")  # 应该是 1.0

x = Value(2.0)
def fx(x):
    return x ** 3

y = fx(x)
y.backward()  # 进行反向传播，计算梯度
print(f"x.grad = {x.grad}")  # 应该是 12.0


# test2
def fx2(x):
    return x.tanh()
    
x2 = Value(0)
x3 = Value(2)
y2 = fx2(x2)
y3 = fx2(x3)
y2.backward()  # 进行反向传播，计算梯度
print(f"x2.grad = {x2.grad}")  # 应该是 fx2 对 x 的导数
y3.backward()  # 进行反向传播，计算梯度
print(f"x3.grad = {x3.grad}")  # 应该是 fx2 对 x 的导数 


# test3 
# 为单个神经元构建计算图：y = relu(w1x1 + w2x2 + b)
w1 = Value(4)
w2 = Value(-1)
b = Value(2)

x1 = Value(2)
x2 = Value(3)

y = (w1 * x1 + w2 * x2 + b).relu()  # 构建计算图
y.backward()  # 进行反向传播，计算梯度
print(f"w1.grad = {w1.grad}")  # 应该是 x1 的值 2
print(f"w2.grad = {w2.grad}")  # 应该是 x2 的值 3
print(f"b.grad = {b.grad}")    # 应该是 1
print(f"x1.grad = {x1.grad}")  # 应该是 w1 的值 4
print(f"x2.grad = {x2.grad}")  # 应该是 w2 的值 -1
