
class Perceptron:
    def __init__(self, n_inputs, learning_rate):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0
    def train(self, train_data, n_epochs):
        for epoch in range(n_epochs):
            errors = 0 # 计算当前epoch的错误数
            for inputs, target in train_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"converged at epoch {epoch + 1}")
                return 
        print(f"did not converge after {n_epochs} epochs")

and_data = [
    ([0,0], 0),
    ([0,1], 0),
    ([1,0], 0),
    ([1,1], 1),
]

or_data = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]
print(f"---- AND ---- ")
p_and = Perceptron(n_inputs=2, learning_rate=0.1)
p_and.train(and_data, n_epochs=100)
for inputs, _ in and_data:
    print(f" {inputs} -> {p_and.predict(inputs)}")

p_or = Perceptron(n_inputs=2, learning_rate=0.1)
p_or.train(or_data, n_epochs=100)
for inputs, _ in or_data:
    print(f" {inputs} -> {p_or.predict(inputs)}")

print(f"---- NOT ---- ")
p_not = Perceptron(n_inputs=1, learning_rate=0.1)
p_not.train(not_data, n_epochs=100)
for inputs, _ in not_data:
    print(f" {inputs} -> {p_not.predict(inputs)}")


# xor
xor_data = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 0),
]

# 单个感知机无法解决异或问题
print(f"---- XOR ---- ")
p_xor = Perceptron(n_inputs=2, learning_rate=0.1)
p_xor.train(xor_data, n_epochs=100)
for inputs, _ in xor_data:
    print(f" {inputs} -> {p_xor.predict(inputs)}")

# 使用两层感知机解决异或问题
# 异或== ((x1 or x2) and not (x1 and x2))
def xor_network(x1, x2):
    '''
| 输入 | OR | NAND | 最终 XOR |
|---|---:|---:|---:|
| `[0, 0]` | 0 | 1 | 0 |
| `[0, 1]` | 1 | 1 | 1 |
| `[1, 0]` | 1 | 1 | 1 |
| `[1, 1]` | 1 | 0 | 0 |
    '''
    # first layer
    or_neuron = Perceptron(n_inputs=2, learning_rate=0.1)  # z = w1x1 + w2x2 + b
    or_neuron.weights = [1.0, 1.0] # or 使用1 1  表示任何一个输入为 1 ， 都应该促进输出 1
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(n_inputs=2, learning_rate=0.1)
    nand_neuron.weights = [-1.0, -1.0] # 输入中的 1 越多，越应该抑制输出 1。
    nand_neuron.bias = 1.5

    # second layer
    and_neuron = Perceptron(n_inputs=2, learning_rate=0.1)
    and_neuron.weights = [1.0, 1.0] # 两个输入都为 1 时，应该促进输出 1
    and_neuron.bias = -1.5

    # forward pass
    or_output = or_neuron.predict([x1, x2]) # hidden 1
    nand_output = nand_neuron.predict([x1, x2]) # hidden 2
    return and_neuron.predict([or_output, nand_output]) # output layer

print("---- XOR Network ---- ")
for inputs, excepted in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f" {inputs} -> {result} (expected: {excepted})")



# 训练一个双层网络
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.1):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1) , random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1) , random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1) , random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        # 计算隐藏层的输出
        for i in range(2): # 遍历隐藏层的每个神经元，计算其输出
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_output = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_output)
        return self.output

    def train(self, training_data, nepoch=10000):
        for epoch in range(nepoch):
            total_error = 0
            for inputs, expected in training_data:
                output = self.forward(inputs)
                error = expected - output
                total_error += error ** 2
                # 反向传播输出层
                delta_output = error * output * (1 - output)
                # 反向传播隐藏层
                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = h * (1 - h) * delta_output * saved_w_output[i]
                    hidden_deltas.append(hd)
                for i in range(2):
                    self.w_output[i] += self.lr * delta_output * self.hidden_outputs[i]
                self.b_output += self.lr * delta_output
                for i in range(2):
                    for j in range(2):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * self.inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")


net = TwoLayerNetwork()
net.train(xor_data, nepoch=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f" {inputs} -> {result} (predicted: {predicted}, expected: {expected})")


