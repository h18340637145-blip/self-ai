
import torch
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

a = x1 * x2 # a = 6
b = a + 1  # b = 7

t = torch.relu(b) # t = 7
t.backward()  # 进行反向传播，计算梯度
print(f"x1.grad = {x1.grad}")  # 应该是 3.0
print(f"x2.grad = {x2.grad}")  # 应该是 2.0


x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
w1 = torch.tensor(4.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
y = torch.relu(w1 * x1 + w2 * x2 + b)  # 构建计算图
y.backward()  # 进行反向传播，计算梯度
print(f"w1.grad = {w1.grad}")  # 应该是 x1 的值 2
print(f"w2.grad = {w2.grad}")  # 应该是 x2 的值 3
print(f"b.grad = {b.grad}")    # 应该是 1
print(f"x1.grad = {x1.grad}")  # 应该是 w1 的值 4
print(f"x2.grad = {x2.grad}")  # 应该是 w2 的值 -1