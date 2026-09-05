import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 1. 准备数据
# =========================

# 1000 个样本
# 每个样本 20 个特征
X = torch.randn(1000, 20)

# 10 个类别
y = torch.randint(
    low=0,
    high=10,
    size=(1000,)
)

dataset = TensorDataset(X, y)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# =========================
# 2. 设备
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("device:", device)


# =========================
# 3. 模型
# =========================

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleModel().to(device)


# =========================
# 4. Loss + Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 5. 训练循环
# =========================

for epoch in range(2):

    model.train()

    for step, (inputs, targets) in enumerate(dataloader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        # =========================
        # breakpoint 1
        # 检查 forward 之后的数据
        # =========================

        if step == 0:
            breakpoint()

        loss.backward()

        # =========================
        # breakpoint 2
        # 检查 backward 之后的梯度
        # =========================

        if step == 0:
            breakpoint()

        optimizer.step()

        print(
            f"epoch={epoch}, "
            f"step={step}, "
            f"loss={loss.item():.4f}"
        )

        # 为了练习，不需要跑完整个 epoch
        if step >= 2:
            break