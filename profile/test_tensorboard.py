import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


# =========================
# 1. 设备
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# =========================
# 2. 数据集
# =========================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

val_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# =========================
# 3. 定义模型
# =========================

class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                28 * 28,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        return self.net(x)


model = SimpleNet().to(device)

print(model)


# =========================
# 4. Loss 和 Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 5. TensorBoard Writer
# =========================

writer = SummaryWriter(
    log_dir="runs/mnist_experiment"
)


# =========================
# 6. 记录模型结构
# =========================

example_batch, _ = next(
    iter(train_loader)
)

example_batch = example_batch.to(device)

writer.add_graph(
    model,
    example_batch
)


# =========================
# 7. 训练参数
# =========================

epochs = 10

global_step = 0


# =========================
# 8. 开始训练
# =========================

for epoch in range(epochs):

    # -------------------------
    # Train
    # -------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        # 累计 loss
        train_loss += (
            loss.item()
            * images.size(0)
        )


        # 计算 accuracy
        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += (
            labels.size(0)
        )

        train_correct += (
            predicted == labels
        ).sum().item()


        # 每个 batch 记录 loss
        writer.add_scalar(
            "Loss/train_batch",
            loss.item(),
            global_step
        )

        global_step += 1


    train_loss /= len(
        train_loader.dataset
    )

    train_accuracy = (
        train_correct
        / train_total
    )


    # =========================
    # 9. Validation
    # =========================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += (
                labels.size(0)
            )

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss /= len(
        val_loader.dataset
    )

    val_accuracy = (
        val_correct
        / val_total
    )


    # =========================
    # 10. 写入 TensorBoard
    # =========================

    writer.add_scalars(
        "Loss",
        {
            "train": train_loss,
            "validation": val_loss
        },
        epoch
    )


    writer.add_scalars(
        "Accuracy",
        {
            "train": train_accuracy,
            "validation": val_accuracy
        },
        epoch
    )


    # =========================
    # 11. 控制台输出
    # =========================

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


# =========================
# 12. 关闭 TensorBoard
# =========================

writer.close()