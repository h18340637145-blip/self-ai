import tracemalloc
import random

import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 构造一个示例 Dataset
# =========================

class DemoDataset(Dataset):
    def __init__(self, size=1000, feature_dim=1024):
        self.size = size
        self.feature_dim = feature_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟读取原始数据
        x = torch.randn(self.feature_dim)

        # 模拟数据增强
        x = x * random.random()

        # 故意创建一个额外 Python list
        # 用来模拟不必要的内存分配
        temp_list = [float(v) for v in x[:500]]

        # 再转回 tensor
        extra_tensor = torch.tensor(temp_list)

        # 拼接，模拟数据处理流程
        x = torch.cat([x, extra_tensor])

        # 假设这是标签
        y = torch.tensor(idx % 10)

        return x, y


# =========================
# 2. 创建 DataLoader
# =========================

dataset = DemoDataset(
    size=1000,
    feature_dim=1024
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# =========================
# 3. tracemalloc 开始追踪
# =========================

tracemalloc.start()


# =========================
# 4. 加载前拍一次快照
# =========================

before = tracemalloc.take_snapshot()


# =========================
# 5. 模拟数据加载
# =========================

for step, batch in enumerate(dataloader):

    x, y = batch

    print(
        f"step={step}, "
        f"x.shape={x.shape}, "
        f"y.shape={y.shape}"
    )

    # 只跑几个 batch
    if step == 10:
        break


# =========================
# 6. 加载后再拍一次快照
# =========================

after = tracemalloc.take_snapshot()


# =========================
# 7. 对比内存变化
# =========================

stats = after.compare_to(
    before,
    "lineno"
)


print("\n========== Top 20 memory increases ==========\n")

for stat in stats[:20]:
    print(stat)


# =========================
# 8. 查看当前内存分配情况
# =========================

current_stats = after.statistics(
    "lineno"
)

print("\n========== Top 20 current allocations ==========\n")

for stat in current_stats[:20]:
    print(stat)


# =========================
# 9. 当前和峰值内存
# =========================

current, peak = tracemalloc.get_traced_memory()

print("\n========== Memory Summary ==========")

print(
    f"Current traced memory: "
    f"{current / 1024 / 1024:.2f} MB"
)

print(
    f"Peak traced memory: "
    f"{peak / 1024 / 1024:.2f} MB"
)


# =========================
# 10. 停止追踪
# =========================

tracemalloc.stop()