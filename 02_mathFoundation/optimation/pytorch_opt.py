import torch
model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

adam = torch.optim.Adam(model.parameters(), lr = 0.001)

adamw = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)

print(sgd)
print(adam)
print(adamw)
print(scheduler)
