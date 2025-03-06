#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/6 17:03
# 软件: PyCharm

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from data_loader import CrystalDataset
from sat_model import SAT4Crystals

# 超参数
config = {
    'batch_size': 64,
    'hidden_dim': 128,
    'num_layers': 4,
    'lr': 1e-3,
    'epochs': 100,
    'k_hop': 2
}

# 数据准备
dataset = CrystalDataset(
    root='./dataset',
    radius=5.0,
    k_hop=config['k_hop']
)

train_loader = DataLoader(dataset[:800], batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(dataset[800:900], batch_size=config['batch_size'])
test_loader = DataLoader(dataset[900:], batch_size=config['batch_size'])

# 模型初始化
model = SAT4Crystals(
    node_dim=3,
    edge_dim=4,
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    k_hop=config['k_hop']
)

# 优化设置
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()

# 训练循环
for epoch in range(config['epochs']):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch)
            val_loss += criterion(pred, batch.y).item()

    print(
        f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

# 测试
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch)
        test_loss += criterion(pred, batch.y).item()
print(f"Final Test MAE: {test_loss / len(test_loader):.4f}")
