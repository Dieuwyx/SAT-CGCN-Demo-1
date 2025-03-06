#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/6 17:03
# 软件: PyCharm

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool


class SAT4Crystals(nn.Module):
    def __init__(self,
                 node_dim=3,
                 edge_dim=4,
                 hidden_dim=64,
                 num_heads=8,
                 num_layers=4,
                 k_hop=2):
        super().__init__()

        # 节点/边嵌入
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)

        # SAT编码器层
        self.layers = nn.ModuleList([
            SATLayer(hidden_dim, edge_dim, num_heads, k_hop)
            for _ in range(num_layers)
        ])

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        # SAT结构感知编码
        for layer in self.layers:
            x = layer(
                x,
                data.edge_index,
                data.subgraph_edge_index,
                data.subgraph_indicator,
                edge_attr
            )

        # 全局池化
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze()


class SATLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads, k_hop):
        super().__init__()
        self.k_hop = k_hop

        # 结构提取器
        self.struct_extractor = GNNBlock(hidden_dim, edge_dim)

        # Transformer注意力
        self.attn = TransformerConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim
        )

    def forward(self, x, edge_index, subgraph_edge, subgraph_indicator, edge_attr):
        # 结构特征提取 (k-hop)
        struct_feat = self.struct_extractor(
            x,
            subgraph_edge,
            edge_attr=edge_attr[subgraph_indicator],
            indicator=subgraph_indicator
        )

        # 拼接原始特征
        x = torch.cat([x, struct_feat], dim=1)

        # 结构增强的注意力
        return self.attn(x, edge_index, edge_attr=edge_attr)


class GNNBlock(nn.Module):
    """k-hop子图处理模块"""

    def __init__(self, dim, edge_dim):
        super().__init__()
        self.conv1 = TransformerConv(dim, dim // 4, edge_dim=edge_dim, heads=4)
        self.conv2 = TransformerConv(dim, dim // 4, edge_dim=edge_dim, heads=4)

    def forward(self, x, edge_index, edge_attr, indicator):
        # 子图处理
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)

        # 聚合到原始图
        return scatter(x, indicator, dim=0, reduce='mean')
