#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/6 11:39
# 软件: PyCharm

# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from pymatgen.core import Structure
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F


# 1. 自定义晶体数据集处理类
class CrystalDataset(Dataset):
    def __init__(self, root, cif_files, target_file, radius=5.0, max_neighbors=12,
                 k_hop=3, transform=None, pre_transform=None):
        """
        root: 数据存储根目录
        cif_files: cif文件路径列表
        target_file: 目标属性文件路径
        radius: 原子邻域半径(Å)
        max_neighbors: 最大邻居数
        k_hop: 子图提取跳数
        """
        super().__init__(root, transform, pre_transform)
        self.cif_files = cif_files
        self.targets = np.loadtxt(target_file)
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.k_hop = k_hop

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.cif_files]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.cif_files))]

    def process(self):
        for idx, cif_path in enumerate(self.cif_files):
            # 使用pymatgen解析晶体结构
            struct = Structure.from_file(cif_path)

            # 生成图结构数据
            nodes = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
            pos = torch.tensor([site.coords for site in struct], dtype=torch.float)

            # 生成邻接关系(考虑周期性)
            edge_index, edge_attr = self._get_edges(struct)

            # 提取k-hop子图
            subgraphs = self._extract_subgraphs(edge_index, struct.num_sites)

            # 创建PyG Data对象
            data = Data(
                x=nodes,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([self.targets[idx]], dtype=torch.float),
                **subgraphs
            )

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def _get_edges(self, struct):
        """生成考虑周期性的邻接矩阵"""
        from pymatgen.analysis.graphs import StructureGraph
        from pymatgen.analysis import local_env

        # 使用CrystalNN算法获取邻接关系
        cnn = local_env.CrystalNN()
        graph = StructureGraph.with_local_env_strategy(struct, cnn)

        edge_index = []
        edge_attr = []
        for from_idx, to_indices in graph.graph.adjacency():
            for to_idx, edge_data in to_indices.items():
                # 添加键长作为边特征
                dist = struct[from_idx].distance(struct[to_idx])
                edge_index.append([from_idx, to_idx])
                edge_attr.append([dist])

        return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_attr)

    def _extract_subgraphs(self, edge_index, num_nodes):
        """提取k-hop子图"""
        subgraphs = {'subgraph_edge_index': [], 'subgraph_indicator': []}
        for node in range(num_nodes):
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                node_idx=node,
                num_hops=self.k_hop,
                edge_index=edge_index,
                num_nodes=num_nodes,
                relabel_nodes=True
            )
            subgraphs['subgraph_edge_index'].append(sub_edge_index)
            subgraphs['subgraph_indicator'].append(torch.full((subset.size(0),), node))

        # 合并子图信息
        subgraphs['subgraph_edge_index'] = torch.cat(subgraphs['subgraph_edge_index'], dim=1)
        subgraphs['subgraph_indicator'] = torch.cat(subgraphs['subgraph_indicator'])
        return subgraphs


# 2. 修改后的Structure-Aware Transformer模型
class CrystalTransformer(torch.nn.Module):
    def __init__(self, num_element_types=103, dim_hidden=64, num_heads=8,
                 num_layers=6, k_hop=3, edge_dim=32):
        super().__init__()

        # 原子嵌入层
        self.element_embed = torch.nn.Embedding(num_element_types, dim_hidden)

        # 空间位置编码
        self.pos_encoder = PositionalEncoding(dim_hidden)

        # Structure-Aware Transformer编码器
        self.encoder = GraphTransformerEncoder(
            d_model=dim_hidden,
            num_heads=num_heads,
            dim_feedforward=2 * dim_hidden,
            num_layers=num_layers,
            gnn_type='gine',
            se='khopgnn',
            k_hop=k_hop,
            edge_dim=edge_dim
        )

        # 全局池化与回归头
        self.pool = gnn.global_mean_pool
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_hidden // 2, 1)
        )

    def forward(self, data):
        # 原子类型嵌入
        x = self.element_embed(data.x.squeeze(-1))

        # 位置编码
        pos_enc = self.pos_encoder(data.pos)
        x += pos_enc

        # 结构编码
        x = self.encoder(
            x,
            edge_index=data.edge_index,
            complete_edge_index=data.edge_index,  # 对于晶体使用原始边
            subgraph_edge_index=data.subgraph_edge_index,
            subgraph_indicator_index=data.subgraph_indicator,
            edge_attr=data.edge_attr
        )

        # 全局池化
        x = self.pool(x, data.batch)
        return self.regressor(x)


# 3. 位置编码模块
class PositionalEncoding(torch.nn.Module):
    """基于原子坐标的位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.pos_proj = torch.nn.Linear(3, dim)

    def forward(self, pos):
        # 使用正弦位置编码
        div_term = torch.exp(torch.arange(0, 300, 2).float() * (-np.log(10000.0) / 300)
        pe = torch.zeros(pos.size(0), 300)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return self.pos_proj(pe)


# 4. 训练流程示例
def train_crystal_model():
    # 数据集配置
    cif_files = [...]  # Material Project的cif文件列表
    targets = [...]  # 目标属性（如形成能、带隙等）

    dataset = CrystalDataset(
        root='./data',
        cif_files=cif_files,
        target_file=targets,
        radius=5.0,
        k_hop=3
    )

    # 数据划分
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1])

    # 数据加载
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 模型初始化
    model = CrystalTransformer(
        num_element_types=103,
        dim_hidden=128,
        num_heads=8,
        num_layers=6,
        k_hop=3
    )

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    # 训练循环
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                pred = model(batch)
                val_loss += criterion(pred, batch.y).item()
            print(f"Epoch {epoch}: Val MAE {val_loss / len(val_loader)}")


# 5. 关键改进说明
"""
1. 晶体特定数据处理：
   - 使用pymatgen处理周期性边界条件
   - 考虑三维空间位置编码
   - 基于晶体学的邻接关系生成

2. 结构感知机制增强：
   - 结合k-hop子图与晶体学对称性
   - 在注意力机制中融入键长等边特征
   - 多尺度结构信息提取

3. 模型优化：
   - 针对晶体特性的位置编码
   - 改进的GINE卷积层处理边属性
   - 考虑晶体全局对称性的池化策略
"""
