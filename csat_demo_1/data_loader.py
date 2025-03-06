#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/6 17:03
# 软件: PyCharm


import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse


class CrystalDataset(Dataset):
    def __init__(self, root, csv_path, radius=5.0, max_neighbors=12, k_hop=2):
        """
        root: 数据根目录
        csv_path: 包含结构和target的csv文件路径
        radius: 邻域原子搜索半径
        k_hop: SAT的跳数
        """
        self.df = pd.read_csv(os.path.join(root, csv_path))
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.k_hop = k_hop
        super().__init__(root)

    @property
    def raw_file_names(self):
        return [f"{id}.cif" for id in self.df['material_id']]

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(len(self.df))]

    def process(self):
        for idx, row in self.df.iterrows():
            # 使用pymatgen解析晶体结构
            struct = Structure.from_file(os.path.join(self.raw_dir, f"{row['material_id']}.cif"))

            # 构建原子特征
            atom_features = [self._get_atom_feature(site.specie) for site in struct]
            x = torch.tensor(atom_features, dtype=torch.float)

            # 构建邻接矩阵
            adj = struct.get_all_neighbors(self.radius, include_index=True)
            edge_index, edge_attr = self._build_edges(struct, adj)

            # 提取k-hop子图 (SAT核心)
            subgraph_data = self._extract_khop_subgraphs(edge_index, len(struct))

            # 创建PyG Data对象
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([row['target']], dtype=torch.float),
                **subgraph_data
            )
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def _get_atom_feature(self, specie):
        # 原子特征：电负性+原子半径+价电子数（示例特征）
        return [
            specie.electronegativity,
            specie.atomic_radius,
            specie.Z
        ]

    def _build_edges(self, struct, adj):
        # 构建边特征（键距+矢量）
        edge_indices = []
        edge_attrs = []
        for i, neighbors in enumerate(adj):
            for neighbor in neighbors[:self.max_neighbors]:
                dist = neighbor[1]
                vec = struct[i].coords - neighbor[0].coords
                edge_indices.append([i, neighbor[2]])
                edge_attrs.append([dist, *vec])

        edge_index = torch.tensor(edge_indices).T.contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return edge_index, edge_attr

    def _extract_khop_subgraphs(self, edge_index, num_nodes):
        # SAT的k-hop子图提取（简化的内存实现）
        from torch_geometric.utils import k_hop_subgraph

        sub_nodes = []
        sub_edge_indices = []
        indicators = []
        edge_start = 0

        for node in range(num_nodes):
            nodes, sub_edge, _, _ = k_hop_subgraph(
                node_idx=node,
                num_hops=self.k_hop,
                edge_index=edge_index,
                relabel_nodes=True
            )
            sub_nodes.append(nodes)
            sub_edge_indices.append(sub_edge + edge_start)
            indicators.append(torch.full((len(nodes),), node))
            edge_start += len(nodes)

        return {
            'subgraph_node_idx': torch.cat(sub_nodes),
            'subgraph_edge_index': torch.cat(sub_edge_indices, dim=1),
            'subgraph_indicator': torch.cat(indicators),
            'num_subgraph_nodes': edge_start
        }

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
