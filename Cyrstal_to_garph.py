#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/7 16:30
# 软件: PyCharm

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.data import get_train_val_test_loader
from sat.data import GraphDataset
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")

# 从cgcnn中加载晶体数据
si_data_path = './dataset'
si_data = CIFData(si_data_path)
data_len = len(si_data)
print("有多少张图数据",data_len)

k_hop = 2
se = 'gnn'
use_edge_attr = True

# 开始转换数据
structures = [] #(atom_fea, nbr_fea, nbr_fea_idx)
target = []
cif_id = []
for i in tqdm(range(0,data_len)):
    structures.append(si_data[i][0])
    target.append(si_data[i][1])
    cif_id.append(si_data[i][2])
# 构建一个garph list来整合5000张图
geometric_data = []
for i in tqdm(range(0,data_len)):
    geometric_data.append(Data(x = structures[i][0],
                             edge_attr = structures[i][2],
                             edge_index = structures[2][1],
                             y = target[i]))
print('len of geometric_data',data_len)
print(geometric_data[0])
print("x的维度",geometric_data[0].x.shape)
print("edge_index的维度",geometric_data[0].edge_index.shape)
# 对图数据中的每一张图的edge_index进行标准化
for i in tqdm(range(0,data_len)):
    # 修改为long()格式
    geometric_data[i].edge_index = geometric_data[i].edge_index.long()
    # 降维成二维
    edge_index = geometric_data[i].edge_index
    edge_index = edge_index.view(-1, edge_index.size(-1))
    geometric_data[i].edge_index = edge_index

print(geometric_data[0].edge_index.shape)
si_graph_data = GraphDataset(geometric_data,degree=True, k_hop=k_hop, se=se,
       use_subgraph_edge_attr=use_edge_attr)

def c_to_g(g):
    structures = []  # (atom_fea, nbr_fea, nbr_fea_idx)
    target = []
    cif_id = []
    data_len = len(g)


    for i in tqdm(range(0, data_len)):
        # 修改为long()格式
        g[i].edge_index = g[i].edge_index.long()
        # 降维成二维
        edge_index = g[i].edge_index
        edge_index = edge_index.view(-1, edge_index.size(-1))
        g[i].edge_index = edge_index

    return g

collate_fn = collate_pool
# 数据集划分函数
# 输入完整数据集 + 划分参数
# 输出：返回 train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=si_data,
        collate_fn=collate_fn,
        batch_size=128,
        train_ratio=0.8,
        num_workers=1,
        val_ratio=0.1,
        test_ratio=0.1,
        pin_memory=True,
        return_test=True)

train_data = c_to_g(train_loader)
val_data = c_to_g(val_loader)
test_data = c_to_g(test_loader)

