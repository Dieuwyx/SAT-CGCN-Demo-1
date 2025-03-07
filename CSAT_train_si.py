#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/7 22:48
# 软件: PyCharm

# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.utils as utils
from torch_geometric.data.remote_backend_utils import num_nodes

from csat.Crytal_data import CIFData
from csat.Crytal_data import collate_pool
from csat.Crytal_data import crystal_graph_list
from csat.Crytal_data import get_train_val_test_loader

from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES

from timeit import default_timer as timer
import argparse
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")



cuda = torch.cuda.is_available()
print("CUDA is", torch.cuda.is_available())

config = {
    "data_path": './sample-regression',
    "batch_size": 128,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "workers": 0,
    "cuda": cuda,
    "k_hop": 3,
    "se": 'gnn',
    "use_edge_attr": True,
    'seed': 0,
    'num_heads': 8,
    'num_layers': 6,
    'dim_hidden': 64,
    'dropout': 0.2,
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'abs_pe': None,
    'abs_pe_dim': 20,
    'warmup': 5000,
    'layer_norm': False,
    'edge_dim': 32,
    'gnn_type': 'graphsage',
    'global_pool': 'mean',
    'batch_norm':True,

}
def load_args():
    parser = argparse.ArgumentParser(description='Structure-Aware Transformer on ZINC')
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key} (default: {value})')
    args = parser.parse_args()


    if args.data_path is None:
        os.makedirs(args.data_path, exist_ok=True)
    args.batch_norm = not args.layer_norm

    return args


def main():
    global args
    args = load_args()
    # 加载数据集，输出：元组 ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    dataset = CIFData(args.data_path)
    structures, _, _ = dataset[0]
    # 原始atom_fea
    orig_atom_fea_len = structures[0].shape[-1]
    print('原始atom_fea长度：',orig_atom_fea_len)
    # 邻居特征长度nbr_fea
    nbr_fea_len = structures[1].shape[-1]
    print('邻居特征长度：',nbr_fea_len)
    dataset = crystal_graph_list(dataset)
    print('请检查一下：')
    print('graph数量：',len(dataset))
    print('每一张graph的数据类型为',dataset[0])
    # 将dataset转换为图数据
    graph_data = GraphDataset(dataset, degree=True, k_hop=args.k_hop, se=args.se,
                                use_subgraph_edge_attr=args.use_edge_attr)
    # 将数据列表整理成用于预测晶体的批处理性能，打包(atom_fea, nbr_fea, nbr_fea_idx, target)


    collate_fn = collate_pool
    # 数据集划分函数
    # 返回的是按照比例划分的graph数据集
    train_dset, val_dset, test_dset = get_train_val_test_loader(
                                                        dataset=graph_data,
                                                        batch_size=args.batch_size,
                                                        train_ratio=args.train_ratio,
                                                        num_workers=args.workers,
                                                        val_ratio=args.val_ratio,
                                                        test_ratio=args.test_ratio,
                                                        pin_memory=args.cuda,
                                                        return_test=True)
    # 加载数据
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,collate_fn=collate_fn,pin_memory=args.cuda)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers,collate_fn=collate_fn,pin_memory=args.cuda)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers,collate_fn=collate_fn,pin_memory=args.cuda)

    # 计算角度
    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])

    model = GraphTransformer(in_size=orig_atom_fea_len,
                             num_class=1,
                             d_model=args.dim_hidden,
                             dim_feedforward=2 * args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=nbr_fea_len,
                             edge_dim=args.edge_dim,
                             k_hop=args.k_hop,
                             se=args.se,
                             deg=deg,
                             global_pool=args.global_pool)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if __name__ == "__main__":
    main()
