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
# from torch.utils.data import DataLoader
import torch_geometric.utils as utils
from torch_geometric.data.remote_backend_utils import num_nodes

from csat_demo_2.Crytal_data import CIFData
from csat_demo_2.Crytal_data import collate_pool
from csat_demo_2.Crytal_data import crystal_graph_list
from csat_demo_2.Crytal_data import get_train_val_test_loader

from csat_demo_2.models import GraphTransformer
from csat_demo_2.data import GraphDataset
from csat_demo_2.utils import count_parameters
from csat_demo_2.position_encoding import POSENCODINGS
from csat_demo_2.gnn_layers import GNN_TYPES
from csat_demo_2.data import custom_collate_fn

from timeit import default_timer as timer
from collections import defaultdict
import argparse
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


print("CUDA is", torch.cuda.is_available())
# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用 GPU
    cuda = torch.cuda.is_available()
else:
    device = torch.device('cpu')   # 使用 CPU
####################################################################################全部拿cpu测试一下
device = torch.device('cpu')   # 使用 CPU

print("Using device:", device)

config = {
    "data_path": './sample-regression',
    "batch_size": 1,

    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    "workers": 0,
    "cuda": False,

    "k_hop": 3,
    "se": 'khopgnn', # 一定要用khopgnn用来计算子图而非子树
    "use_edge_attr": False,

    'seed': 0,
    'num_heads': 8,
    'num_layers': 6,
    'dim_hidden': 64,
    'dropout': 0.2,

    'lr': 0.001,
    'weight_decay': 1e-5,
    'abs_pe': None,
    'abs_pe_dim': 20,
    'warmup': 5000,
    'layer_norm': True,
    'edge_dim': 32,
    'gnn_type': 'graphsage',
    'global_pool': 'mean',
    'batch_norm':False,

    'start_epoch': 0,
    'epochs': 50,

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


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()



    for i, data in enumerate(loader):
        # print(data)
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
        epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('{} loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_mse, epoch_mae, toc - tic))
    return epoch_mae, epoch_mse



def main():
    global args
    args = load_args()
    print('batch norm is', args.batch_norm)
    # 加载数据集，输出：元组 ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    dataset = CIFData(args.data_path)
    data_len = len(dataset)
    print("数据集数量有", data_len)

    structures, _, _ = dataset[0]
    # 原始atom_fea
    orig_atom_fea_len = structures[0].shape[-1]
    print('原始atom_fea长度：',orig_atom_fea_len)
    # 邻居特征长度nbr_fea
    nbr_fea_len = structures[1].shape[-1]
    print('邻居特征长度：',nbr_fea_len)
    print('batch norm is',args.batch_norm)

    '''
    # 将数据集转换为图数据
    crystal_dataset = crystal_graph_list(crystal_dataset)
    print('请检查一下：')
    print('graph数量：',len(crystal_dataset))
    print('每一张graph的数据类型为',crystal_dataset[0])
    
    # 将dataset转换为含有子图的图数据
    graph_data = GraphDataset(crystal_dataset, degree=True, k_hop=args.k_hop, se=args.se,
                                use_subgraph_edge_attr=args.use_edge_attr)
    # 将数据列表整理成用于预测晶体的批处理性能，打包(atom_fea, nbr_fea, nbr_fea_idx, target)
    print("含有子图的graphdataset形式为:",graph_data[0])
    '''
    collate_fn = custom_collate_fn
    # 数据集划分函数
    # 返回的是按照比例划分的晶体数据集
    train_dset, val_dset, test_dset = get_train_val_test_loader(
                                                        dataset=dataset,
                                                        batch_size=args.batch_size,
                                                        train_ratio=args.train_ratio,
                                                        val_ratio=args.val_ratio,
                                                        test_ratio=args.test_ratio,
                                                        return_test=True)
    #
    train_dset = crystal_graph_list(train_dset)
    print('训练集数据的形式为', train_dset[0])
    # 计算角度，确保train_dset中的数据是图数据
    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])
    # 对val和test进行同样的操作
    val_dset = crystal_graph_list(val_dset)
    test_dset = crystal_graph_list(test_dset)

    # 将图数据集转化为含有子图的数据集
    train_graph = GraphDataset(train_dset,degree=True,k_hop=args.k_hop, se=args.se,use_subgraph_edge_attr=args.use_edge_attr)
    print("含有子图的数据形式为",train_graph[0])
    val_graph = GraphDataset(val_dset,degree=True,k_hop=args.k_hop, se=args.se,use_subgraph_edge_attr=args.use_edge_attr)
    test_graph = GraphDataset(test_dset,degree=True,k_hop=args.k_hop, se=args.se,use_subgraph_edge_attr=args.use_edge_attr)
    # 加载数据，进入Dataloader中的一定是要含有子图的GraphDataset
    train_loader = DataLoader(train_graph, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers,pin_memory=args.cuda,collate_fn=collate_fn)
    val_loader = DataLoader(val_graph, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers,pin_memory=args.cuda,collate_fn=collate_fn)
    test_loader = DataLoader(test_graph, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers,pin_memory=args.cuda,collate_fn=collate_fn)
    print("数据加载完成！")
    print("加载完成的训练集形式为：",train_loader)
    for batch in train_loader:
        print("train node features (x):", batch.x.shape)
        print("train edge index:", batch.edge_index.shape)
        print("edge_index max:",    batch.edge_index.max())
        print("train edge features (edge_attr):", batch.edge_attr.shape)
        print("train target (y):", batch.y.shape)
        print("train complete edge index:", batch.complete_edge_index.shape)
        print("train degree:", batch.degree.shape)
        print("train indices:", batch.batch.shape)
        break  # 只读取第一个批次

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

    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    #
    print("准备完成，开始训练！")
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        #########################################################################################FFFFFFFFFFFFalse
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False)
        val_loss, _ = eval_epoch(model, val_loader, criterion, args.cuda, split='Val')
        test_loss, _ = eval_epoch(model, test_loader, criterion, args.cuda, split='Test')

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_mae'].append(train_loss)
        logs['val_mae'].append(val_loss)
        logs['test_mae'].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)
    print("正在测试！")
    test_loss, test_mse_loss = eval_epoch(model, test_loader, criterion, args.cuda, split='Test')
    print("test MAE loss {:.4f}".format(test_loss))
    print(args)


if __name__ == "__main__":
    main()
