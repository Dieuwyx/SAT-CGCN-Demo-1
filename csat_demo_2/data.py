# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os
from torch_geometric.data import  Batch

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes 
    if key == 'subgraph_node_idx':
        return self.num_nodes 
    if key == 'subgraph_indicator':
        return self.num_nodes 
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0


class GraphDataset(object):
    def __init__(self, dataset, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,
                 cache_path=None, return_complete_index=True):
        '''
        crystal_dataset：图数据集，通常是一个包含图结构数据的列表。
        degree：是否计算节点的度。
        k_hop：子图提取的k-hop值，用于提取节点的邻居信息。
        se (structure extractor)：指定使用的结构提取器类型，默认为gnn，也可以选择khopgnn。
        use_subgraph_edge_attr：是否在子图中使用边特征。
        cache_path：用于缓存的路径，避免每次都重新计算。
        return_complete_index：是否返回完整的索引信息。
        cif_path 和 atom_init_file：这两个用于加载和初始化原子特征

        Initialize the graph crystal_dataset.
        '''


        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree

        self.compute_degree()
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index
        self.k_hop = k_hop
        self.se = se
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        # 子图的缓存路径
        self.cache_path = cache_path
        # 子图提取
        if self.se == 'khopgnn':
            Data.__inc__ = my_inc
            self.extract_subgraphs()
 
    def compute_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    self.k_hop, 
                    graph.edge_index,
                    relabel_nodes=True, 
                    num_nodes=graph.num_nodes
                    )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        if self.n_features == 1:
            data.x = data.x.squeeze(-1)
        if not isinstance(data.y, list):
            data.y = data.y.view(data.y.shape[0], -1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[index]
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list[index]
         
        # add subgraphs and relevant meta data
        if self.se == "khopgnn":
            if self.cache_path is not None:
                cache_file = torch.load("{}_{}.pt".format(self.cache_path, index))
                data.subgraph_edge_index = cache_file['subgraph_edge_index']
                data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
                data.subgraph_node_idx = cache_file['subgraph_node_index']
                data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
                data.subgraph_indicator = cache_file['subgraph_indicator_index']
                return data
            data.subgraph_edge_index = self.subgraph_edge_index[index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[index])
            data.subgraph_node_idx = self.subgraph_node_index[index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[index]
            data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)
        else:
            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data


def custom_collate_fn(data_list):
    """
    自定义 collate_fn，用于处理维度不一致的图数据。
    """
    batch = Batch()

    # 处理节点特征 (x)
    batch.x = torch.cat([data.x for data in data_list], dim=0)

    # 处理边索引 (edge_index)
    edge_index_list = []
    num_nodes = 0
    for data in data_list:
        edge_index_list.append(data.edge_index.long() + num_nodes)  # 偏移边索引
        num_nodes += data.x.size(0)  # 累加节点数
    batch.edge_index = torch.cat(edge_index_list, dim=1)

    # 处理边特征 (edge_attr)
    batch.edge_attr = torch.cat([data.edge_attr for data in data_list], dim=0)

    # 处理目标标签 (y)
    batch.y = torch.cat([data.y for data in data_list], dim=0)

    # 处理完全图的边索引 (complete_edge_index)
    complete_edge_index_list = []
    num_nodes = 0
    for data in data_list:
        complete_edge_index_list.append(data.complete_edge_index.long() + num_nodes)  # 偏移边索引
        num_nodes += data.x.size(0)  # 累加节点数
    batch.complete_edge_index = torch.cat(complete_edge_index_list, dim=1)


    # 处理节点的度 (degree)
    batch.degree = torch.cat([data.degree for data in data_list], dim=0)

    # 添加 batch 索引
    batch.batch = torch.cat([
        torch.full((data.x.size(0),), i, type=torch.long) for i, data in enumerate(data_list)
    ], dim=0)

    return batch

