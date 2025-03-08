'''这是CGCNN的data文件'''

from __future__ import print_function, division
# 用于文件处理的包
import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

'''修改'''
import torch_geometric.utils as utils
from torch_geometric.utils import k_hop_subgraph, to_undirected

# 数据集划分函数
def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False,
                              **kwargs):
    """
    用于将数据集划分为 train、val、test 数据集的实用函数。
    ----------
    
    输入：
    完整数据集 + 划分参数 
    crystal_dataset: torch.utils.data.Dataset.
    collate_fn: torch.utils.data.DataLoader  用于批处理的函数，默认为 default_collate,用于将样本打包成一个批次。
    
    划分参数：
    batch_size: int  每个批次的大小
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool  是否返回测试集的 DataLoader；是否返回测试数据集加载器。如果为 False,则最后test_size数据将被隐藏。
    num_workers: int  数据加载时使用的工作线程数。
    pin_memory: bool 是否启用内存锁页（加速GPU传输）

    输出：
    返回 train_loader, val_loader, test_loader（当 return_test=True）
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if 
        return_test=True.
    """
    
    # 数据集的总大小
    total_size = len(dataset)
    # 如果 train_size 为 None，则使用 train_ratio 划分训练集
    if kwargs['train_size'] is None:
        # 如果 train_ratio 为 None，则使用 1 - val_ratio - test_ratio 作为训练数据
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    
    # 生成数据集的索引
    indices = list(range(total_size))
    
    # 
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
        
    # 
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

# 将多个样本打包成一个批次，处理变长原子序列。

def collate_pool(dataset_list):
    """
    输入数据结构：
    每个样本为元组 (atom_fea, nbr_fea, nbr_fea_idx, target)：
    ● atom_fea: (n_i, atom_fea_len) 原子特征矩阵（n_i为当前晶体的原子数）
    ● nbr_fea: (n_i, M, nbr_fea_len) 邻居特征矩阵（M为最大邻居数）
    ● nbr_fea_idx: (n_i, M) 邻居原子索引
    ● target: (1,) 目标属性值或标签

    输出结构：
    返回元组 (inputs, target, batch_cif_ids)：
    ● inputs 包含：
    ○ batch_atom_fea: (N, atom_fea_len) 拼接后的原子特征（N为批次总原子数）
    ○ batch_nbr_fea: (N, M, nbr_fea_len) 邻居特征
    ○ batch_nbr_fea_idx: (N, M) 修正后的全局邻居索引
    ○ crystal_atom_idx: 列表，记录每个晶体在批次中的原子范围
    ● target: (B,) 批次目标值（B为批次大小）
    ● batch_cif_ids: 列表，批次样本的CIF ID

    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_sub_nodes = []      # 新增
    batch_sub_edges = []      # 新增
    batch_sub_indicator = []  # 新增
    base_idx = 0
    edge_offset = 0
    
    
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), 
            sub_nodes, 
            sub_edges,
            sub_indicator,target, cif_id)in enumerate(dataset_list):
        # === 过滤无效边 ===
        valid_mask = (nbr_fea_idx != -1)
        nbr_fea = nbr_fea[valid_mask]
        nbr_fea_idx = nbr_fea_idx[valid_mask]

        n_i = atom_fea.shape[0]
        # 1. 收集原子特征
        batch_atom_fea.append(atom_fea)
        # 2. 收集邻居特征（过滤后的有效特征）
        batch_nbr_fea.append(nbr_fea)
        # 3. 收集邻居索引
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)

        # === 调整子图边索引 ===
        # 关键修改：基于节点偏移base_idx调整边索引
        adjusted_sub_edges = sub_edges + base_idx  # 每个样本的节点从base_idx开始
        batch_sub_edges.append(adjusted_sub_edges)

        # === 其他逻辑保持不变 ===
        batch_sub_nodes.append(sub_nodes + base_idx)
        batch_sub_indicator.append(sub_indicator + base_idx)
        base_idx += atom_fea.shape[0]  # 更新节点偏移

        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i



        # === 正确拼接边索引 ===
        # 合并后形状为 [2, total_edges]
    combined_sub_edges = torch.cat(batch_sub_edges, dim=1) if batch_sub_edges else torch.empty((2, 0))

    return (
        (   torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_sub_nodes),
            torch.cat(batch_sub_edges, dim=1),
            torch.cat(batch_sub_indicator)
        ),   # 新增
            torch.stack(batch_target, dim=0),
            batch_cif_ids
            )

# 高斯距离扩展
class GaussianDistance(object):
    """
    该类用于计算原子之间距离的高斯扩展。具体功能是根据给定的最小距离 dmin、最大距离 dmax
    和步长 step,创建一个高斯滤波器，并应用于原子间的距离矩阵。
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
          最小原子间距离
        dmax: float
          Maximum interatomic distance
          最大原子间距离
        step: float
          Step size for the Gaussian filter
          Gaussian 滤波器的步长
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        方法接受一个距离矩阵，将其与高斯滤波器进行卷积，得到扩展后的距离。
        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape
          任意形状的距离矩阵

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
          具有 length 的最后一个维度的扩展距离矩阵

        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

# 原子特征初始化器
class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    该类的作用是初始化原子的特征表示。
    初始化过程是通过一个字典将每种元素的原子特征表示存储在内存中。
    !!! Use one AtomInitializer per crystal_dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

# 读取JSON
class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    继承自 AtomInitializer，专门通过一个 JSON 文件加载原子特征，该文件包含了每种元素的特征向量。

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    
    输入：dataset的位置
    
    Parameters
    ----------

    root_dir: str
        The path to the root directory of the crystal_dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the crystal_dataset

    
    输出：元组 ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, k_hop=2):
        
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.k_hop = k_hop # 新增，用于构建k-hop子图
        
        
        # 三个文件的路径和存在性检查
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)


    # 加载一个样本时，将从 CIF 文件中读取原子特征，并使用 GaussianDistance 类扩展邻居特征，
    # 最终返回包含原子特征、邻居特征、邻居索引以及目标值（属性）的一组数据。
    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        # === 新增：过滤无效原子索引 ===
        valid_atoms = [i for i in range(len(crystal)) 
                  if crystal[i].specie.number in self.ari.atom_types]
        if not valid_atoms:
            raise ValueError(f"No valid atoms in {cif_id}")
        
        
        
        nbr_fea_idx, nbr_fea = [], []
        for i, nbr in enumerate(all_nbrs):
            # 过滤无效填充值
            valid_nbrs = [x for x in nbr if x[2] < len(crystal)]  # 确保邻居索引不越界
            if len(valid_nbrs) < self.max_num_nbr:
                valid_indices = [x[2] for x in valid_nbrs]
                valid_distances = [x[1] for x in valid_nbrs]
                # 使用真实数据填充剩余部分
                padded_indices = valid_indices + [-1]*(self.max_num_nbr - len(valid_nbrs))
                padded_distances = valid_distances + [0.]*(self.max_num_nbr - len(valid_nbrs))
            else:
                valid_nbrs = valid_nbrs[:self.max_num_nbr]
                padded_indices = [x[2] for x in valid_nbrs]
                padded_distances = [x[1] for x in valid_nbrs]
            
            nbr_fea_idx.append(padded_indices)
            nbr_fea.append(padded_distances)
        
        # === 转换时明确维度 ===
        nbr_fea = np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)  # [num_valid_edges, nbr_fea_len]
        nbr_fea = nbr_fea.view(-1, self.gdf.filter.size)  # [原子数*最大邻居数, 特征数]
        
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
        # [原子数*最大邻居数, 特征数]
        # 将 nbr_fea_idx 展平为一维 (总边数)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(-1)  # [原子数*最大邻居数]
        
        
        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor([float(target)])
        
        # === 新增子图提取 ===
        edge_index = self._get_undirected_edges(nbr_fea_idx)  # 转换为无向边
        sub_nodes_list, sub_edge_index_list, sub_indicator_list = [], [], []
        edge_offset = 0

        # 确保每个样本的 nbr_fea_idx 长度 = 原子数 * max_num_nbr
        assert len(nbr_fea_idx) == atom_fea.shape[0] * 12, \
            f"Invalid nbr_fea_idx length: {len(nbr_fea_idx)} vs {atom_fea.shape[0] * 12}"
        
        for node_idx in valid_atoms:
            # 安全获取邻居索引
            try:
                nbr_info = all_nbrs[node_idx][:self.max_num_nbr]
                nbr_indices = [n[2] for n in nbr_info]
                nbr_distances = [n[1] for n in nbr_info]
            except IndexError:
                continue
            
            # 生成有效边索引
            edge_index = self._get_valid_edges(node_idx, nbr_indices, len(crystal))
            
            # 安全提取子图
            if edge_index.size(1) == 0:  # 无邻居的情况
                sub_nodes = torch.tensor([node_idx],dtype=torch.long)
                edge_index_sub = torch.empty((2, 0), dtype=torch.long)
            else:
                try:
                    sub_nodes, edge_index_sub, _, _ = k_hop_subgraph(
                    node_idx=node_idx,
                    num_hops=self.k_hop,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=len(crystal))
                    if edge_index_sub.dim() != 2 or edge_index_sub.size(0) != 2:
                        raise ValueError(f"Invalid edge_index shape: {edge_index_sub.shape}")
                except Exception as e:
                    print(f"Error processing {cif_id} node {node_idx}: {str(e)}")
                    sub_nodes = torch.tensor([node_idx])
                    edge_index_sub = torch.empty((2, 0), dtype=torch.long)
            
            # 处理索引偏移
            sub_nodes_list.append(sub_nodes + edge_offset)
            sub_edge_index_list.append(edge_index_sub + edge_offset)
            sub_indicator_list.append(torch.full((sub_nodes.size(0),), node_idx))
            edge_offset += len(crystal)  # 按晶体结构偏移索引

            
        return ((atom_fea, nbr_fea, nbr_fea_idx),             
            torch.cat(sub_nodes_list) if sub_nodes_list else torch.empty(0),
            torch.cat(sub_edge_index_list, dim=1) if sub_edge_index_list else torch.empty((2, 0)),
            torch.cat(sub_indicator_list) if sub_indicator_list else torch.empty(0),
            target, cif_id
        )
        
    def _get_undirected_edges(self, nbr_fea_idx):
        """将邻接矩阵转换为无向边索引"""
        src_nodes = []
        dst_nodes = []
        for i in range(nbr_fea_idx.size(0)):
            valid_nbrs = nbr_fea_idx[i][nbr_fea_idx[i] != 0]
            src_nodes.extend([i] * len(valid_nbrs))
            dst_nodes.extend(valid_nbrs.tolist())
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        return to_undirected(edge_index)  # 转换为无向图
    
    
    def _get_valid_edges(self, node_idx, nbr_indices, num_nodes):
        """生成有效边索引并过滤越界索引"""
        valid_nbrs = []
        for nbr in nbr_indices:
            if 0 <= nbr < num_nodes:  # 确保邻居索引有效
                valid_nbrs.append(nbr)
        
        # 构建边索引
        src = [node_idx]*len(valid_nbrs)
        dst = valid_nbrs
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # 无向图
        return edge_index.unique(dim=1)  # 去重