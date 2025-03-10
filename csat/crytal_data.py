'''这是CGCNN的data文件'''

from __future__ import print_function, division
# 用于文件处理的包
import csv
import functools
import json
import os
import random
import warnings
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import coalesce
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data import random_split
from torch_sparse import SparseTensor
import scipy.sparse as sp




# 数据集划分函数
def get_train_val_test_loader(dataset,
                              batch_size=128,
                              train_ratio=0.8,
                              val_ratio=0.1,
                              test_ratio=0.1, return_test=False, **kwargs):
    # 数据集的总大小
    total_size = len(dataset)
    # 如果 train_ratio 为 None，则使用 1 - val_ratio - test_ratio 作为训练数据
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    
    # 生成数据集的索引
    # indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    # 使用 random_split 划分数据集
    train_sampler = dataset[:train_size]
    val_sampler = dataset[train_size:train_size+valid_size]
    if return_test:
        test_sampler = dataset[train_size+valid_size:]


    if return_test:
        return train_sampler, val_sampler, test_sampler
    else:
        return train_sampler, val_sampler

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
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # 此晶体的原子数
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx
            ),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

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
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
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
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id



def crystal_graph_list(g):
    device = torch.device('cuda')
    g_list = []
    data_len = len(g)

    for i in tqdm(range(0, data_len)):
        structures, target, cif_ids = g[i]
        # structure:
            # atom_fea: torch.Tensor shape (n_i, atom_fea_len)    atom_fea_len: int Number of atom hidden features.
            # nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)   nbr_fea_len: int Number of bond features.
            # nbr_fea_idx: torch.LongTensor shape (n_i, M)        M: Max number of neighbors
        # target: torch.Tensor shape (1, )

        '''Graph Data
        x (节点特征)：
        形状：[num_nodes, num_node_features]
        描述：每个节点的特征矩阵。num_nodes 是图中节点的数量，num_node_features 是每个节点的特征维度。
        例如，在分子图中，x 可能表示原子的类型、电荷等特征。
        edge_index (边的连接关系)：
        
        形状：[2, num_edges]
        
        edge_attr (边特征)：
        形状：[num_edges, num_edge_features]
        描述：每条边的特征矩阵。num_edges 是图中边的数量，num_edge_features 是每条边的特征维度。
        
        y (目标值)：
        形状：[num_targets]
        '''

        x = structures[0]
        nbr_fea = structures[1]
        nbr_fea_idx = structures[2]

        atom_fea_len = x.shape[1]
        n_i = nbr_fea.shape[0]
        M = nbr_fea.shape[1]
        nbr_fea_len = nbr_fea.shape[2]

        # 检查 nbr_fea_idx 中的值是否合法
        #if torch.any(nbr_fea_idx >= n_i) or torch.any(nbr_fea_idx < 0):
        #    raise ValueError("nbr_fea_idx contains invalid node indices. "
        #                     "Node indices must be in the range [0, n_i - 1].")

        # 1.构建edge_index和edge_attr
        # nbr_fea_idx 是 (n_i, M)，表示每个节点的 M 个邻居
        edge_index = torch.stack([
                        torch.arange(n_i).repeat_interleave(M),  # 每个节点的索引重复 M 次
                        nbr_fea_idx.view(-1)  # 展平为一维
                        ], dim=0)
        # print('edge_index', edge_index.shape)
        edge_index = torch.LongTensor(edge_index)

        # 生成无向图的核心修改：添加反向边
        reverse_edge_index = edge_index.flip(0)  # 交换源节点和目标节点 [dst, src]
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)  # (2, 2*n_i*M)

        # nbr_fea 是 (n_i, M, nbr_fea_len)，需要展平为 [num_edges, nbr_fea_len]
        edge_attr = nbr_fea.view(-1, nbr_fea_len)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # 复制反向边特征 (2*n_i*M, nbr_fea_len)

        # 检查并去除重复边
        # 将 edge_index 和 edge_attr 拼接在一起
        #combined = torch.cat([edge_index.T, edge_attr], dim=1)  # [num_edges, 2 + nbr_fea_len]
        # 使用 torch.unique 去重
        #unique_combined, indices = torch.unique(combined, dim=0, return_inverse=True)
        # 检查是否有重复边
        #if unique_combined.shape[0] < combined.shape[0]:
        #    print("Warning: edge_index contains duplicate edges (including edge_attr).")
        # 提取去重后的 edge_index 和 edge_attr
        #edge_index = unique_combined[:, :2].T  # [2, num_unique_edges]
        #edge_attr = unique_combined[:, 2:]  # [num_unique_edges, nbr_fea_len]

        # 检查并去除自环边
        self_loops = edge_index[0] == edge_index[1]
        if torch.any(self_loops):
            print("Warning: edge_index contains self-loops.")
            mask = edge_index[0] != edge_index[1]  # 过滤自环边
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

        edge_index, edge_attr = coalesce(edge_index, edge_attr, n_i,reduce='mean')# 聚合方式：sum/mean/min/max/mul

        # 2.x降维
        # 降到指定维度


        target_dim = 1  # 目标维度
        linear_layer = nn.Linear(atom_fea_len, target_dim)
        x = linear_layer(x) # 降维

        # edge_attr是否需要经理降维？
        # edge_attr = linear_layer(edge_attr)  # 降维



        # 将float转化成long
        x = torch.tensor(x, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        g_graph = Data(x=x,
                       edge_attr=edge_attr,
                       edge_index=edge_index,
                       y=target)
        g_list.append(g_graph)



    return g_list