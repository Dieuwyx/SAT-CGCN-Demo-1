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
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import random_split
from scipy.sparse import coo_matrix

# 数据集划分函数
def get_train_val_test_loader(dataset,
                              batch_size=128,
                              train_ratio=0.8,
                              val_ratio=0.1, test_ratio=0.1, return_test=False, **kwargs):
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
    train_dset, val_dset, test_dset = random_split(dataset, [train_size, valid_size, test_size])
    if return_test:
        return train_dset, val_dset, test_dset
    else:
        return train_dset, val_dset

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
    print('正在转化的数据集大小为：',data_len)

    for i in tqdm(range(0, data_len)):
        structures, target, cif_ids = g[i]
        g_graph = Data(x = structures[0],
                             edge_attr = structures[1],
                             edge_index = structures[2],
                             y = target)
        g_list.append(g_graph)
        g_list[i].x = g_list[i].x.long()
        g_list[i].y = g_list[i].y.long()

        # 对三维的边特征进行降维
        edge_attr = g_list[i].edge_attr
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        g_list[i].edge_attr = edge_attr.long()
        # print('edge_attr的维度，请判断是否为2维',edge_attr.shape)

        # 将edge_index重塑为 (num_edges, 2)
        edge_index = g_list[i].edge_index
        num_edges = edge_index.size(0) * edge_index.size(1) // 2
        edge_index = edge_index.view(-1, 2)  # 重塑为 (num_edges, 2)
        # 转置为 (2, num_edges)
        edge_index = edge_index.t()
        # 打印调整后的edge_index
        # print('edge_index为，请判断是否为(2, num_edges)',edge_index.shape)  # 应该是 (2, num_edges)
        g_list[i].edge_index = edge_index.long()

    print("转化已完成，请检查一下！")
    print(g_list[0])
    print('edge_idex的维度是',g_list[0].edge_index.shape)

    return g_list