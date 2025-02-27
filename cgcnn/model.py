from __future__ import print_function, division

import torch
import torch.nn as nn

# 卷积层
class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    ConvLayer 是用于图卷积的核心层，其作用是从原子和邻居的特征中提取新的特征表示。
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
          原子特征的长度。
        nbr_fea_len: int
          Number of bond features.
          邻居特征的长度。
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        '''
        网络层
        fc_full：全连接层，将原子特征、邻居特征拼接后进行变换。
        sigmoid：激活函数，用于对特征进行门控。
        softplus1 和 softplus2：平滑的激活函数，用于特征的非线性变换。
        bn1 和 bn2：批量归一化层，用于稳定训练过程。
        '''
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass
        atom_in_fea[nbr_fea_idx, :]：根据邻居索引选取邻居的原子特征。
        total_nbr_fea：将原子特征、邻居原子特征以及邻居特征拼接，形成一个综合特征表示。
        fc_full：通过全连接层对综合特征进行变换。
        门控机制：通过 sigmoid 激活函数对邻居特征进行门控，决定哪些邻居的影响更强。
        nbr_filter * nbr_core：通过门控特征与核心特征的乘积，得到每个邻居的加权影响。
        最后对加权后的特征进行池化和非线性变换。

        N: Total number of atoms in the batch
        批次中的原子总数
        M: Max number of neighbors
        最大相邻节点数

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
          卷积前的 Atom 隐藏特征
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
          每个原子的 M 个邻居的键特征
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
          每个原子的 M 个邻居的索引


        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution
          卷积后的 Atom 隐藏特征

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

# 晶体图卷积网络
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    创建用于预测总数的晶体图卷积神经网络材料属性
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        embedding：将输入的原子特征通过全连接层映射到隐藏空间。
        convs：构建多个图卷积层（使用 ConvLayer）。
        conv_to_fc：卷积层输出到全连接层的转换层，用于连接卷积层与后续的全连接层。
        池化与全连接层：
            fcs：根据需要构建多个全连接层，处理卷积层后的特征。
            fc_out：最终的输出层，回归任务输出一个值，分类任务输出两个类的概率。
            分类：如果是分类任务，使用 LogSoftmax 和 Dropout。

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
          输入中的 atom 特征数
        nbr_fea_len: int
          Number of bond features.
          键特征的数量。
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
          卷积层中隐藏的原子特征数
        n_conv: int
          Number of convolutional layers
          卷积层数
        h_fea_len: int
          Number of hidden features after pooling
          池化后的隐藏特征数
        n_h: int
          Number of hidden layers after pooling
          池化后的隐藏层数
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        atom_fea = self.embedding(atom_fea)：将原子特征通过嵌入层转换为隐藏特征。
        依次通过每个卷积层 conv_func，对原子特征进行卷积操作，提取邻居信息。
        pooling：将原子特征池化为晶体特征，得到整体晶体的表示。
        通过全连接层进一步处理晶体特征，并根据任务类型（回归或分类）进行输出。

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
