from __future__ import print_function, division

import torch
import torch.nn as nn

# 使用sat.layers中的TransformerEncoderLayer 
# from sat.layers import TransformerEncoderLayer
from torch_geometric.nn import TransformerConv, global_mean_pool
import torch.nn.functional as F
from torch_scatter import scatter
import warnings



# 自注意力层
class SATLayer(nn.Module):
    """
    Structure-aware Transformer Layer for Crystal Graphs
    结合k-hop子图信息与Transformer注意力机制
    """
    def __init__(self, atom_fea_len, nbr_fea_len, heads=8, k_hop=3):
        super(SATLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.k_hop = k_hop
        
        # Structure-aware Attention Components
        self.transformer_conv = TransformerConv(
            in_channels=atom_fea_len,
            out_channels=atom_fea_len // heads,
            heads=heads,
            edge_dim=nbr_fea_len
        )
        
        # Subgraph processing
        self.subgraph_norm = nn.LayerNorm(atom_fea_len)
        self.subgraph_proj = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.SiLU(),
            nn.Linear(atom_fea_len, atom_fea_len)
        )

        # Final融合
        self.out_norm = nn.LayerNorm(atom_fea_len)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr, subgraph_data):
        """
        输入:
        x: [N, atom_fea_len] 原子特征
        edge_index: [2, E] 邻接关系
        edge_attr: [E, nbr_fea_len] 边特征
        subgraph_data: 包含子图信息的字典
        """
        # === 新增维度验证 ===
        assert edge_index.size(1) == edge_attr.size(0), \
            f"Edge mismatch: {edge_index.shape} vs {edge_attr.shape}"
            
        # 空边处理
        if edge_index.size(1) == 0:
            return x  # 直接返回原始特征，不进行消息传递
        
        # 原有断言调整为警告
        if edge_index.size(1) != edge_attr.size(0):
            warnings.warn(f"Edge dimension mismatch: {edge_index.shape} vs {edge_attr.shape}")
            edge_attr = edge_attr[:edge_index.size(1)]  # 截断匹配维度
        
        
        # 原始Transformer卷积
        x_trans = self.transformer_conv(x, edge_index, edge_attr)
        
        # 子图信息处理
        sub_x = self.process_subgraphs(x, subgraph_data)
        
        # 特征融合
        x = x + self.dropout(x_trans) + sub_x
        x = self.out_norm(x)
        return self.act(x)

    def process_subgraphs(self, x, subgraph_data):
        # 从子图数据中提取信息
        sub_nodes = subgraph_data['sub_nodes']  # [total_subgraph_nodes]
        sub_indicator = subgraph_data['sub_indicator']  # [total_subgraph_nodes]
        sub_edge_index = subgraph_data['sub_edges']  # [2, total_subgraph_edges]
        
        # 获取子图节点特征
        sub_x = x[sub_nodes]
        
        # 子图内消息传递
        sub_x = self.subgraph_proj(
            self.subgraph_norm(sub_x)
        )
        
        # 聚合到中心节点
        return scatter(sub_x, sub_indicator, dim=0, reduce='mean')


# 晶体图卷积网络
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    创建用于预测总数的晶体图卷积神经网络材料属性
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False,max_num_nbr=12):
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
        self.max_num_nbr = max_num_nbr
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # SAT卷积层
        self.convs = nn.ModuleList([
            SATLayer(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                heads=8,
                k_hop=3
            ) for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        # 池化与全连接
        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.SiLU()
        )
        
        self.fcs = nn.ModuleList([
            nn.Linear(h_fea_len, h_fea_len)
            for _ in range(n_h-1)
        ])
        self.softpluses = nn.ModuleList([nn.SiLU() for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 1)


    def forward(self, inputs):
        """
        修改后的前向传播，支持子图数据
        输入格式:
        (atom_fea, nbr_fea, nbr_fea_idx, sub_nodes, sub_edges, sub_indicator)
        """
        (atom_fea, nbr_fea, nbr_fea_idx,
         sub_nodes, sub_edges, sub_indicator) = inputs
        
        # 转换边索引格式
        edge_index = self._create_edge_index(nbr_fea_idx,atom_fea)
        
        # 准备子图数据
        subgraph_data = {
            'sub_nodes': sub_nodes,
            'sub_edges': sub_edges,
            'sub_indicator': sub_indicator
        }
        
        # 嵌入原子特征
        x = self.embedding(atom_fea)
        
        # 逐层处理
        for conv in self.convs:
            x = conv(
                x=x,
                edge_index=edge_index,
                edge_attr=nbr_fea.view(-1, nbr_fea.size(-1)),
                subgraph_data=subgraph_data
            )
        
        # 池化
        crys_fea = self.pooling(x, sub_indicator)
        
        # 全连接
        crys_fea = self.conv_to_fc(crys_fea)
        for fc, act in zip(self.fcs, self.softpluses):
            crys_fea = act(fc(crys_fea))
        
        # 输出
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def _create_edge_index(self, nbr_fea_idx,atom_fea):
        """处理展平后的边索引"""
        device = nbr_fea_idx.device

        """根据原子特征维度重建原子数"""
        num_atoms = atom_fea.size(0)  # 直接通过原子特征维度获取原子数
        src_nodes = torch.arange(num_atoms, device=device).repeat_interleave(self.max_num_nbr)


        valid_mask = (nbr_fea_idx != -1)
        src = src_nodes[valid_mask]
        dst = nbr_fea_idx[valid_mask]
        
        return torch.stack([src, dst], dim=0)

    def pooling(self, x, sub_indicator):
        """
        改进的池化方法，考虑子图结构
        """
        # 子图级平均池化
        subgraph_feats = scatter(x, sub_indicator, dim=0, reduce='mean')
        
        # 全局平均池化
        return subgraph_feats.mean(dim=0, keepdim=True)

