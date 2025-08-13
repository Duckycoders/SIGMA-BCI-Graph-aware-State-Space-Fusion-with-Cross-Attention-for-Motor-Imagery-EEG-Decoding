#!/usr/bin/env python3
"""
EEG图神经网络模块
基于电极位置构建图结构，使用PyTorch Geometric实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# 尝试导入PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_adj
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric 成功导入")
except ImportError as e:
    print(f"警告: PyTorch Geometric 导入失败 - {e}")
    print("图卷积功能将被禁用，使用简单的全连接层代替")
    TORCH_GEOMETRIC_AVAILABLE = False


class SimpleGraphNet(nn.Module):
    """简化的图网络（当PyG不可用时的回退）"""
    
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        
        for hidden_dim in hidden_channels:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_channels))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (batch, n_nodes, in_channels)
        batch_size, n_nodes, in_channels = x.shape
        x = x.view(-1, in_channels)  # (batch * n_nodes, in_channels)
        x = self.net(x)  # (batch * n_nodes, out_channels)
        x = x.view(batch_size, n_nodes, -1)  # (batch, n_nodes, out_channels)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch, out_channels)
        return x


class EEGGraphNet(nn.Module):
    """EEG图神经网络"""
    
    def __init__(self,
                 electrode_positions: Dict[str, np.ndarray],
                 in_channels: int = 1,
                 hidden_channels: List[int] = [64, 128],
                 out_channels: int = 128,
                 graph_type: str = 'gcn',
                 use_global_pooling: bool = True,
                 dropout: float = 0.1):
        """
        初始化EEG图网络
        
        Args:
            electrode_positions: 电极位置字典
            in_channels: 输入特征维度
            hidden_channels: 隐层维度列表
            out_channels: 输出维度
            graph_type: 图网络类型 ('gcn', 'gat')
            use_global_pooling: 是否使用全局池化
            dropout: Dropout概率
        """
        super().__init__()
        
        self.electrode_positions = electrode_positions
        self.electrode_names = list(electrode_positions.keys())
        self.n_electrodes = len(self.electrode_names)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_global_pooling = use_global_pooling
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("使用简化图网络（PyG不可用）")
            self.graph_net = SimpleGraphNet(in_channels, hidden_channels, out_channels)
            return
        
        # 构建图结构
        self.edge_index = self._build_graph_structure()
        self.register_buffer('edge_index_buffer', self.edge_index)
        
        # 构建图网络层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 输入层
        if graph_type == 'gcn':
            self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        elif graph_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels[0], heads=4, concat=False))
        else:
            raise ValueError(f"不支持的图网络类型: {graph_type}")
        
        self.norms.append(nn.BatchNorm1d(hidden_channels[0]))
        self.dropouts.append(nn.Dropout(dropout))
        
        # 隐层
        for i in range(len(hidden_channels) - 1):
            if graph_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            elif graph_type == 'gat':
                self.convs.append(GATConv(hidden_channels[i], hidden_channels[i+1], heads=4, concat=False))
            
            self.norms.append(nn.BatchNorm1d(hidden_channels[i+1]))
            self.dropouts.append(nn.Dropout(dropout))
        
        # 输出层
        if graph_type == 'gcn':
            self.convs.append(GCNConv(hidden_channels[-1], out_channels))
        elif graph_type == 'gat':
            self.convs.append(GATConv(hidden_channels[-1], out_channels, heads=1, concat=False))
        
        print(f"EEG图网络初始化:")
        print(f"  电极数: {self.n_electrodes}")
        print(f"  图类型: {graph_type}")
        print(f"  边数: {self.edge_index.shape[1] if TORCH_GEOMETRIC_AVAILABLE else 0}")
        
    def _build_graph_structure(self) -> torch.Tensor:
        """构建图的边连接结构"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            return torch.empty(2, 0, dtype=torch.long)
        
        positions = np.array([self.electrode_positions[name] for name in self.electrode_names])
        
        # 计算电极间距离
        distances = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2))
        
        # 基于距离阈值构建邻接关系
        threshold = np.percentile(distances[distances > 0], 30)  # 使用30%分位数作为阈值
        
        edges = []
        for i in range(self.n_electrodes):
            for j in range(i + 1, self.n_electrodes):
                if distances[i, j] < threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # 无向图
        
        if not edges:
            # 如果没有边，创建一个完全图
            for i in range(self.n_electrodes):
                for j in range(self.n_electrodes):
                    if i != j:
                        edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, n_electrodes, in_channels)
            
        Returns:
            图特征 (batch_size, out_channels) 如果use_global_pooling=True
            或 (batch_size, n_electrodes, out_channels) 如果use_global_pooling=False
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            return self.graph_net(x)
        
        batch_size, n_electrodes, in_channels = x.shape
        
        # 重塑为PyG格式
        x = x.contiguous().view(-1, in_channels)  # (batch_size * n_electrodes, in_channels)
        
        # 创建批次索引
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(n_electrodes)
        
        # 扩展边索引到批次
        edge_index = self.edge_index_buffer
        batch_edge_index = []
        for b in range(batch_size):
            batch_edge_index.append(edge_index + b * n_electrodes)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # 图卷积层
        for i, (conv, norm, dropout) in enumerate(zip(self.convs[:-1], self.norms, self.dropouts)):
            x = conv(x, batch_edge_index)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
        
        # 最后一层
        x = self.convs[-1](x, batch_edge_index)
        
        if self.use_global_pooling:
            # 全局池化
            x = global_mean_pool(x, batch_idx)  # (batch_size, out_channels)
        else:
            # 重塑回原始形状
            x = x.view(batch_size, n_electrodes, -1)  # (batch_size, n_electrodes, out_channels)
        
        return x


class ElectrodeGraphBuilder:
    """电极图构建器"""
    
    def __init__(self, electrode_names: List[str], layout: str = 'standard'):
        self.electrode_names = electrode_names
        self.layout = layout
        self.electrode_positions = self._create_electrode_positions()
    
    def _create_electrode_positions(self) -> Dict[str, np.ndarray]:
        """创建电极位置"""
        # 标准10-20系统位置
        standard_positions = {
            'Fp1': [-0.3, 0.9, 0.3], 'Fp2': [0.3, 0.9, 0.3],
            'F3': [-0.6, 0.6, 0.6], 'F4': [0.6, 0.6, 0.6], 'Fz': [0, 0.7, 0.7],
            'C3': [-0.7, 0, 0.7], 'C4': [0.7, 0, 0.7], 'Cz': [0, 0, 1],
            'P3': [-0.6, -0.6, 0.6], 'P4': [0.6, -0.6, 0.6], 'Pz': [0, -0.7, 0.7],
            'O1': [-0.3, -0.9, 0.3], 'O2': [0.3, -0.9, 0.3],
            'F7': [-0.8, 0.4, 0.4], 'F8': [0.8, 0.4, 0.4],
            'T7': [-0.9, 0, 0.4], 'T8': [0.9, 0, 0.4],
            'P7': [-0.8, -0.4, 0.4], 'P8': [0.8, -0.4, 0.4],
            'FC1': [-0.4, 0.3, 0.8], 'FC2': [0.4, 0.3, 0.8],
            'CP1': [-0.4, -0.3, 0.8], 'CP2': [0.4, -0.3, 0.8],
        }
        
        positions = {}
        for name in self.electrode_names:
            if name in standard_positions:
                positions[name] = np.array(standard_positions[name])
            else:
                # 为未知电极生成随机位置
                positions[name] = np.random.randn(3) * 0.1
        
        return positions


def create_electrode_graph(electrode_names: List[str], 
                         layout: str = 'standard') -> ElectrodeGraphBuilder:
    """创建电极图的便捷函数"""
    return ElectrodeGraphBuilder(electrode_names, layout)


if __name__ == "__main__":
    # 测试图网络
    print("EEG图网络测试")
    
    # 创建模拟电极
    electrode_names = ['C3', 'C4', 'Cz', 'F3', 'F4']
    graph_builder = create_electrode_graph(electrode_names)
    
    # 创建图网络
    graph_net = EEGGraphNet(
        electrode_positions=graph_builder.electrode_positions,
        in_channels=1,
        hidden_channels=[32, 64],
        out_channels=128
    )
    
    # 测试前向传播
    batch_size = 4
    n_electrodes = len(electrode_names)
    x = torch.randn(batch_size, n_electrodes, 1)
    
    output = graph_net(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    print("图网络测试完成!")
