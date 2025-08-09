#!/usr/bin/env python3
"""
MI-Net: 运动想象EEG解码的完整网络架构
整合FilterBank + GraphConv + S4 + Mamba + Cross-Attention + MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .filterbank import FilterBankLayer, AdaptiveFilterBank
from .graph import EEGGraphNet, create_electrode_graph
from .s4_branch import S4Branch
from .mamba_branch import MambaBranch, create_mamba_branch
from .fusion import CrossAttentionFusion, MultiScaleFusion
from .moe_adapter import MoEAdapter, MultiTaskMoEAdapter


class MINet(nn.Module):
    """
    运动想象EEG解码网络
    
    架构流程:
    Input EEG -> FilterBank -> GraphConv -> [S4分支, Mamba分支] -> Cross-Attention融合 -> MoE Adapter -> 分类头
    """
    
    def __init__(self,
                 # 数据配置
                 n_channels: int,
                 n_samples: int,
                 sfreq: float = 250.0,
                 electrode_positions: Dict[str, np.ndarray] = None,
                 
                 # FilterBank配置
                 filter_bands: List[Tuple[float, float]] = None,
                 use_adaptive_filterbank: bool = False,
                 
                 # GraphConv配置
                 use_graph_conv: bool = True,
                 graph_hidden_dims: List[int] = [64, 128],
                 graph_output_dim: int = 128,
                 
                 # S4分支配置
                 use_s4_branch: bool = True,
                 s4_d_model: int = 128,
                 s4_d_state: int = 64,
                 s4_n_layers: int = 2,
                 
                 # Mamba分支配置
                 use_mamba_branch: bool = True,
                 mamba_d_model: int = 128,
                 mamba_d_state: int = 16,
                 mamba_n_layers: int = 2,
                 
                 # 融合配置
                 fusion_method: str = 'cross_attention',  # 'cross_attention', 'simple', 'adaptive'
                 fusion_n_heads: int = 8,
                 
                 # MoE配置
                 use_moe: bool = True,
                 moe_n_experts: int = 4,
                 moe_top_k: int = 2,
                 
                 # 多任务配置
                 task_configs: Dict[str, Dict] = None,
                 
                 # 通用配置
                 dropout: float = 0.1,
                 
                 # 模块开关（用于消融实验）
                 module_switches: Dict[str, bool] = None):
        """
        初始化MI-Net
        
        Args:
            n_channels: EEG通道数
            n_samples: 时间采样点数
            sfreq: 采样频率
            electrode_positions: 电极位置字典
            filter_bands: FilterBank频带
            use_adaptive_filterbank: 是否使用自适应FilterBank
            use_graph_conv: 是否使用图卷积
            graph_hidden_dims: 图卷积隐层维度
            graph_output_dim: 图卷积输出维度
            use_s4_branch: 是否使用S4分支
            s4_d_model: S4模型维度
            s4_d_state: S4状态维度
            s4_n_layers: S4层数
            use_mamba_branch: 是否使用Mamba分支
            mamba_d_model: Mamba模型维度
            mamba_d_state: Mamba状态维度
            mamba_n_layers: Mamba层数
            fusion_method: 融合方法
            fusion_n_heads: 融合注意力头数
            use_moe: 是否使用MoE
            moe_n_experts: MoE专家数
            moe_top_k: MoE top-k
            task_configs: 多任务配置
            dropout: Dropout概率
            module_switches: 模块开关（消融实验用）
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        
        # 模块开关设置
        if module_switches is None:
            module_switches = {}
        
        self.use_filterbank = module_switches.get('filterbank', True)
        self.use_graph_conv = module_switches.get('graph_conv', use_graph_conv)
        self.use_s4_branch = module_switches.get('s4_branch', use_s4_branch)
        self.use_mamba_branch = module_switches.get('mamba_branch', use_mamba_branch)
        self.use_fusion = module_switches.get('fusion', True)
        self.use_moe = module_switches.get('moe', use_moe)
        
        # 确保至少有一个序列建模分支
        if not (self.use_s4_branch or self.use_mamba_branch):
            self.use_s4_branch = True
            print("警告: 至少需要一个序列建模分支，启用S4分支")
        
        # 设置默认值
        if filter_bands is None:
            filter_bands = [(4, 8), (8, 14), (14, 30), (30, 45)]
        
        if task_configs is None:
            task_configs = {
                'default': {'d_output': 4, 'top_k': 2}  # 默认4类分类
            }
        
        self.filter_bands = filter_bands
        self.n_bands = len(filter_bands)
        self.task_configs = task_configs
        
        # 1. FilterBank模块
        if self.use_filterbank:
            if use_adaptive_filterbank:
                self.filterbank = AdaptiveFilterBank(
                    n_channels=n_channels,
                    n_bands=self.n_bands,
                    sfreq=sfreq,
                    learnable=True
                )
            else:
                self.filterbank = FilterBankLayer(
                    n_channels=n_channels,
                    filter_bands=filter_bands,
                    sfreq=sfreq,
                    filter_type='conv1d'
                )
        
        # 2. 图卷积模块
        if self.use_graph_conv:
            if electrode_positions is None:
                # 创建默认电极位置
                electrode_names = [f'C{i}' for i in range(n_channels)]
                self.graph_builder = create_electrode_graph(electrode_names, 'standard')
                electrode_positions = self.graph_builder.electrode_positions
            
            # 图卷积在每个时间点对通道间关系建模：每个节点(电极)的特征取单通道标量
            # 因此输入特征维度设置为1
            graph_input_dim = 1
            
            self.graph_net = EEGGraphNet(
                electrode_positions=electrode_positions,
                in_channels=graph_input_dim,
                hidden_channels=graph_hidden_dims,
                out_channels=graph_output_dim,
                # 对节点做全局池化，得到每个时间点的图特征（便于送入序列模型）
                use_global_pooling=True
            )
            feature_dim = graph_output_dim
        else:
            # 不使用图卷积时，特征维度就是1（平均池化后的通道特征）
            feature_dim = 1
        
        # 3. 序列建模分支
        branch_output_dim = max(s4_d_model, mamba_d_model)
        
        if self.use_s4_branch:
            self.s4_branch = S4Branch(
                d_input=feature_dim,
                d_model=s4_d_model,
                d_state=s4_d_state,
                n_layers=s4_n_layers,
                dropout=dropout
            )
        
        if self.use_mamba_branch:
            self.mamba_branch = create_mamba_branch(
                d_input=feature_dim,
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                n_layers=mamba_n_layers,
                dropout=dropout
            )
        
        # 4. 融合模块
        if self.use_fusion and self.use_s4_branch and self.use_mamba_branch:
            if fusion_method == 'cross_attention':
                # 确保两个分支输出维度一致
                if s4_d_model != mamba_d_model:
                    self.s4_projection = nn.Linear(s4_d_model, branch_output_dim)
                    self.mamba_projection = nn.Linear(mamba_d_model, branch_output_dim)
                else:
                    self.s4_projection = nn.Identity()
                    self.mamba_projection = nn.Identity()
                
                self.fusion_module = CrossAttentionFusion(
                    d_model=branch_output_dim,
                    n_heads=fusion_n_heads,
                    dropout=dropout,
                    fusion_method='bidirectional'
                )
                
            elif fusion_method == 'simple':
                # 简单的加权融合
                self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
                
            elif fusion_method == 'adaptive':
                # 自适应融合（实现略）
                pass
            
            fusion_output_dim = branch_output_dim
        else:
            # 如果只有一个分支或不使用融合
            if self.use_s4_branch and self.use_mamba_branch:
                fusion_output_dim = branch_output_dim  # 简单拼接
            elif self.use_s4_branch:
                fusion_output_dim = s4_d_model
            else:
                fusion_output_dim = mamba_d_model
        
        # 5. 多尺度融合（处理FilterBank的多频带）
        if self.use_filterbank and self.n_bands > 1:
            self.multiscale_fusion = MultiScaleFusion(
                d_model=fusion_output_dim,
                n_bands=self.n_bands,
                fusion_type='attention'
            )
        
        # 6. 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 7. MoE Adapter和分类头
        if self.use_moe and len(task_configs) > 1:
            # 多任务MoE
            self.moe_adapter = MultiTaskMoEAdapter(
                d_input=fusion_output_dim,
                task_configs=task_configs,
                shared_experts=2,
                task_specific_experts=moe_n_experts - 2,
                dropout=dropout
            )
        elif self.use_moe:
            # 单任务MoE
            self.moe_adapter = MoEAdapter(
                d_input=fusion_output_dim,
                d_output=fusion_output_dim,
                n_experts=moe_n_experts,
                top_k=moe_top_k,
                dropout=dropout
            )
            # 添加分类头
            self.classifiers = nn.ModuleDict()
            for task_name, config in task_configs.items():
                self.classifiers[task_name] = nn.Linear(fusion_output_dim, config['d_output'])
        else:
            # 简单分类头
            self.classifiers = nn.ModuleDict()
            for task_name, config in task_configs.items():
                self.classifiers[task_name] = nn.Linear(fusion_output_dim, config['d_output'])
        
        # 8. Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._print_architecture()
    
    def _print_architecture(self):
        """打印网络架构信息"""
        print(f"\nMI-Net架构初始化:")
        print(f"  输入: {self.n_channels}通道 × {self.n_samples}采样点")
        print(f"  FilterBank: {'✓' if self.use_filterbank else '✗'} ({self.n_bands}频带)")
        print(f"  图卷积: {'✓' if self.use_graph_conv else '✗'}")
        print(f"  S4分支: {'✓' if self.use_s4_branch else '✗'}")
        print(f"  Mamba分支: {'✓' if self.use_mamba_branch else '✗'}")
        print(f"  融合模块: {'✓' if self.use_fusion else '✗'}")
        print(f"  MoE: {'✓' if self.use_moe else '✗'}")
        print(f"  任务数: {len(self.task_configs)}")
    
    def forward(self, 
                x: torch.Tensor, 
                task_name: str = 'default',
                return_features: bool = False,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_samples)
            task_name: 任务名称
            return_features: 是否返回中间特征
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            包含输出和可选特征的字典
        """
        batch_size, n_channels, n_samples = x.shape
        features = {}
        
        # 1. FilterBank处理
        if self.use_filterbank:
            x = self.filterbank(x)  # (batch, n_bands, n_channels, n_samples)
            features['filterbank'] = x
        else:
            x = x.unsqueeze(1)  # (batch, 1, n_channels, n_samples)
        
        # 2. 图卷积处理
        if self.use_graph_conv:
            # 输入来自FilterBank: (batch, n_bands, n_channels, n_samples)
            batch_size, n_bands, n_channels, n_samples = x.shape

            # 我们对每个时间点进行一次图卷积：节点为电极，特征为该时刻该频带的幅值（标量）
            # 将数据重排为 (batch*n_bands, n_samples, n_channels)
            x = x.permute(0, 1, 3, 2).contiguous()  # (batch, n_bands, n_samples, n_channels)
            x = x.view(batch_size * n_bands, n_samples, n_channels)

            # 逐时间步送入图网络，得到每个时间点的图级特征（全局池化后维度为 graph_output_dim）
            graph_features = []
            for t in range(n_samples):
                # 取第t个时间点的节点特征：(batch*n_bands, n_channels)
                x_t = x[:, t, :].unsqueeze(-1)  # (batch*n_bands, n_channels, 1)
                # EEGGraphNet 期望输入为 (batch, n_nodes, in_channels)
                g_t = self.graph_net(x_t)  # (batch*n_bands, graph_output_dim)
                graph_features.append(g_t)

            # 堆叠回时间维度 -> (batch*n_bands, n_samples, graph_output_dim)
            x = torch.stack(graph_features, dim=1)
            # 还原批次与频带维度 -> (batch, n_bands, n_samples, graph_output_dim)
            x = x.view(batch_size, n_bands, n_samples, -1)

            features['graph_conv'] = x
        else:
            # 不使用图卷积时，简单处理维度
            # x shape: (batch, n_bands, n_channels, n_samples)
            # 转换为: (batch, n_bands, n_samples, n_channels)
            x = x.permute(0, 1, 3, 2)
            # 对通道维度进行平均池化，得到每个频带的时间序列特征
            x = torch.mean(x, dim=-1, keepdim=True)  # (batch, n_bands, n_samples, 1)
        
        # 3. 序列建模分支
        s4_output = None
        mamba_output = None
        
        if self.use_s4_branch:
            s4_input = x  # (batch, n_bands, n_samples, feature_dim)
            s4_output = self.s4_branch(s4_input)  # (batch, n_bands, n_samples, s4_d_model)
            
            if hasattr(self, 's4_projection'):
                s4_output = self.s4_projection(s4_output)
            
            features['s4_branch'] = s4_output
        
        if self.use_mamba_branch:
            mamba_input = x
            mamba_output = self.mamba_branch(mamba_input)  # (batch, n_bands, n_samples, mamba_d_model)
            
            if hasattr(self, 'mamba_projection'):
                mamba_output = self.mamba_projection(mamba_output)
            
            features['mamba_branch'] = mamba_output
        
        # 4. 分支融合
        attention_weights = None
        
        if self.use_fusion and s4_output is not None and mamba_output is not None:
            # 跨注意力融合
            if hasattr(self, 'fusion_module'):
                # 重塑为融合模块所需格式
                batch_size, n_bands, n_samples, d_model = s4_output.shape
                s4_flat = s4_output.view(batch_size * n_bands, n_samples, d_model)
                mamba_flat = mamba_output.view(batch_size * n_bands, n_samples, d_model)
                
                fusion_result = self.fusion_module(
                    s4_flat, mamba_flat, 
                    return_attention_weights=return_attention_weights
                )
                
                fused_output = fusion_result['fused']  # (batch * n_bands, n_samples, d_model)
                fused_output = fused_output.view(batch_size, n_bands, n_samples, d_model)
                
                if return_attention_weights:
                    attention_weights = fusion_result['attention_weights']
            
            elif hasattr(self, 'fusion_weights'):
                # 简单加权融合
                weights = F.softmax(self.fusion_weights, dim=0)
                fused_output = weights[0] * s4_output + weights[1] * mamba_output
            
            features['fused'] = fused_output
            
        elif s4_output is not None and mamba_output is not None:
            # 简单拼接
            fused_output = torch.cat([s4_output, mamba_output], dim=-1)
            features['concatenated'] = fused_output
            
        elif s4_output is not None:
            fused_output = s4_output
            
        else:
            fused_output = mamba_output
        
        # 5. 多尺度融合（处理多频带）
        if self.use_filterbank and self.n_bands > 1 and hasattr(self, 'multiscale_fusion'):
            fused_output = self.multiscale_fusion(fused_output)  # (batch, n_samples, d_model)
        else:
            # 简单平均池化多频带
            if fused_output.dim() == 4:  # (batch, n_bands, n_samples, d_model)
                fused_output = torch.mean(fused_output, dim=1)  # (batch, n_samples, d_model)
        
        features['multiscale_fused'] = fused_output
        
        # 6. 全局池化
        pooled = self.global_pool(fused_output.transpose(-1, -2)).squeeze(-1)  # (batch, d_model)
        pooled = self.dropout(pooled)
        features['pooled'] = pooled
        
        # 7. MoE和分类
        aux_loss = None
        
        if self.use_moe:
            if isinstance(self.moe_adapter, MultiTaskMoEAdapter):
                # 多任务MoE
                moe_result = self.moe_adapter(pooled, task_name)
                logits = moe_result['output']
                if 'aux_loss' in moe_result:
                    aux_loss = moe_result['aux_loss']
            else:
                # 单任务MoE
                moe_result = self.moe_adapter(pooled)
                moe_output = moe_result['output']
                if 'aux_loss' in moe_result:
                    aux_loss = moe_result['aux_loss']
                
                # 通过分类头
                logits = self.classifiers[task_name](moe_output)
        else:
            # 直接分类
            logits = self.classifiers[task_name](pooled)
        
        # 构建返回结果
        result = {
            'logits': logits,
            'predictions': F.softmax(logits, dim=-1)
        }
        
        if aux_loss is not None:
            result['aux_loss'] = aux_loss
        
        if return_features:
            result['features'] = features
        
        if attention_weights is not None:
            result['attention_weights'] = attention_weights
        
        return result
    
    def get_num_parameters(self) -> Dict[str, int]:
        """获取模型参数统计"""
        param_counts = {}
        
        for name, module in self.named_children():
            param_counts[name] = sum(p.numel() for p in module.parameters())
        
        param_counts['total'] = sum(p.numel() for p in self.parameters())
        
        return param_counts


def create_mi_net(dataset_type: str = 'bnci2a', **kwargs) -> MINet:
    """
    创建MI-Net的便捷函数
    
    Args:
        dataset_type: 数据集类型 ('bnci2a', 'bnci2b', 'eegmmi')
        **kwargs: 其他参数
        
    Returns:
        MI-Net模型实例
    """
    # 数据集特定配置
    if dataset_type == 'bnci2a':
        default_config = {
            'n_channels': 22,
            'n_samples': 1000,  # 4秒 * 250Hz
            'task_configs': {
                'motor_imagery': {'d_output': 4, 'top_k': 2}  # 4类
            }
        }
    elif dataset_type == 'bnci2b':
        default_config = {
            'n_channels': 3,  # 双极导联
            'n_samples': 1000,
            'task_configs': {
                'motor_imagery': {'d_output': 2, 'top_k': 2}  # 2类
            }
        }
    elif dataset_type == 'eegmmi':
        default_config = {
            'n_channels': 64,
            'n_samples': 640,  # 4秒 * 160Hz
            'sfreq': 160.0,
            'task_configs': {
                'motor_imagery': {'d_output': 4, 'top_k': 2}  # 4类
            }
        }
    else:
        default_config = {
            'n_channels': 22,
            'n_samples': 1000,
            'task_configs': {
                'default': {'d_output': 4, 'top_k': 2}
            }
        }
    
    # 合并配置
    config = {**default_config, **kwargs}
    
    return MINet(**config)


if __name__ == "__main__":
    # 测试MI-Net
    print("MI-Net完整架构测试")
    
    # 测试BNCI 2a配置
    print("\n=== 测试BNCI 2a配置 ===")
    model_2a = create_mi_net('bnci2a')
    
    batch_size = 4
    x_2a = torch.randn(batch_size, 22, 1000)
    
    output_2a = model_2a(x_2a, task_name='motor_imagery', return_features=True)
    print(f"输入形状: {x_2a.shape}")
    print(f"输出形状: {output_2a['logits'].shape}")
    print(f"预测形状: {output_2a['predictions'].shape}")
    
    # 参数统计
    param_counts = model_2a.get_num_parameters()
    print(f"总参数量: {param_counts['total']:,}")
    
    # 测试BNCI 2b配置
    print("\n=== 测试BNCI 2b配置 ===")
    model_2b = create_mi_net('bnci2b')
    
    x_2b = torch.randn(batch_size, 3, 1000)
    output_2b = model_2b(x_2b, task_name='motor_imagery')
    print(f"BNCI 2b输出形状: {output_2b['logits'].shape}")
    
    # 测试消融实验配置
    print("\n=== 测试消融实验配置 ===")
    
    # 只使用S4分支
    model_s4_only = create_mi_net(
        'bnci2a',
        module_switches={'mamba_branch': False}
    )
    output_s4 = model_s4_only(x_2a, task_name='motor_imagery')
    print(f"S4-only输出形状: {output_s4['logits'].shape}")
    
    # 不使用FilterBank
    model_no_fb = create_mi_net(
        'bnci2a',
        module_switches={'filterbank': False}
    )
    output_no_fb = model_no_fb(x_2a, task_name='motor_imagery')
    print(f"No-FilterBank输出形状: {output_no_fb['logits'].shape}")
    
    print("\nMI-Net测试完成!")
