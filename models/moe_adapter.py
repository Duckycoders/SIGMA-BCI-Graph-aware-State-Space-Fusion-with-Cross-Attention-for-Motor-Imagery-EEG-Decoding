#!/usr/bin/env python3
"""
MoE (Mixture of Experts) Adapter模块
实现轻量级的专家混合机制，用于多任务学习和域适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math


class Expert(nn.Module):
    """单个专家网络"""
    
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 d_output: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化专家网络
        
        Args:
            d_input: 输入维度
            d_hidden: 隐层维度
            d_output: 输出维度
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        
        # 激活函数选择
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'swish':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        # 专家网络结构
        self.network = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
            nn.Dropout(dropout)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


class Router(nn.Module):
    """路由器：决定如何组合专家"""
    
    def __init__(self,
                 d_input: int,
                 n_experts: int,
                 top_k: int = 2,
                 noisy_gating: bool = True,
                 noise_epsilon: float = 1e-2):
        """
        初始化路由器
        
        Args:
            d_input: 输入特征维度
            n_experts: 专家数量
            top_k: 选择的专家数量
            noisy_gating: 是否使用噪声门控
            noise_epsilon: 噪声强度
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        
        # 门控网络
        self.gate = nn.Linear(d_input, n_experts, bias=False)
        
        # 噪声网络（用于训练时的探索）
        if noisy_gating:
            self.noise_gate = nn.Linear(d_input, n_experts, bias=False)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.gate.weight)
        if self.noisy_gating:
            nn.init.xavier_uniform_(self.noise_gate.weight)
    
    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """计算专家负载"""
        return torch.sum(gates, dim=0)
    
    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor) -> torch.Tensor:
        """计算在top-k中的概率"""
        batch = clean_values.size(0)
        m = noisy_values.size(-1)
        
        # 获取top-k阈值
        top_k_values, _ = torch.topk(clean_values, self.top_k, dim=-1)
        threshold = top_k_values[:, -1:]  # (batch, 1)
        
        # 计算在top-k中的概率
        prob = torch.sum(clean_values >= threshold, dim=-1, keepdim=True).float() / self.top_k
        
        return prob
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, ..., d_input)
            
        Returns:
            (expert_weights, expert_indices, aux_loss_info)
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # (batch * ..., d_input)
        
        # 计算门控值
        clean_logits = self.gate(x_flat)  # (batch * ..., n_experts)
        
        if self.noisy_gating and self.training:
            # 添加噪声用于探索
            noise_logits = self.noise_gate(x_flat)
            noise = torch.randn_like(clean_logits) * F.softplus(noise_logits) * self.noise_epsilon
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        # 计算门控权重
        gates = F.softmax(noisy_logits, dim=-1)  # (batch * ..., n_experts)
        
        # 选择top-k专家
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # 重新归一化top-k权重
        top_k_gates = top_k_gates / (torch.sum(top_k_gates, dim=-1, keepdim=True) + 1e-8)
        
        # 计算辅助损失信息
        aux_loss_info = {}
        if self.training:
            # 负载均衡损失
            load = self._gates_to_load(gates)
            aux_loss_info['load'] = load
            aux_loss_info['gates'] = gates
        
        # 恢复原始形状
        batch_size = original_shape[0]
        other_dims = original_shape[1:-1]
        
        if len(other_dims) > 0:
            new_shape = (batch_size,) + other_dims + (self.top_k,)
            top_k_gates = top_k_gates.view(new_shape)
            top_k_indices = top_k_indices.view(new_shape)
        
        return top_k_gates, top_k_indices, aux_loss_info


class MoEAdapter(nn.Module):
    """MoE Adapter：混合专家适配器"""
    
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 n_experts: int = 4,
                 expert_hidden_dim: int = None,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 load_balance_weight: float = 0.01,
                 noisy_gating: bool = True):
        """
        初始化MoE Adapter
        
        Args:
            d_input: 输入维度
            d_output: 输出维度
            n_experts: 专家数量
            expert_hidden_dim: 专家隐层维度
            top_k: 选择的专家数量
            dropout: Dropout概率
            load_balance_weight: 负载均衡损失权重
            noisy_gating: 是否使用噪声门控
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_output = d_output
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # 默认专家隐层维度
        if expert_hidden_dim is None:
            expert_hidden_dim = max(d_input, d_output) * 2
        
        # 路由器
        self.router = Router(
            d_input=d_input,
            n_experts=n_experts,
            top_k=top_k,
            noisy_gating=noisy_gating
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(
                d_input=d_input,
                d_hidden=expert_hidden_dim,
                d_output=d_output,
                dropout=dropout
            )
            for _ in range(n_experts)
        ])
        
        # 残差连接
        if d_input == d_output:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(d_input, d_output)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"MoE Adapter初始化:")
        print(f"  输入维度: {d_input}")
        print(f"  输出维度: {d_output}")
        print(f"  专家数量: {n_experts}")
        print(f"  专家隐层维度: {expert_hidden_dim}")
        print(f"  Top-K: {top_k}")
    
    def _compute_load_balance_loss(self, gates: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        # 计算每个专家的使用频率
        expert_usage = torch.zeros(self.n_experts, device=gates.device)
        
        for i in range(self.n_experts):
            expert_usage[i] = torch.sum((expert_indices == i).float())
        
        # 归一化
        expert_usage = expert_usage / torch.sum(expert_usage)
        
        # 期望的均匀分布
        uniform_dist = torch.ones(self.n_experts, device=gates.device) / self.n_experts
        
        # KL散度作为负载均衡损失
        load_balance_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            uniform_dist,
            reduction='sum'
        )
        
        return load_balance_loss
    
    def forward(self, x: torch.Tensor, return_aux_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, ..., d_input)
            return_aux_loss: 是否返回辅助损失
            
        Returns:
            输出字典，包含主要输出和辅助损失
        """
        original_shape = x.shape
        
        # 路由决策
        expert_weights, expert_indices, aux_info = self.router(x)
        
        # 展平输入用于专家计算
        x_flat = x.view(-1, x.shape[-1])  # (batch * ..., d_input)
        
        # 初始化输出
        output_flat = torch.zeros(x_flat.shape[0], self.d_output, device=x.device, dtype=x.dtype)
        
        # 对于每个选择的专家
        for i in range(self.top_k):
            # 获取当前专家的权重和索引
            current_weights = expert_weights[..., i]  # (...,)
            current_indices = expert_indices[..., i]  # (...,)
            
            # 展平权重和索引
            weights_flat = current_weights.view(-1)  # (batch * ...,)
            indices_flat = current_indices.view(-1)  # (batch * ...,)
            
            # 对于每个专家
            for expert_id in range(self.n_experts):
                # 找到使用当前专家的样本
                expert_mask = (indices_flat == expert_id)
                
                if expert_mask.any():
                    # 获取使用当前专家的输入
                    expert_input = x_flat[expert_mask]  # (n_samples, d_input)
                    expert_weight = weights_flat[expert_mask]  # (n_samples,)
                    
                    # 通过专家网络
                    expert_output = self.experts[expert_id](expert_input)  # (n_samples, d_output)
                    
                    # 加权累加到输出
                    output_flat[expert_mask] += expert_weight.unsqueeze(-1) * expert_output
        
        # 残差连接
        residual = self.residual_projection(x_flat)
        output_flat = output_flat + self.dropout(residual)
        
        # 恢复原始形状
        output_shape = original_shape[:-1] + (self.d_output,)
        output = output_flat.view(output_shape)
        
        result = {'output': output}
        
        # 计算辅助损失
        if return_aux_loss and self.training and 'gates' in aux_info:
            load_balance_loss = self._compute_load_balance_loss(
                aux_info['gates'], 
                expert_indices.view(-1)
            )
            result['aux_loss'] = self.load_balance_weight * load_balance_loss
        
        return result


class MultiTaskMoEAdapter(nn.Module):
    """多任务MoE Adapter：为不同任务使用不同的专家组合"""
    
    def __init__(self,
                 d_input: int,
                 task_configs: Dict[str, Dict],
                 shared_experts: int = 2,
                 task_specific_experts: int = 2,
                 expert_hidden_dim: int = None,
                 dropout: float = 0.1):
        """
        初始化多任务MoE Adapter
        
        Args:
            d_input: 输入维度
            task_configs: 任务配置字典，格式：{task_name: {d_output: int, ...}}
            shared_experts: 共享专家数量
            task_specific_experts: 每个任务特定的专家数量
            expert_hidden_dim: 专家隐层维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.d_input = d_input
        self.task_configs = task_configs
        self.shared_experts = shared_experts
        self.task_specific_experts = task_specific_experts
        
        # 为每个任务创建MoE Adapter
        self.task_adapters = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            d_output = config['d_output']
            n_experts = shared_experts + task_specific_experts
            
            self.task_adapters[task_name] = MoEAdapter(
                d_input=d_input,
                d_output=d_output,
                n_experts=n_experts,
                expert_hidden_dim=expert_hidden_dim,
                top_k=config.get('top_k', 2),
                dropout=dropout
            )
        
        print(f"多任务MoE Adapter初始化:")
        print(f"  任务数量: {len(task_configs)}")
        print(f"  共享专家: {shared_experts}")
        print(f"  任务特定专家: {task_specific_experts}")
        
    def forward(self, x: torch.Tensor, task_name: str) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
            task_name: 任务名称
            
        Returns:
            任务特定的输出
        """
        if task_name not in self.task_adapters:
            raise ValueError(f"未知任务: {task_name}")
        
        return self.task_adapters[task_name](x)
    
    def get_task_output(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """获取指定任务的输出"""
        result = self.forward(x, task_name)
        return result['output']


if __name__ == "__main__":
    # 测试MoE Adapter模块
    print("MoE Adapter模块测试")
    
    # 参数设置
    batch_size = 4
    seq_len = 128
    d_input = 256
    d_output = 128
    
    # 创建模拟输入
    x = torch.randn(batch_size, seq_len, d_input)
    
    # 测试基础MoE Adapter
    print("\n测试基础MoE Adapter:")
    moe_adapter = MoEAdapter(
        d_input=d_input,
        d_output=d_output,
        n_experts=4,
        top_k=2
    )
    
    result = moe_adapter(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {result['output'].shape}")
    if 'aux_loss' in result:
        print(f"辅助损失: {result['aux_loss'].item():.6f}")
    
    # 测试多任务MoE Adapter
    print("\n测试多任务MoE Adapter:")
    task_configs = {
        'task_2a': {'d_output': 4, 'top_k': 2},  # 4类运动想象
        'task_2b': {'d_output': 2, 'top_k': 2},  # 2类运动想象
        'task_eegmmi': {'d_output': 4, 'top_k': 2}  # EEG-MMI 4类
    }
    
    multi_task_moe = MultiTaskMoEAdapter(
        d_input=d_input,
        task_configs=task_configs,
        shared_experts=2,
        task_specific_experts=2
    )
    
    # 测试不同任务
    for task_name in task_configs.keys():
        task_output = multi_task_moe.get_task_output(x, task_name)
        print(f"{task_name} 输出形状: {task_output.shape}")
    
    # 测试不同输入维度
    print("\n测试多维输入:")
    n_channels = 22
    x_multi = torch.randn(batch_size, n_channels, seq_len, d_input)
    
    result_multi = moe_adapter(x_multi)
    print(f"多维输入形状: {x_multi.shape}")
    print(f"多维输出形状: {result_multi['output'].shape}")
    
    print("MoE Adapter测试完成!")

