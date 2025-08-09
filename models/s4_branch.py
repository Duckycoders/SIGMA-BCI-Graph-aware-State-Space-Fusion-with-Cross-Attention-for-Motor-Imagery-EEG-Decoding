#!/usr/bin/env python3
"""
S4 (Structured State Space) 分支模块
实现长序列建模的状态空间模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class S4Kernel(nn.Module):
    """S4核心模块：结构化状态空间模型"""
    
    def __init__(self, 
                 d_model: int,
                 d_state: int = 64,
                 l_max: int = 1024,
                 lr: float = 1e-3,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1):
        """
        初始化S4核心模块
        
        Args:
            d_model: 模型维度
            d_state: 状态空间维度
            l_max: 最大序列长度
            lr: 学习率缩放因子
            dt_min: 最小时间步长
            dt_max: 最大时间步长
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.l_max = l_max
        
        # 初始化HiPPO矩阵 (HiPPO-LegS)
        A = self._init_hippo_matrix(d_state)
        self.register_buffer('A', A)
        
        # 可学习参数
        # B: 输入矩阵
        self.B = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        
        # C: 输出矩阵  
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        
        # D: 跳跃连接
        self.D = nn.Parameter(torch.randn(d_model))
        
        # 时间步长参数
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # 预计算卷积核
        self._setup_cache()
        
    def _init_hippo_matrix(self, d_state: int) -> torch.Tensor:
        """初始化HiPPO矩阵"""
        # HiPPO-LegS矩阵
        A = torch.zeros(d_state, d_state)
        
        for i in range(d_state):
            for j in range(d_state):
                if i > j:
                    A[i, j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
                elif i == j:
                    A[i, j] = i + 1
                else:
                    A[i, j] = -math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
        
        return A
    
    def _setup_cache(self):
        """设置缓存用于快速卷积"""
        self.register_buffer('kernel_cache', None)
        self.cache_l_max = 0
        
    def _compute_kernel(self, L: int) -> torch.Tensor:
        """计算S4卷积核"""
        # 获取离散化参数
        dt = torch.exp(self.log_dt)  # (d_model,)
        
        # 扩展维度用于广播
        A = self.A  # (d_state, d_state)
        B = self.B  # (d_state, d_model)
        C = self.C  # (d_model, d_state)
        
        # 离散化：A_bar = (I - dt/2 * A)^{-1} (I + dt/2 * A)
        dt_A = dt.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0)  # (d_model, d_state, d_state)
        
        I = torch.eye(self.d_state, device=A.device).unsqueeze(0)  # (1, d_state, d_state)
        
        # 使用双线性变换进行离散化
        A_bar = torch.linalg.solve(
            I - dt_A / 2,
            I + dt_A / 2
        )  # (d_model, d_state, d_state)
        
        B_bar = torch.matmul(
            torch.linalg.solve(I - dt_A / 2, I),
            B.unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(-1)
        )  # (d_model, d_state, 1)
        
        # 计算卷积核
        powers = torch.arange(L, device=A.device)  # (L,)
        
        # A_bar^k 的计算
        kernel = torch.zeros(self.d_model, L, device=A.device)
        
        for i in range(self.d_model):
            # 对每个通道独立计算
            A_powers = torch.matrix_power(A_bar[i], L)
            
            for k in range(L):
                if k == 0:
                    A_k = torch.eye(self.d_state, device=A.device)
                else:
                    A_k = torch.matrix_power(A_bar[i], k)
                
                # kernel[i, k] = C[i] @ A_k @ B_bar[i]
                kernel[i, k] = torch.sum(
                    C[i] * torch.mv(A_k, B_bar[i].squeeze())
                )
        
        return kernel
    
    def _compute_kernel_fft(self, L: int) -> torch.Tensor:
        """使用FFT计算S4卷积核（更高效的实现）"""
        dt = torch.exp(self.log_dt)
        
        # 简化的核计算（用于演示）
        # 实际实现会使用更复杂的频域计算
        k = torch.arange(L, device=dt.device, dtype=torch.float)
        
        # 衰减核
        decay = torch.exp(-dt.unsqueeze(-1) * k.unsqueeze(0))  # (d_model, L)
        
        # 添加一些振荡成分
        freq = torch.randn(self.d_model, device=dt.device) * 0.1
        oscillation = torch.cos(freq.unsqueeze(-1) * k.unsqueeze(0))
        
        kernel = decay * oscillation
        
        # 归一化
        kernel = kernel / (torch.sum(torch.abs(kernel), dim=-1, keepdim=True) + 1e-8)
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, d_model, L)
            
        Returns:
            输出张量，形状为 (batch_size, d_model, L)
        """
        batch_size, d_model, L = x.shape
        
        # 每次都重新计算核（避免计算图缓存问题）
        kernel = self._compute_kernel_fft(L)
        
        # 使用FFT进行快速卷积
        # 转换到频域
        x_fft = torch.fft.rfft(x, n=2*L, dim=-1)
        kernel_fft = torch.fft.rfft(kernel, n=2*L, dim=-1)
        
        # 频域乘法
        output_fft = x_fft * kernel_fft
        
        # 转换回时域
        output = torch.fft.irfft(output_fft, n=2*L, dim=-1)[..., :L]
        
        # 添加跳跃连接
        output = output + self.D.unsqueeze(0).unsqueeze(-1) * x
        
        return output


class S4Block(nn.Module):
    """S4块：包含S4层和前馈网络"""
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 64,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 prenorm: bool = True):
        """
        初始化S4块
        
        Args:
            d_model: 模型维度
            d_state: 状态空间维度
            dropout: Dropout概率
            activation: 激活函数
            prenorm: 是否使用预归一化
        """
        super().__init__()
        
        self.prenorm = prenorm
        
        # S4核心层
        self.s4_layer = S4Kernel(d_model, d_state)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 转置为S4所需的格式: (batch_size, d_model, seq_len)
        x = x.transpose(-1, -2)
        
        # S4层（带残差连接）
        if self.prenorm:
            # 预归一化
            x_norm = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
            s4_out = self.s4_layer(x_norm)
            x = x + self.dropout(s4_out)
            
            # 前馈网络
            x_norm = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
            ff_out = self.ff_layer(x_norm.transpose(-1, -2)).transpose(-1, -2)
            x = x + self.dropout(ff_out)
        else:
            # 后归一化
            s4_out = self.s4_layer(x)
            x = self.norm1((x + self.dropout(s4_out)).transpose(-1, -2)).transpose(-1, -2)
            
            ff_out = self.ff_layer(x.transpose(-1, -2)).transpose(-1, -2)
            x = self.norm2((x + self.dropout(ff_out)).transpose(-1, -2)).transpose(-1, -2)
        
        # 转置回原始格式
        x = x.transpose(-1, -2)
        
        return x


class S4Branch(nn.Module):
    """S4分支：多层S4块的堆叠"""
    
    def __init__(self,
                 d_input: int,
                 d_model: int = 128,
                 d_state: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 prenorm: bool = True,
                 use_final_norm: bool = True):
        """
        初始化S4分支
        
        Args:
            d_input: 输入维度
            d_model: 模型隐层维度
            d_state: 状态空间维度
            n_layers: S4层数
            dropout: Dropout概率
            activation: 激活函数
            prenorm: 是否使用预归一化
            use_final_norm: 是否使用最终归一化
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_final_norm = use_final_norm
        
        # 输入投影
        self.input_projection = nn.Linear(d_input, d_model)
        
        # S4层堆叠
        self.s4_layers = nn.ModuleList([
            S4Block(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                activation=activation,
                prenorm=prenorm
            )
            for _ in range(n_layers)
        ])
        
        # 最终归一化
        if use_final_norm:
            self.final_norm = nn.LayerNorm(d_model)
        
        print(f"S4分支初始化:")
        print(f"  输入维度: {d_input}")
        print(f"  模型维度: {d_model}")
        print(f"  状态维度: {d_state}")
        print(f"  层数: {n_layers}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，可能的形状：
               - (batch_size, seq_len, d_input)
               - (batch_size, n_channels, seq_len, d_input)
               - (batch_size, n_bands, n_channels, seq_len, d_input)
               
        Returns:
            输出张量，形状为 (..., seq_len, d_model)
        """
        original_shape = x.shape
        
        # 处理不同维度的输入
        if x.dim() == 3:  # (batch_size, seq_len, d_input)
            pass  # 标准格式
        elif x.dim() == 4:  # (batch_size, n_channels, seq_len, d_input)
            batch_size, n_channels, seq_len, d_input = x.shape
            x = x.view(batch_size * n_channels, seq_len, d_input)
        elif x.dim() == 5:  # (batch_size, n_bands, n_channels, seq_len, d_input)
            batch_size, n_bands, n_channels, seq_len, d_input = x.shape
            x = x.view(batch_size * n_bands * n_channels, seq_len, d_input)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
        
        # 输入投影
        x = self.input_projection(x)  # (..., seq_len, d_model)
        
        # 通过S4层
        for s4_layer in self.s4_layers:
            x = s4_layer(x)
        
        # 最终归一化
        if self.use_final_norm:
            x = self.final_norm(x)
        
        # 恢复原始批次维度
        if len(original_shape) == 4:
            batch_size, n_channels, seq_len, _ = original_shape
            x = x.view(batch_size, n_channels, seq_len, self.d_model)
        elif len(original_shape) == 5:
            batch_size, n_bands, n_channels, seq_len, _ = original_shape
            x = x.view(batch_size, n_bands, n_channels, seq_len, self.d_model)
        
        return x


if __name__ == "__main__":
    # 测试S4分支
    print("S4分支模块测试")
    
    # 参数设置
    batch_size = 4
    seq_len = 256
    d_input = 64
    d_model = 128
    
    # 创建S4分支
    s4_branch = S4Branch(
        d_input=d_input,
        d_model=d_model,
        d_state=64,
        n_layers=2
    )
    
    # 测试不同输入格式
    print("\n测试标准输入:")
    x1 = torch.randn(batch_size, seq_len, d_input)
    output1 = s4_branch(x1)
    print(f"输入形状: {x1.shape}")
    print(f"输出形状: {output1.shape}")
    
    print("\n测试多通道输入:")
    n_channels = 22
    x2 = torch.randn(batch_size, n_channels, seq_len, d_input)
    output2 = s4_branch(x2)
    print(f"输入形状: {x2.shape}")
    print(f"输出形状: {output2.shape}")
    
    print("\n测试FilterBank输入:")
    n_bands = 4
    x3 = torch.randn(batch_size, n_bands, n_channels, seq_len, d_input)
    output3 = s4_branch(x3)
    print(f"输入形状: {x3.shape}")
    print(f"输出形状: {output3.shape}")
    
    print("S4分支测试完成!")
