#!/usr/bin/env python3
"""
Mamba分支模块
优先使用mamba-ssm包，安装失败时自动回退到S4-only模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Any
import warnings
import logging

# 尝试导入Mamba
MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    MAMBA_AVAILABLE = True
    print("Mamba-SSM 成功导入")
except ImportError as e:
    print(f"警告: Mamba-SSM 导入失败 - {e}")
    print("将使用S4-only回退模式")
    # 导入S4作为回退
    from .s4_branch import S4Block
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    """Mamba块：原生Mamba实现的封装"""
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_fast_path: bool = True):
        """
        初始化Mamba块
        
        Args:
            d_model: 模型维度
            d_state: 状态空间维度
            d_conv: 1D卷积核大小
            expand: 扩展因子
            dropout: Dropout概率
            use_fast_path: 是否使用快速路径
        """
        super().__init__()
        
        self.d_model = d_model
        
        if MAMBA_AVAILABLE:
            # 使用原生Mamba
            self.mamba_layer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=use_fast_path
            )
        else:
            # 回退到S4
            print("警告: 使用S4作为Mamba的回退实现")
            self.mamba_layer = S4Block(
                d_model=d_model,
                d_state=d_state * 4,  # S4使用更大的状态维度
                dropout=dropout
            )
        
        # 归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 预归一化
        x_norm = self.norm(x)
        
        # Mamba/S4层
        if MAMBA_AVAILABLE:
            # 原生Mamba
            mamba_out = self.mamba_layer(x_norm)
        else:
            # S4回退
            mamba_out = self.mamba_layer(x_norm)
        
        # 残差连接和dropout
        output = x + self.dropout(mamba_out)
        
        return output


class SimpleMambaImplementation(nn.Module):
    """简化的Mamba实现（当mamba-ssm不可用时的备选）"""
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        """
        简化Mamba实现
        
        Args:
            d_model: 模型维度
            d_state: 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # 状态空间矩阵
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """简化的Mamba前向传播"""
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影和分割
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # 1D卷积
        x = x.transpose(-1, -2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :seq_len]  # 移除padding
        x = x.transpose(-1, -2)  # (B, L, d_inner)
        x = F.silu(x)
        
        # SSM计算（简化版）
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 选择性机制
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B = x_dbl.chunk(2, dim=-1)  # (B, L, d_state) each
        
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # 离散化（简化）
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)  # (B, L, d_inner, d_state)
        
        # 状态空间递推（简化为并行扫描）
        # 这里使用简化的计算，实际Mamba使用更复杂的并行扫描算法
        states = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # 状态更新
            states = deltaA[:, t] * states + deltaB[:, t] * x[:, t].unsqueeze(-1)
            # 输出计算
            C = B[:, t].unsqueeze(1)  # (B, 1, d_state)
            y = torch.sum(C * states, dim=-1)  # (B, d_inner)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # 跳跃连接
        y = y + self.D * x
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output


class MambaBranch(nn.Module):
    """Mamba分支：多层Mamba块的堆叠"""
    
    def __init__(self,
                 d_input: int,
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 use_simple_fallback: bool = False,
                 use_final_norm: bool = True):
        """
        初始化Mamba分支
        
        Args:
            d_input: 输入维度
            d_model: 模型隐层维度
            d_state: 状态空间维度
            d_conv: 1D卷积核大小
            expand: 扩展因子
            n_layers: Mamba层数
            dropout: Dropout概率
            use_simple_fallback: 强制使用简化实现
            use_final_norm: 是否使用最终归一化
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_final_norm = use_final_norm
        
        # 检查Mamba可用性
        self.mamba_available = MAMBA_AVAILABLE and not use_simple_fallback
        
        # 输入投影
        self.input_projection = nn.Linear(d_input, d_model)
        
        # Mamba层堆叠
        if self.mamba_available:
            print(f"使用原生Mamba实现，{n_layers}层")
            self.mamba_layers = nn.ModuleList([
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ])
        else:
            if use_simple_fallback:
                print(f"使用简化Mamba实现，{n_layers}层")
                self.mamba_layers = nn.ModuleList([
                    SimpleMambaImplementation(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                    for _ in range(n_layers)
                ])
                # 添加归一化层
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(d_model) for _ in range(n_layers)
                ])
            else:
                print(f"使用S4回退实现，{n_layers}层")
                from .s4_branch import S4Block
                self.mamba_layers = nn.ModuleList([
                    S4Block(
                        d_model=d_model,
                        d_state=d_state * 4,
                        dropout=dropout
                    )
                    for _ in range(n_layers)
                ])
        
        # 最终归一化
        if use_final_norm:
            self.final_norm = nn.LayerNorm(d_model)
        
        print(f"Mamba分支初始化:")
        print(f"  输入维度: {d_input}")
        print(f"  模型维度: {d_model}")
        print(f"  状态维度: {d_state}")
        print(f"  层数: {n_layers}")
        print(f"  实现类型: {'原生Mamba' if self.mamba_available else '回退实现'}")
        
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
        
        # 通过Mamba层
        for i, mamba_layer in enumerate(self.mamba_layers):
            if hasattr(self, 'layer_norms') and not self.mamba_available:
                # 简化实现需要手动添加残差连接和归一化
                residual = x
                x = self.layer_norms[i](x)
                x = mamba_layer(x)
                x = residual + x
            else:
                x = mamba_layer(x)
        
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


def create_mamba_branch(**kwargs) -> MambaBranch:
    """
    创建Mamba分支的便捷函数，自动处理可用性检查
    
    Returns:
        MambaBranch实例
    """
    if not MAMBA_AVAILABLE:
        print("注意: Mamba-SSM不可用，将使用回退实现")
        kwargs['use_simple_fallback'] = kwargs.get('use_simple_fallback', False)
    
    return MambaBranch(**kwargs)


if __name__ == "__main__":
    # 测试Mamba分支
    print("Mamba分支模块测试")
    print(f"Mamba可用性: {MAMBA_AVAILABLE}")
    
    # 参数设置
    batch_size = 4
    seq_len = 256
    d_input = 64
    d_model = 128
    
    # 创建Mamba分支
    mamba_branch = create_mamba_branch(
        d_input=d_input,
        d_model=d_model,
        d_state=16,
        n_layers=2
    )
    
    # 测试不同输入格式
    print("\n测试标准输入:")
    x1 = torch.randn(batch_size, seq_len, d_input)
    output1 = mamba_branch(x1)
    print(f"输入形状: {x1.shape}")
    print(f"输出形状: {output1.shape}")
    
    print("\n测试多通道输入:")
    n_channels = 22
    x2 = torch.randn(batch_size, n_channels, seq_len, d_input)
    output2 = mamba_branch(x2)
    print(f"输入形状: {x2.shape}")
    print(f"输出形状: {output2.shape}")
    
    print("\n测试FilterBank输入:")
    n_bands = 4
    x3 = torch.randn(batch_size, n_bands, n_channels, seq_len, d_input)
    output3 = mamba_branch(x3)
    print(f"输入形状: {x3.shape}")
    print(f"输出形状: {output3.shape}")
    
    # 测试简化实现
    print("\n测试简化Mamba实现:")
    mamba_simple = MambaBranch(
        d_input=d_input,
        d_model=d_model,
        n_layers=1,
        use_simple_fallback=True
    )
    
    output_simple = mamba_simple(x1)
    print(f"简化实现输出形状: {output_simple.shape}")
    
    print("Mamba分支测试完成!")

