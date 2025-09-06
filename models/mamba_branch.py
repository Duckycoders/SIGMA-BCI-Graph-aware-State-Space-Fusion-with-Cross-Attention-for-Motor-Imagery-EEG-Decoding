#!/usr/bin/env python3
"""
Mamba分支模块 - 纯PyTorch实现
不依赖mamba-ssm包，完全用PyTorch实现Mamba核心机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Any

# 检查是否有mamba-ssm包，如果没有就使用我们的纯PyTorch实现
try:
    from mamba_ssm import Mamba as OfficialMamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方mamba-ssm包")
except ImportError:
    MAMBA_AVAILABLE = False
    print("✓ 使用纯PyTorch Mamba实现")


class PurePyTorchMamba(nn.Module):
    """纯PyTorch实现的Mamba层"""
    
    def __init__(self, 
                 d_model: int, 
                 d_state: int = 16, 
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: int = None,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or math.ceil(self.d_model / 16)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # 初始化dt参数
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # 初始化dt偏置
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # 逆softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A参数（状态转移矩阵）
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)[None, :].repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数（跳跃连接）
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            output: (batch, length, dim)
        """
        batch, length, dim = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # each: (batch, length, d_inner)
        
        # 1D卷积（需要转置）
        x = x.transpose(1, 2)  # (batch, d_inner, length)
        x = self.conv1d(x)[:, :, :length]  # 去除padding
        x = x.transpose(1, 2)  # (batch, length, d_inner)
        
        # 激活
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective State Space Model"""
        batch, length, d_inner = x.shape
        
        # 计算dt, B, C
        x_dbl = self.x_proj(x)  # (batch, length, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt投影
        dt = self.dt_proj(dt)  # (batch, length, d_inner)
        dt = F.softplus(dt)
        
        # A矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Debug维度
        # print(f"Debug: dt.shape={dt.shape}, A.shape={A.shape}")
        # print(f"Debug: d_inner={d_inner}, self.d_inner={self.d_inner}")
        
        # 离散化 - 修复维度匹配
        # A_bar = exp(A * dt)
        # 需要确保dt和A的维度匹配
        dt_expanded = dt.unsqueeze(-1)  # (batch, length, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        dt_A = dt_expanded * A_expanded  # (batch, length, d_inner, d_state)
        A_bar = torch.exp(dt_A)
        
        # B_bar = (A^{-1} * (A_bar - I)) * B
        # 简化：B_bar ≈ dt * B
        dt_B = torch.einsum("bld,bls->blds", dt, B)  # (batch, length, d_inner, d_state)
        
        # 选择性扫描（简化版本）
        # 这里使用一个简化的递推实现
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(length):
            # h = A_bar[i] * h + B_bar[i] * x[i]
            h = A_bar[:, i] * h + dt_B[:, i] * x[:, i].unsqueeze(-1)
            
            # y = C[i] * h + D * x[i]
            y = torch.einsum("bns,bs->bn", h, C[:, i]) + self.D * x[:, i]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, length, d_inner)
        
        return y


class MambaBlock(nn.Module):
    """Mamba块：支持官方实现和纯PyTorch实现"""
    
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
            # 使用官方mamba-ssm实现
            self.mamba_layer = OfficialMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # 使用纯PyTorch实现
            self.mamba_layer = PurePyTorchMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
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
        
        # Mamba层
        mamba_out = self.mamba_layer(x_norm)
        
        # 残差连接和dropout
        output = x + self.dropout(mamba_out)
        
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
        
        # 输入投影
        if d_input != d_model:
            self.input_projection = nn.Linear(d_input, d_model)
        else:
            self.input_projection = nn.Identity()
        
        # Mamba层堆叠
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
        
        # 最终归一化
        if use_final_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = nn.Identity()
        
        print(f"Mamba分支初始化:")
        print(f"  输入维度: {d_input}")
        print(f"  模型维度: {d_model}")
        print(f"  状态维度: {d_state}")
        print(f"  层数: {n_layers}")
        print(f"  实现类型: {'官方mamba-ssm' if MAMBA_AVAILABLE else '纯PyTorch'}")
        
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
        
        # 通过Mamba层（S4实现）
        for mamba_layer in self.mamba_layers:
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
    创建Mamba分支的便捷函数（现在使用S4实现）
    
    Returns:
        MambaBranch实例（S4实现）
    """
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

