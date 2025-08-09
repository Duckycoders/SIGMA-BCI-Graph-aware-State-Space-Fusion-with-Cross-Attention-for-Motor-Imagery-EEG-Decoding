#!/usr/bin/env python3
"""
FilterBank模块
实现并行子带滤波，支持多个频带的同时处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from scipy import signal


class FilterBankLayer(nn.Module):
    """FilterBank层：并行子带滤波"""
    
    def __init__(self, 
                 n_channels: int,
                 filter_bands: List[Tuple[float, float]], 
                 sfreq: float = 250.0,
                 filter_order: int = 4,
                 filter_type: str = 'butterworth'):
        """
        初始化FilterBank层
        
        Args:
            n_channels: 输入通道数
            filter_bands: 频带列表，格式[(low1,high1), (low2,high2), ...]
            sfreq: 采样频率
            filter_order: 滤波器阶数
            filter_type: 滤波器类型 ('butterworth', 'conv1d')
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.filter_bands = filter_bands
        self.n_bands = len(filter_bands)
        self.sfreq = sfreq
        self.filter_order = filter_order
        self.filter_type = filter_type
        
        print(f"FilterBank初始化:")
        print(f"  通道数: {n_channels}")
        print(f"  频带数: {self.n_bands}")
        print(f"  频带: {filter_bands}")
        print(f"  滤波器类型: {filter_type}")
        
        if filter_type == 'conv1d':
            self._build_conv1d_filters()
        elif filter_type == 'butterworth':
            self._build_butterworth_filters()
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
    
    def _build_conv1d_filters(self):
        """构建基于Conv1d的并行滤波器"""
        # 为每个频带创建一个Conv1d层
        self.conv_filters = nn.ModuleList()
        
        # 滤波器长度（通常为采样频率的1/4到1/2）
        filter_length = int(self.sfreq // 2)
        if filter_length % 2 == 0:
            filter_length += 1  # 确保奇数长度
        
        for i, (low_freq, high_freq) in enumerate(self.filter_bands):
            # 创建带通滤波器核
            filter_kernel = self._create_bandpass_kernel(
                low_freq, high_freq, filter_length
            )
            
            # 创建Conv1d层，使用分组卷积对每个通道独立滤波
            conv_layer = nn.Conv1d(
                in_channels=self.n_channels,
                out_channels=self.n_channels, 
                kernel_size=filter_length,
                groups=self.n_channels,  # 分组卷积，每组一个通道
                padding=filter_length // 2,
                bias=False
            )
            
            # 初始化权重为滤波器核
            with torch.no_grad():
                for ch in range(self.n_channels):
                    conv_layer.weight[ch, 0, :] = torch.FloatTensor(filter_kernel)
            
            # 冻结滤波器权重（不可训练）
            conv_layer.weight.requires_grad = False
            
            self.conv_filters.append(conv_layer)
    
    def _create_bandpass_kernel(self, low_freq: float, high_freq: float, 
                               length: int) -> np.ndarray:
        """创建带通滤波器核"""
        nyquist = self.sfreq / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # 使用Butterworth滤波器设计
        b, a = signal.butter(
            self.filter_order, 
            [low_norm, high_norm], 
            btype='band'
        )
        
        # 转换为FIR滤波器（截断冲激响应）
        impulse = np.zeros(length)
        impulse[length // 2] = 1
        kernel = signal.lfilter(b, a, impulse)
        
        # 应用窗函数
        try:
            window = signal.windows.hann(length)
        except AttributeError:
            window = signal.hann(length)
        kernel = kernel * window
        
        # 归一化
        kernel = kernel / np.sum(np.abs(kernel))
        
        return kernel
    
    def _build_butterworth_filters(self):
        """构建Butterworth滤波器系数（用于后续IIR滤波）"""
        self.filter_coeffs = []
        
        nyquist = self.sfreq / 2
        
        for low_freq, high_freq in self.filter_bands:
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # 设计Butterworth带通滤波器
            b, a = signal.butter(
                self.filter_order,
                [low_norm, high_norm],
                btype='band'
            )
            
            self.filter_coeffs.append((b, a))
    
    def _apply_butterworth_filter(self, x: torch.Tensor, 
                                 band_idx: int) -> torch.Tensor:
        """应用Butterworth滤波器"""
        b, a = self.filter_coeffs[band_idx]
        
        # 转换为numpy进行滤波
        x_np = x.detach().cpu().numpy()
        
        # 对每个样本和通道应用滤波
        filtered = np.zeros_like(x_np)
        batch_size, n_channels, n_samples = x_np.shape
        
        for batch in range(batch_size):
            for ch in range(n_channels):
                # 使用filtfilt进行零相位滤波
                filtered[batch, ch, :] = signal.filtfilt(
                    b, a, x_np[batch, ch, :]
                )
        
        return torch.FloatTensor(filtered).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, n_channels, n_samples)
            
        Returns:
            FilterBank输出，形状为 (batch_size, n_bands, n_channels, n_samples)
        """
        batch_size, n_channels, n_samples = x.shape
        
        # 初始化输出张量
        output = torch.zeros(
            batch_size, self.n_bands, n_channels, n_samples,
            device=x.device, dtype=x.dtype
        )
        
        if self.filter_type == 'conv1d':
            # 使用Conv1d滤波器
            for i, conv_filter in enumerate(self.conv_filters):
                filtered = conv_filter(x)  # (batch_size, n_channels, n_samples)
                output[:, i, :, :] = filtered
                
        elif self.filter_type == 'butterworth':
            # 使用Butterworth滤波器
            for i in range(self.n_bands):
                filtered = self._apply_butterworth_filter(x, i)
                output[:, i, :, :] = filtered
        
        return output


class AdaptiveFilterBank(nn.Module):
    """自适应FilterBank：可学习的频带划分"""
    
    def __init__(self, 
                 n_channels: int,
                 n_bands: int = 4,
                 sfreq: float = 250.0,
                 freq_range: Tuple[float, float] = (4.0, 45.0),
                 learnable: bool = True):
        """
        初始化自适应FilterBank
        
        Args:
            n_channels: 输入通道数
            n_bands: 频带数量
            sfreq: 采样频率
            freq_range: 频率范围
            learnable: 是否可学习频带边界
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.sfreq = sfreq
        self.freq_range = freq_range
        self.learnable = learnable
        
        # 初始化频带边界
        freq_boundaries = np.linspace(freq_range[0], freq_range[1], n_bands + 1)
        
        if learnable:
            # 可学习的频带边界（使用sigmoid确保单调性）
            self.freq_boundaries = nn.Parameter(
                torch.FloatTensor(freq_boundaries)
            )
        else:
            # 固定频带边界
            self.register_buffer(
                'freq_boundaries', 
                torch.FloatTensor(freq_boundaries)
            )
        
        # 使用Conv1d实现滤波
        self.filter_length = int(sfreq // 2)
        if self.filter_length % 2 == 0:
            self.filter_length += 1
            
        # 为每个频带创建可学习的滤波器
        self.conv_filters = nn.ModuleList([
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=self.filter_length,
                groups=n_channels,
                padding=self.filter_length // 2,
                bias=False
            )
            for _ in range(n_bands)
        ])
        
        self._initialize_filters()
    
    def _initialize_filters(self):
        """初始化滤波器权重"""
        for i, conv_filter in enumerate(self.conv_filters):
            # 获取当前频带
            low_freq = float(self.freq_boundaries[i])
            high_freq = float(self.freq_boundaries[i + 1])
            
            # 创建初始滤波器核
            filter_kernel = self._create_bandpass_kernel(low_freq, high_freq)
            
            # 初始化权重
            with torch.no_grad():
                for ch in range(self.n_channels):
                    conv_filter.weight[ch, 0, :] = torch.FloatTensor(filter_kernel)
    
    def _create_bandpass_kernel(self, low_freq: float, high_freq: float) -> np.ndarray:
        """创建带通滤波器核"""
        nyquist = self.sfreq / 2
        low_norm = max(0.01, low_freq / nyquist)  # 避免DC
        high_norm = min(0.99, high_freq / nyquist)  # 避免Nyquist
        
        # 使用Butterworth滤波器
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # 转换为FIR
        impulse = np.zeros(self.filter_length)
        impulse[self.filter_length // 2] = 1
        kernel = signal.lfilter(b, a, impulse)
        
        # 应用窗函数
        try:
            window = signal.windows.hann(self.filter_length)
        except AttributeError:
            window = signal.hann(self.filter_length)
        kernel = kernel * window
        
        # 归一化
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, n_channels, n_samples)
            
        Returns:
            FilterBank输出，形状为 (batch_size, n_bands, n_channels, n_samples)
        """
        batch_size, n_channels, n_samples = x.shape
        
        # 如果频带边界是可学习的，重新初始化滤波器
        if self.learnable and self.training:
            # 确保频带边界单调递增
            sorted_boundaries = torch.sort(self.freq_boundaries)[0]
            
            # 更新滤波器（仅在训练时）
            for i, conv_filter in enumerate(self.conv_filters):
                if i < len(sorted_boundaries) - 1:
                    low_freq = float(sorted_boundaries[i])
                    high_freq = float(sorted_boundaries[i + 1])
                    
                    if high_freq > low_freq + 1.0:  # 确保最小带宽
                        filter_kernel = self._create_bandpass_kernel(low_freq, high_freq)
                        
                        with torch.no_grad():
                            for ch in range(n_channels):
                                conv_filter.weight[ch, 0, :] = torch.FloatTensor(
                                    filter_kernel
                                ).to(conv_filter.weight.device)
        
        # 应用滤波
        output = torch.zeros(
            batch_size, self.n_bands, n_channels, n_samples,
            device=x.device, dtype=x.dtype
        )
        
        for i, conv_filter in enumerate(self.conv_filters):
            filtered = conv_filter(x)
            output[:, i, :, :] = filtered
        
        return output


if __name__ == "__main__":
    # 测试FilterBank
    print("FilterBank模块测试")
    
    # 参数设置
    batch_size = 4
    n_channels = 22
    n_samples = 1000
    sfreq = 250.0
    
    # 创建模拟输入
    x = torch.randn(batch_size, n_channels, n_samples)
    
    # 测试固定FilterBank
    print("\n测试固定FilterBank:")
    filter_bands = [(4, 8), (8, 14), (14, 30), (30, 45)]
    
    filterbank = FilterBankLayer(
        n_channels=n_channels,
        filter_bands=filter_bands,
        sfreq=sfreq,
        filter_type='conv1d'
    )
    
    output = filterbank(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试自适应FilterBank
    print("\n测试自适应FilterBank:")
    adaptive_fb = AdaptiveFilterBank(
        n_channels=n_channels,
        n_bands=4,
        sfreq=sfreq,
        learnable=True
    )
    
    output_adaptive = adaptive_fb(x)
    print(f"自适应FilterBank输出形状: {output_adaptive.shape}")
    print(f"频带边界: {adaptive_fb.freq_boundaries.data}")
    
    print("FilterBank测试完成!")
