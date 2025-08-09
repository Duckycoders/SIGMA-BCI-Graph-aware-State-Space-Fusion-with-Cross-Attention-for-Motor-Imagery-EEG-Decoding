#!/usr/bin/env python3
"""
Cross-Attention融合模块
实现S4和Mamba分支之间的交互融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class CrossAttentionFusion(nn.Module):
    """跨注意力融合模块"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 fusion_method: str = 'bidirectional'):
        """
        初始化跨注意力融合模块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout概率
            temperature: 注意力温度参数
            fusion_method: 融合方法 ('bidirectional', 'unidirectional', 'alternating')
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.fusion_method = fusion_method
        
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_head = d_model // n_heads
        
        # S4 → Mamba 注意力
        self.s4_to_mamba_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Mamba → S4 注意力
        self.mamba_to_s4_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 自注意力（可选）
        self.s4_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.mamba_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 归一化层
        self.norm_s4_1 = nn.LayerNorm(d_model)
        self.norm_s4_2 = nn.LayerNorm(d_model)
        self.norm_mamba_1 = nn.LayerNorm(d_model)
        self.norm_mamba_2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ff_s4 = self._build_feedforward(d_model, dropout)
        self.ff_mamba = self._build_feedforward(d_model, dropout)
        
        # 融合投影
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"CrossAttention融合模块初始化:")
        print(f"  模型维度: {d_model}")
        print(f"  注意力头数: {n_heads}")
        print(f"  融合方法: {fusion_method}")
        
    def _build_feedforward(self, d_model: int, dropout: float) -> nn.Module:
        """构建前馈网络"""
        return nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def _apply_cross_attention(self, 
                             query: torch.Tensor,
                             key_value: torch.Tensor,
                             attn_module: nn.MultiheadAttention,
                             mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用跨注意力
        
        Args:
            query: 查询张量 (batch_size, seq_len, d_model)
            key_value: 键值张量 (batch_size, seq_len, d_model)
            attn_module: 注意力模块
            mask: 注意力掩码
            
        Returns:
            (注意力输出, 注意力权重)
        """
        attn_output, attn_weights = attn_module(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=mask,
            need_weights=True
        )
        
        return attn_output, attn_weights
    
    def forward(self, 
                s4_features: torch.Tensor,
                mamba_features: torch.Tensor,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            s4_features: S4分支特征 (batch_size, seq_len, d_model) 或多维度
            mamba_features: Mamba分支特征 (batch_size, seq_len, d_model) 或多维度
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            融合结果字典
        """
        # 处理多维输入
        s4_original_shape = s4_features.shape
        mamba_original_shape = mamba_features.shape
        
        # 展平为 (batch_size * other_dims, seq_len, d_model)
        if s4_features.dim() > 3:
            s4_flat = s4_features.view(-1, s4_features.shape[-2], s4_features.shape[-1])
            mamba_flat = mamba_features.view(-1, mamba_features.shape[-2], mamba_features.shape[-1])
        else:
            s4_flat = s4_features
            mamba_flat = mamba_features
        
        batch_size, seq_len, d_model = s4_flat.shape
        
        if self.fusion_method == 'bidirectional':
            # 双向跨注意力
            s4_enhanced, s4_to_mamba_weights = self._apply_cross_attention(
                s4_flat, mamba_flat, self.s4_to_mamba_attn
            )
            
            mamba_enhanced, mamba_to_s4_weights = self._apply_cross_attention(
                mamba_flat, s4_flat, self.mamba_to_s4_attn
            )
            
            # 残差连接和归一化
            s4_out = self.norm_s4_1(s4_flat + self.dropout(s4_enhanced))
            mamba_out = self.norm_mamba_1(mamba_flat + self.dropout(mamba_enhanced))
            
            # 前馈网络
            s4_ff = self.ff_s4(s4_out)
            mamba_ff = self.ff_mamba(mamba_out)
            
            s4_final = self.norm_s4_2(s4_out + self.dropout(s4_ff))
            mamba_final = self.norm_mamba_2(mamba_out + self.dropout(mamba_ff))
            
        elif self.fusion_method == 'alternating':
            # 交替注意力
            # 第一步：S4 → Mamba
            s4_enhanced, s4_to_mamba_weights = self._apply_cross_attention(
                s4_flat, mamba_flat, self.s4_to_mamba_attn
            )
            s4_intermediate = self.norm_s4_1(s4_flat + self.dropout(s4_enhanced))
            
            # 第二步：增强的S4 → Mamba
            mamba_enhanced, mamba_to_s4_weights = self._apply_cross_attention(
                mamba_flat, s4_intermediate, self.mamba_to_s4_attn
            )
            mamba_final = self.norm_mamba_1(mamba_flat + self.dropout(mamba_enhanced))
            
            # 前馈网络
            s4_ff = self.ff_s4(s4_intermediate)
            s4_final = self.norm_s4_2(s4_intermediate + self.dropout(s4_ff))
            
            mamba_ff = self.ff_mamba(mamba_final)
            mamba_final = self.norm_mamba_2(mamba_final + self.dropout(mamba_ff))
            
        else:  # unidirectional
            # 单向注意力：S4 → Mamba
            s4_enhanced, s4_to_mamba_weights = self._apply_cross_attention(
                s4_flat, mamba_flat, self.s4_to_mamba_attn
            )
            
            s4_final = self.norm_s4_1(s4_flat + self.dropout(s4_enhanced))
            mamba_final = mamba_flat
            
            # 前馈网络
            s4_ff = self.ff_s4(s4_final)
            s4_final = self.norm_s4_2(s4_final + self.dropout(s4_ff))
            
            mamba_to_s4_weights = None
        
        # 融合两个分支
        concatenated = torch.cat([s4_final, mamba_final], dim=-1)  # (batch, seq_len, 2*d_model)
        
        # 门控机制
        gate = torch.sigmoid(self.fusion_gate(concatenated))
        
        # 融合投影
        fused = self.fusion_projection(concatenated)
        fused = gate * fused + (1 - gate) * (s4_final + mamba_final) / 2
        
        # 恢复原始形状
        if len(s4_original_shape) > 3:
            fused = fused.view(*s4_original_shape[:-1], self.d_model)
            s4_final = s4_final.view(*s4_original_shape[:-1], self.d_model)
            mamba_final = mamba_final.view(*mamba_original_shape[:-1], self.d_model)
        
        result = {
            'fused': fused,
            's4_enhanced': s4_final,
            'mamba_enhanced': mamba_final
        }
        
        if return_attention_weights:
            result['attention_weights'] = {
                's4_to_mamba': s4_to_mamba_weights,
                'mamba_to_s4': mamba_to_s4_weights
            }
        
        return result


class MultiScaleFusion(nn.Module):
    """多尺度融合模块：处理FilterBank的多频带融合"""
    
    def __init__(self,
                 d_model: int,
                 n_bands: int = 4,
                 fusion_type: str = 'attention',
                 dropout: float = 0.1):
        """
        初始化多尺度融合模块
        
        Args:
            d_model: 模型维度
            n_bands: 频带数量
            fusion_type: 融合类型 ('attention', 'weighted_sum', 'conv')
            dropout: Dropout概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_bands = n_bands
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 频带间注意力
            self.band_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            
        elif fusion_type == 'weighted_sum':
            # 可学习权重求和
            self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)
            
        elif fusion_type == 'conv':
            # 1D卷积融合
            self.conv_fusion = nn.Conv1d(
                in_channels=n_bands * d_model,
                out_channels=d_model,
                kernel_size=1
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, n_bands, seq_len, d_model)
            
        Returns:
            融合后的张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, n_bands, seq_len, d_model = x.shape
        
        if self.fusion_type == 'attention':
            # 重塑为 (batch_size * seq_len, n_bands, d_model)
            x_reshaped = x.transpose(1, 2).contiguous().view(batch_size * seq_len, n_bands, d_model)
            
            # 频带间注意力
            attn_out, _ = self.band_attention(
                query=x_reshaped,
                key=x_reshaped,
                value=x_reshaped
            )
            
            # 池化到单个频带
            fused = torch.mean(attn_out, dim=1)  # (batch_size * seq_len, d_model)
            fused = fused.view(batch_size, seq_len, d_model)
            
            # 残差连接
            avg_bands = torch.mean(x, dim=1)  # (batch_size, seq_len, d_model)
            fused = self.norm(avg_bands + self.dropout(fused))
            
        elif self.fusion_type == 'weighted_sum':
            # 加权求和
            weights = F.softmax(self.band_weights, dim=0)  # (n_bands,)
            fused = torch.sum(x * weights.view(1, -1, 1, 1), dim=1)  # (batch_size, seq_len, d_model)
            
        elif self.fusion_type == 'conv':
            # 卷积融合
            x_flat = x.view(batch_size, n_bands * d_model, seq_len)  # (batch, n_bands*d_model, seq_len)
            fused = self.conv_fusion(x_flat)  # (batch, d_model, seq_len)
            fused = fused.transpose(-1, -2)  # (batch, seq_len, d_model)
            
        else:
            # 简单平均
            fused = torch.mean(x, dim=1)
        
        return fused


class AdaptiveFusion(nn.Module):
    """自适应融合模块：根据输入动态调整融合策略"""
    
    def __init__(self,
                 d_model: int,
                 fusion_strategies: List[str] = ['cross_attention', 'weighted_sum', 'gated'],
                 dropout: float = 0.1):
        """
        初始化自适应融合模块
        
        Args:
            d_model: 模型维度
            fusion_strategies: 可用的融合策略列表
            dropout: Dropout概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion_strategies = fusion_strategies
        self.n_strategies = len(fusion_strategies)
        
        # 策略选择网络
        self.strategy_selector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_strategies),
            nn.Softmax(dim=-1)
        )
        
        # 不同融合策略的实现
        self.cross_attn = CrossAttentionFusion(d_model, dropout=dropout)
        
        self.weighted_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )
        
        self.gated_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, s4_features: torch.Tensor, mamba_features: torch.Tensor) -> torch.Tensor:
        """自适应融合前向传播"""
        # 计算特征统计用于策略选择
        s4_pooled = torch.mean(s4_features, dim=1)  # (batch, d_model)
        mamba_pooled = torch.mean(mamba_features, dim=1)  # (batch, d_model)
        
        combined_stats = torch.cat([s4_pooled, mamba_pooled], dim=-1)  # (batch, 2*d_model)
        
        # 选择融合策略
        strategy_weights = self.strategy_selector(combined_stats)  # (batch, n_strategies)
        
        # 应用不同策略
        fusion_outputs = []
        
        for i, strategy in enumerate(self.fusion_strategies):
            if strategy == 'cross_attention':
                fusion_result = self.cross_attn(s4_features, mamba_features)
                fusion_outputs.append(fusion_result['fused'])
            elif strategy == 'weighted_sum':
                weighted = self.weighted_fusion(torch.cat([s4_features, mamba_features], dim=-1))
                fusion_outputs.append(weighted)
            elif strategy == 'gated':
                gate = self.gated_fusion(torch.cat([s4_features, mamba_features], dim=-1))
                gated = gate * s4_features + (1 - gate) * mamba_features
                fusion_outputs.append(gated)
        
        # 加权组合不同策略的输出
        fusion_stack = torch.stack(fusion_outputs, dim=0)  # (n_strategies, batch, seq_len, d_model)
        strategy_weights = strategy_weights.T.unsqueeze(-1).unsqueeze(-1)  # (n_strategies, batch, 1, 1)
        
        adaptive_fused = torch.sum(fusion_stack * strategy_weights, dim=0)
        
        return adaptive_fused


if __name__ == "__main__":
    # 测试融合模块
    print("Cross-Attention融合模块测试")
    
    # 参数设置
    batch_size = 4
    seq_len = 256
    d_model = 128
    n_bands = 4
    
    # 创建模拟输入
    s4_features = torch.randn(batch_size, seq_len, d_model)
    mamba_features = torch.randn(batch_size, seq_len, d_model)
    
    # 测试基础融合
    print("\n测试CrossAttention融合:")
    fusion_module = CrossAttentionFusion(d_model=d_model, n_heads=8)
    
    result = fusion_module(s4_features, mamba_features, return_attention_weights=True)
    print(f"S4输入形状: {s4_features.shape}")
    print(f"Mamba输入形状: {mamba_features.shape}")
    print(f"融合输出形状: {result['fused'].shape}")
    print(f"注意力权重: {result['attention_weights']['s4_to_mamba'].shape}")
    
    # 测试多尺度融合
    print("\n测试多尺度融合:")
    multi_band_features = torch.randn(batch_size, n_bands, seq_len, d_model)
    
    multiscale_fusion = MultiScaleFusion(d_model=d_model, n_bands=n_bands, fusion_type='attention')
    multiscale_output = multiscale_fusion(multi_band_features)
    print(f"多频带输入形状: {multi_band_features.shape}")
    print(f"多尺度融合输出形状: {multiscale_output.shape}")
    
    # 测试自适应融合
    print("\n测试自适应融合:")
    adaptive_fusion = AdaptiveFusion(d_model=d_model)
    adaptive_output = adaptive_fusion(s4_features, mamba_features)
    print(f"自适应融合输出形状: {adaptive_output.shape}")
    
    print("融合模块测试完成!")

