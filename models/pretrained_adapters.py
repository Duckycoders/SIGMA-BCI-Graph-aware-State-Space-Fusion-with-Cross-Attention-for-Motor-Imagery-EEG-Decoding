#!/usr/bin/env python3
"""
预训练模型适配器
支持从HuggingFace Hub和其他来源加载预训练模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, List
import warnings
import os
from pathlib import Path

# 尝试导入transformers
try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
    print("Transformers库可用，支持HuggingFace预训练模型")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: Transformers库不可用，无法使用HuggingFace预训练模型")


class EEGTransformerAdapter(nn.Module):
    """EEG Transformer预训练模型适配器"""
    
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 n_channels: int = 22,
                 n_samples: int = 1000,
                 d_model: int = 128,
                 freeze_pretrained: bool = False,
                 adapter_layers: int = 2):
        """
        初始化EEG Transformer适配器
        
        Args:
            model_name: HuggingFace模型名称
            n_channels: EEG通道数
            n_samples: 时间采样点数
            d_model: 输出特征维度
            freeze_pretrained: 是否冻结预训练权重
            adapter_layers: 适配器层数
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        self.freeze_pretrained = freeze_pretrained
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers库: pip install transformers")
        
        # 加载预训练配置和模型
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.pretrained_model = AutoModel.from_pretrained(model_name)
            
            print(f"成功加载预训练模型: {model_name}")
            print(f"模型配置: hidden_size={self.config.hidden_size}")
            
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            # 回退到随机初始化
            self.config = None
            self.pretrained_model = None
        
        # EEG输入投影层
        pretrained_dim = self.config.hidden_size if self.config else 768
        self.eeg_projection = nn.Sequential(
            nn.Linear(n_channels, pretrained_dim),
            nn.LayerNorm(pretrained_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_samples, pretrained_dim) * 0.02
        )
        
        # 适配器层
        adapter_layers_list = []
        for _ in range(adapter_layers):
            adapter_layers_list.extend([
                nn.Linear(pretrained_dim, pretrained_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.adapter = nn.Sequential(*adapter_layers_list)
        
        # 输出投影
        self.output_projection = nn.Linear(pretrained_dim, d_model)
        
        # 冻结预训练权重
        if freeze_pretrained and self.pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            print("预训练模型权重已冻结")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: EEG输入 (batch_size, n_channels, n_samples)
            
        Returns:
            特征输出 (batch_size, n_samples, d_model)
        """
        batch_size, n_channels, n_samples = x.shape
        
        # 转置为 (batch_size, n_samples, n_channels)
        x = x.transpose(1, 2)
        
        # EEG投影到预训练模型维度
        x = self.eeg_projection(x)  # (batch_size, n_samples, pretrained_dim)
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :n_samples, :]
        
        # 通过预训练模型
        if self.pretrained_model:
            # 对于某些模型可能需要attention_mask
            attention_mask = torch.ones(batch_size, n_samples, device=x.device)
            
            outputs = self.pretrained_model(
                inputs_embeds=x,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 使用最后一层隐状态
            x = outputs.last_hidden_state
        
        # 适配器处理
        x = self.adapter(x)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x


class BraindecodePretrained(nn.Module):
    """Braindecode预训练模型适配器"""
    
    def __init__(self,
                 model_type: str = "eegnet",
                 n_channels: int = 22,
                 n_samples: int = 1000,
                 d_model: int = 128):
        """
        初始化Braindecode预训练模型适配器
        
        Args:
            model_type: 模型类型 ('eegnet', 'shallow_fbcsp', 'deep4')
            n_channels: EEG通道数
            n_samples: 时间采样点数
            d_model: 输出特征维度
        """
        super().__init__()
        
        try:
            from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net
            
            if model_type == "eegnet":
                self.backbone = EEGNetv4(
                    n_chans=n_channels,
                    n_outputs=4,  # 临时设置，后面会替换分类头
                    n_times=n_samples,
                    final_conv_length='auto'
                )
                backbone_dim = 4  # EEGNet的输出维度
                
            elif model_type == "shallow_fbcsp":
                self.backbone = ShallowFBCSPNet(
                    n_chans=n_channels,
                    n_outputs=4,
                    n_times=n_samples,
                    final_conv_length='auto'
                )
                backbone_dim = 4
                
            elif model_type == "deep4":
                self.backbone = Deep4Net(
                    n_chans=n_channels,
                    n_outputs=4,
                    n_times=n_samples,
                    final_conv_length='auto'
                )
                backbone_dim = 4
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 移除原始分类头，添加特征提取
            self.feature_extractor = nn.Sequential(
                self.backbone,
                nn.Linear(backbone_dim, d_model)
            )
            
            print(f"成功初始化Braindecode {model_type}模型")
            
        except ImportError:
            print("警告: Braindecode库不可用，使用简单CNN代替")
            self.feature_extractor = self._create_simple_cnn(n_channels, d_model)
    
    def _create_simple_cnn(self, n_channels: int, d_model: int) -> nn.Module:
        """创建简单的CNN作为回退"""
        return nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(32 * 128, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: EEG输入 (batch_size, n_channels, n_samples)
            
        Returns:
            特征输出 (batch_size, d_model)
        """
        return self.feature_extractor(x)


class PretrainedModelLoader:
    """预训练模型加载器"""
    
    @staticmethod
    def load_from_huggingface(model_name: str, 
                             n_channels: int = 22,
                             n_samples: int = 1000,
                             d_model: int = 128,
                             **kwargs) -> EEGTransformerAdapter:
        """从HuggingFace加载预训练模型"""
        return EEGTransformerAdapter(
            model_name=model_name,
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=d_model,
            **kwargs
        )
    
    @staticmethod
    def load_from_braindecode(model_type: str,
                             n_channels: int = 22,
                             n_samples: int = 1000,
                             d_model: int = 128) -> BraindecodePretrained:
        """从Braindecode加载预训练模型"""
        return BraindecodePretrained(
            model_type=model_type,
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=d_model
        )
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str,
                       model: nn.Module,
                       strict: bool = False) -> nn.Module:
        """加载本地检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试加载权重
        try:
            model.load_state_dict(state_dict, strict=strict)
            print("成功加载完整模型权重")
        except RuntimeError as e:
            if not strict:
                print(f"部分权重加载失败，继续使用可用权重: {e}")
                # 只加载匹配的权重
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                print(f"成功加载 {len(filtered_dict)}/{len(state_dict)} 个权重参数")
            else:
                raise e
        
        return model


# 预训练模型配置
PRETRAINED_CONFIGS = {
    'huggingface': {
        'distilbert': 'distilbert-base-uncased',
        'bert_tiny': 'prajjwal1/bert-tiny',
        'gpt2_small': 'gpt2',
        'roberta_base': 'roberta-base'
    },
    'braindecode': {
        'eegnet': 'eegnet',
        'shallow_fbcsp': 'shallow_fbcsp',
        'deep4': 'deep4'
    }
}


def create_pretrained_backbone(pretrained_type: str,
                              model_name: str,
                              n_channels: int = 22,
                              n_samples: int = 1000,
                              d_model: int = 128,
                              **kwargs) -> nn.Module:
    """
    创建预训练主干网络
    
    Args:
        pretrained_type: 预训练类型 ('huggingface', 'braindecode', 'checkpoint')
        model_name: 模型名称或路径
        n_channels: EEG通道数
        n_samples: 时间采样点数
        d_model: 输出特征维度
        **kwargs: 其他参数
        
    Returns:
        预训练模型实例
    """
    if pretrained_type == 'huggingface':
        return PretrainedModelLoader.load_from_huggingface(
            model_name=model_name,
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=d_model,
            **kwargs
        )
    
    elif pretrained_type == 'braindecode':
        return PretrainedModelLoader.load_from_braindecode(
            model_type=model_name,
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=d_model
        )
    
    elif pretrained_type == 'checkpoint':
        # 这种情况需要先创建模型，然后加载权重
        raise ValueError("checkpoint类型需要先创建模型实例")
    
    else:
        raise ValueError(f"不支持的预训练类型: {pretrained_type}")


if __name__ == "__main__":
    # 测试预训练模型适配器
    print("预训练模型适配器测试")
    
    batch_size = 4
    n_channels = 22
    n_samples = 1000
    d_model = 128
    
    # 创建模拟输入
    x = torch.randn(batch_size, n_channels, n_samples)
    
    # 测试Braindecode适配器
    print("\n测试Braindecode适配器:")
    try:
        braindecode_model = create_pretrained_backbone(
            'braindecode', 'eegnet',
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=d_model
        )
        output = braindecode_model(x)
        print(f"Braindecode输出形状: {output.shape}")
    except Exception as e:
        print(f"Braindecode测试失败: {e}")
    
    # 测试HuggingFace适配器（如果可用）
    if TRANSFORMERS_AVAILABLE:
        print("\n测试HuggingFace适配器:")
        try:
            hf_model = create_pretrained_backbone(
                'huggingface', 'prajjwal1/bert-tiny',
                n_channels=n_channels,
                n_samples=n_samples,
                d_model=d_model,
                freeze_pretrained=True
            )
            output = hf_model(x)
            print(f"HuggingFace输出形状: {output.shape}")
        except Exception as e:
            print(f"HuggingFace测试失败: {e}")
    
    print("预训练模型适配器测试完成!")

