#!/usr/bin/env python3
"""
自监督预训练脚本
实现遮罩重建预训练，支持时间块和通道遮罩
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.mi_net import create_mi_net
from eeg.preprocess import EEGPreprocessor


class MaskedEEGDataset(Dataset):
    """遮罩EEG数据集"""
    
    def __init__(self, 
                 data: np.ndarray,
                 mask_ratio: float = 0.5,
                 mask_type: str = 'time_block',
                 min_mask_length: int = 10,
                 max_mask_length: int = 50):
        """
        初始化遮罩数据集
        
        Args:
            data: EEG数据 (n_trials, n_channels, n_samples)
            mask_ratio: 遮罩比例
            mask_type: 遮罩类型 ('time_block', 'channel_time', 'random')
            min_mask_length: 最小遮罩长度
            max_mask_length: 最大遮罩长度
        """
        self.data = torch.FloatTensor(data)
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length
        
        self.n_trials, self.n_channels, self.n_samples = data.shape
        
    def _create_time_block_mask(self, n_samples: int) -> torch.Tensor:
        """创建时间块遮罩"""
        mask = torch.ones(n_samples, dtype=torch.bool)
        
        # 计算需要遮罩的总长度
        total_mask_length = int(n_samples * self.mask_ratio)
        
        masked_length = 0
        while masked_length < total_mask_length:
            # 随机选择遮罩块长度
            remaining_length = total_mask_length - masked_length
            max_block_length = min(self.max_mask_length, remaining_length, n_samples)
            if max_block_length < self.min_mask_length:
                break
            block_length = random.randint(self.min_mask_length, max_block_length)
            
            # 随机选择起始位置
            if block_length >= n_samples:
                start_pos = 0
                block_length = n_samples
            else:
                start_pos = random.randint(0, n_samples - block_length)
            
            # 应用遮罩
            mask[start_pos:start_pos + block_length] = False
            masked_length += block_length
        
        return mask
    
    def _create_channel_time_mask(self, n_channels: int, n_samples: int) -> torch.Tensor:
        """创建通道×时间遮罩"""
        mask = torch.ones(n_channels, n_samples, dtype=torch.bool)
        
        # 计算需要遮罩的总元素数
        total_elements = n_channels * n_samples
        total_mask_elements = int(total_elements * self.mask_ratio)
        
        masked_elements = 0
        while masked_elements < total_mask_elements:
            # 随机选择通道
            channel = random.randint(0, n_channels - 1)
            
            # 随机选择时间块
            block_length = random.randint(self.min_mask_length, self.max_mask_length)
            block_length = min(block_length, total_mask_elements - masked_elements)
            
            start_pos = random.randint(0, n_samples - block_length)
            
            # 应用遮罩
            mask[channel, start_pos:start_pos + block_length] = False
            masked_elements += block_length
        
        return mask
    
    def _create_random_mask(self, n_channels: int, n_samples: int) -> torch.Tensor:
        """创建随机遮罩"""
        total_elements = n_channels * n_samples
        n_mask = int(total_elements * self.mask_ratio)
        
        mask = torch.ones(n_channels, n_samples, dtype=torch.bool)
        mask_flat = mask.view(-1)
        
        # 随机选择遮罩位置
        mask_indices = torch.randperm(total_elements)[:n_mask]
        mask_flat[mask_indices] = False
        
        return mask.view(n_channels, n_samples)
    
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, idx):
        trial = self.data[idx]  # (n_channels, n_samples)
        
        # 创建遮罩
        if self.mask_type == 'time_block':
            mask = self._create_time_block_mask(self.n_samples)
            mask = mask.unsqueeze(0).expand(self.n_channels, -1)  # 广播到所有通道
        elif self.mask_type == 'channel_time':
            mask = self._create_channel_time_mask(self.n_channels, self.n_samples)
        elif self.mask_type == 'random':
            mask = self._create_random_mask(self.n_channels, self.n_samples)
        else:
            raise ValueError(f"不支持的遮罩类型: {self.mask_type}")
        
        # 应用遮罩（遮罩位置设为0）
        masked_trial = trial.clone()
        masked_trial[~mask] = 0
        
        return {
            'input': masked_trial,
            'target': trial,
            'mask': mask
        }


class SSLPretrainer:
    """自监督预训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 warmup_epochs: int = 10,
                 max_epochs: int = 100):
        """
        初始化预训练器
        
        Args:
            model: MI-Net模型
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            max_epochs: 最大轮数
        """
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 预热调度器
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        
    def compute_reconstruction_loss(self, 
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算重建损失
        
        Args:
            predictions: 模型预测 (batch, n_channels, n_samples)
            targets: 目标数据 (batch, n_channels, n_samples)
            masks: 遮罩 (batch, n_channels, n_samples)
            
        Returns:
            损失字典
        """
        # 只在遮罩位置计算损失
        masked_positions = ~masks
        
        if masked_positions.sum() == 0:
            # 如果没有遮罩位置，返回零损失
            return {
                'mse_loss': torch.tensor(0.0, device=predictions.device),
                'l1_loss': torch.tensor(0.0, device=predictions.device),
                'total_loss': torch.tensor(0.0, device=predictions.device)
            }
        
        # MSE损失（主要损失）
        mse_loss = F.mse_loss(
            predictions[masked_positions],
            targets[masked_positions]
        )
        
        # L1损失（正则化）
        l1_loss = F.l1_loss(
            predictions[masked_positions],
            targets[masked_positions]
        )
        
        # 频域损失（可选）
        freq_loss = self.compute_frequency_loss(predictions, targets, masks)
        
        # 总损失
        total_loss = mse_loss + 0.1 * l1_loss + 0.05 * freq_loss
        
        return {
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'freq_loss': freq_loss,
            'total_loss': total_loss
        }
    
    def compute_frequency_loss(self,
                             predictions: torch.Tensor,
                             targets: torch.Tensor,
                             masks: torch.Tensor) -> torch.Tensor:
        """计算频域损失"""
        # 计算FFT
        pred_fft = torch.fft.rfft(predictions, dim=-1)
        target_fft = torch.fft.rfft(targets, dim=-1)
        
        # 频域MSE损失
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        return freq_loss
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {
            'mse_loss': 0.0,
            'l1_loss': 0.0,
            'freq_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch in tqdm(dataloader, desc="Training"):
            # 数据移到设备
            inputs = batch['input'].to(self.device)  # (batch, n_channels, n_samples)
            targets = batch['target'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 通过模型（使用重建任务）
            # 这里我们需要修改模型来支持重建任务
            # 简化版本：直接使用backbone特征进行重建
            model_output = self.model(inputs, task_name='default', return_features=True)
            
            # 获取特征并重建
            features = model_output['features']['pooled']  # (batch, feature_dim)
            
            # 重建头（这里需要添加到模型中，或者临时创建）
            if not hasattr(self, 'reconstruction_head'):
                feature_dim = features.shape[-1]
                n_channels, n_samples = inputs.shape[1], inputs.shape[2]
                self.reconstruction_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 2),
                    nn.ReLU(),
                    nn.Linear(feature_dim * 2, n_channels * n_samples)
                ).to(self.device)
            
            # 重建
            reconstructed = self.reconstruction_head(features)
            reconstructed = reconstructed.view(inputs.shape)  # (batch, n_channels, n_samples)
            
            # 计算损失
            losses = self.compute_reconstruction_loss(reconstructed, targets, masks)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累加损失
            for key, value in losses.items():
                total_losses[key] += value.item()
        
        # 计算平均损失
        avg_losses = {key: value / len(dataloader) for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_losses = {
            'mse_loss': 0.0,
            'l1_loss': 0.0,
            'freq_loss': 0.0,
            'total_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                model_output = self.model(inputs, task_name='default', return_features=True)
                features = model_output['features']['pooled']
                
                # 重建
                reconstructed = self.reconstruction_head(features)
                reconstructed = reconstructed.view(inputs.shape)
                
                # 计算损失
                losses = self.compute_reconstruction_loss(reconstructed, targets, masks)
                
                # 累加损失
                for key, value in losses.items():
                    total_losses[key] += value.item()
        
        # 计算平均损失
        avg_losses = {key: value / len(dataloader) for key, value in total_losses.items()}
        
        return avg_losses
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader = None,
              save_dir: str = 'checkpoints/ssl') -> None:
        """执行预训练"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # 训练
            train_losses = self.train_epoch(train_dataloader)
            self.train_losses.append(train_losses)
            
            # 验证
            if val_dataloader is not None:
                val_losses = self.validate_epoch(val_dataloader)
                self.val_losses.append(val_losses)
                
                # 打印损失
                print(f"Train Loss: {train_losses['total_loss']:.6f}, "
                      f"Val Loss: {val_losses['total_loss']:.6f}")
                
                # 保存最佳模型
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'best_val_loss': best_val_loss
                    }, save_dir / 'best_ssl_model.pth')
            else:
                print(f"Train Loss: {train_losses['total_loss']:.6f}")
            
            # 更新学习率
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, save_dir / f'ssl_checkpoint_epoch_{epoch + 1}.pth')
        
        # 保存训练曲线
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir: Path):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        loss_names = ['total_loss', 'mse_loss', 'l1_loss', 'freq_loss']
        
        for i, loss_name in enumerate(loss_names):
            ax = axes[i]
            
            # 训练损失
            train_values = [losses[loss_name] for losses in self.train_losses]
            ax.plot(train_values, label='Train', color='blue')
            
            # 验证损失
            if self.val_losses:
                val_values = [losses[loss_name] for losses in self.val_losses]
                ax.plot(val_values, label='Validation', color='red')
            
            ax.set_title(f'{loss_name.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'ssl_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_eeg_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载EEG数据"""
    # 这里是示例加载函数，实际需要根据数据格式调整
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        trials = data['trials']  # (n_trials, n_channels, n_samples)
        labels = data['labels']  # (n_trials,)
        return trials, labels
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="EEG自监督预训练")
    parser.add_argument('--config', type=str, default='configs/ssl_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='预处理后的数据路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints/ssl',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 默认配置
        config = {
            'model': {
                'dataset_type': 'bnci2a',
                'n_channels': 22,
                'n_samples': 1000,
                'sfreq': 250.0
            },
            'ssl': {
                'mask_ratio': 0.5,
                'mask_type': 'time_block',
                'min_mask_length': 10,
                'max_mask_length': 50,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'max_epochs': 100,
                'warmup_epochs': 10
            }
        }
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    print("加载数据...")
    trials, labels = load_eeg_data(args.data_path)
    print(f"数据形状: {trials.shape}")
    
    # 数据分割
    n_trials = len(trials)
    train_size = int(0.8 * n_trials)
    indices = np.random.permutation(n_trials)
    
    train_data = trials[indices[:train_size]]
    val_data = trials[indices[train_size:]]
    
    # 创建数据集
    ssl_config = config['ssl']
    train_dataset = MaskedEEGDataset(
        train_data,
        mask_ratio=float(ssl_config['mask_ratio']),
        mask_type=ssl_config['mask_type'],
        min_mask_length=int(ssl_config['min_mask_length']),
        max_mask_length=int(ssl_config['max_mask_length'])
    )
    
    val_dataset = MaskedEEGDataset(
        val_data,
        mask_ratio=float(ssl_config['mask_ratio']),
        mask_type=ssl_config['mask_type'],
        min_mask_length=int(ssl_config['min_mask_length']),
        max_mask_length=int(ssl_config['max_mask_length'])
    )
    
    # CPU环境下DataLoader更安全设置
    is_cpu = (args.device == 'cpu')
    num_workers = 0 if is_cpu else 4
    pin_memory = False if is_cpu else True
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(ssl_config['batch_size']),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(ssl_config['batch_size']),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 创建模型
    print("创建模型...")
    model = create_mi_net(**config['model'])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建预训练器
    pretrainer = SSLPretrainer(
        model=model,
        device=args.device,
        learning_rate=float(ssl_config['learning_rate']),
        weight_decay=float(ssl_config['weight_decay']),
        warmup_epochs=int(ssl_config['warmup_epochs']),
        max_epochs=int(ssl_config['max_epochs'])
    )
    
    # 开始预训练
    print("开始自监督预训练...")
    pretrainer.train(train_dataloader, val_dataloader, args.save_dir)
    
    print("预训练完成!")


if __name__ == "__main__":
    main()

