#!/usr/bin/env python3
"""
有监督联合训练脚本
支持多任务学习、多数据集联合训练
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.mi_net import create_mi_net
from eeg.preprocess import EEGPreprocessor


class SimpleEEGAugmentation:
    """简化的EEG数据增强器，用于有监督训练"""
    
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
    
    def apply_augmentation(self, trials, labels=None):
        """应用简单的增强策略"""
        if random.random() > self.aug_prob:
            return trials, labels
            
        augmented_trials = []
        augmented_labels = []
        
        for i in range(len(trials)):
            trial = torch.FloatTensor(trials[i]) if not isinstance(trials[i], torch.Tensor) else trials[i]
            
            # 随机应用一种增强
            aug_type = random.choice(['noise', 'scale', 'shift'])
            
            if aug_type == 'noise':
                noise = torch.randn_like(trial) * 0.01 * trial.std()
                trial = trial + noise
            elif aug_type == 'scale':
                scale = random.uniform(0.9, 1.1)
                trial = trial * scale
            elif aug_type == 'shift':
                n_samples = trial.shape[-1]
                shift = random.randint(-int(0.05*n_samples), int(0.05*n_samples))
                if shift != 0:
                    shifted = torch.zeros_like(trial)
                    if shift > 0:
                        shifted[:, shift:] = trial[:, :-shift]
                    else:
                        shifted[:, :shift] = trial[:, -shift:]
                    trial = shifted
            
            augmented_trials.append(trial.numpy())
            if labels is not None:
                augmented_labels.append(labels[i])
        
        return np.array(augmented_trials), np.array(augmented_labels) if labels is not None else None


class MultiTaskEEGDataset(Dataset):
    """多任务EEG数据集"""
    
    def __init__(self,
                 data_dict: Dict[str, Dict],
                 task_weights: Dict[str, float] = None,
                 augmentation_pipeline=None,
                 augment_prob: float = 0.5):
        """
        初始化多任务数据集
        
        Args:
            data_dict: 数据字典，格式：{task_name: {'trials': array, 'labels': array}}
            task_weights: 任务权重
            augmentation_pipeline: 数据增广管线
            augment_prob: 增广概率
        """
        self.data_dict = data_dict
        self.task_names = list(data_dict.keys())
        self.augmentation_pipeline = augmentation_pipeline
        self.augment_prob = augment_prob
        
        # 设置任务权重
        if task_weights is None:
            task_weights = {task: 1.0 for task in self.task_names}
        self.task_weights = task_weights
        
        # 构建样本索引
        self.samples = []
        for task_name, data in data_dict.items():
            trials = data['trials']
            labels = data['labels']
            
            for i in range(len(trials)):
                self.samples.append({
                    'task_name': task_name,
                    'trial_idx': i,
                    'label': labels[i],
                    'weight': task_weights[task_name]
                })
        
        print(f"多任务数据集初始化:")
        for task_name in self.task_names:
            n_samples = len(data_dict[task_name]['trials'])
            n_classes = len(np.unique(data_dict[task_name]['labels']))
            print(f"  {task_name}: {n_samples}样本, {n_classes}类")
        print(f"  总样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        task_name = sample_info['task_name']
        trial_idx = sample_info['trial_idx']
        
        # 获取数据
        trial = self.data_dict[task_name]['trials'][trial_idx]  # (n_channels, n_samples)
        label = self.data_dict[task_name]['labels'][trial_idx]
        
        # 数据增广
        if self.augmentation_pipeline is not None and random.random() < self.augment_prob:
            trial_batch = np.array([trial])
            aug_trials, _ = self.augmentation_pipeline.apply_augmentation(trial_batch)
            trial = aug_trials[0]
        
        return {
            'trial': torch.FloatTensor(trial),
            'label': torch.LongTensor([label])[0],
            'task_name': task_name,
            'weight': sample_info['weight']
        }


class SupervisedTrainer:
    """有监督训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 task_configs: Dict[str, Dict],
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = 'cosine',
                 max_epochs: int = 200,
                 early_stopping_patience: int = 20,
                 label_smoothing: float = 0.1):
        """
        初始化监督训练器
        
        Args:
            model: MI-Net模型
            task_configs: 任务配置
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_type: 调度器类型
            max_epochs: 最大轮数
            early_stopping_patience: 早停耐心
            label_smoothing: 标签平滑
        """
        self.model = model.to(device)
        self.device = device
        self.task_configs = task_configs
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=max_epochs // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 记录
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       targets: np.ndarray,
                       task_name: str) -> Dict[str, float]:
        """计算评估指标"""
        acc = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='macro')
        kappa = cohen_kappa_score(targets, predictions)
        
        return {
            f'{task_name}_accuracy': acc,
            f'{task_name}_f1': f1,
            f'{task_name}_kappa': kappa
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_configs.keys()}
        task_predictions = {task: [] for task in self.task_configs.keys()}
        task_targets = {task: [] for task in self.task_configs.keys()}
        task_counts = {task: 0 for task in self.task_configs.keys()}
        
        for batch in tqdm(dataloader, desc="Training"):
            # 数据移到设备
            trials = batch['trial'].to(self.device)  # (batch, n_channels, n_samples)
            labels = batch['label'].to(self.device)  # (batch,)
            task_names = batch['task_name']
            weights = batch['weight'].to(self.device)  # (batch,)
            
            self.optimizer.zero_grad()
            
            # 按任务分组处理
            task_batch_losses = []
            
            for task_name in self.task_configs.keys():
                # 找到当前任务的样本
                task_mask = [tn == task_name for tn in task_names]
                if not any(task_mask):
                    continue
                
                task_indices = torch.tensor(task_mask, device=self.device)
                task_trials = trials[task_indices]
                task_labels = labels[task_indices]
                task_weights_batch = weights[task_indices]
                
                if len(task_trials) == 0:
                    continue
                
                # 前向传播
                outputs = self.model(task_trials, task_name=task_name)
                logits = outputs['logits']
                
                # 计算损失
                ce_loss = self.criterion(logits, task_labels)
                
                # 加权损失
                weighted_loss = ce_loss * task_weights_batch.mean()
                
                # 辅助损失（MoE负载均衡）
                aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
                
                # 总损失
                task_loss = weighted_loss + aux_loss
                task_batch_losses.append(task_loss)
                
                # 记录
                task_losses[task_name] += task_loss.item()
                task_counts[task_name] += 1
                total_aux_loss += aux_loss.item()
                
                # 预测
                preds = torch.argmax(logits, dim=1)
                task_predictions[task_name].extend(preds.cpu().numpy())
                task_targets[task_name].extend(task_labels.cpu().numpy())
            
            # 反向传播
            if task_batch_losses:
                batch_loss = sum(task_batch_losses) / len(task_batch_losses)
                batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += batch_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_aux_loss = total_aux_loss / len(dataloader)
        
        # 计算各任务指标
        metrics = {'total_loss': avg_loss, 'aux_loss': avg_aux_loss}
        
        for task_name in self.task_configs.keys():
            if task_counts[task_name] > 0:
                # 任务损失
                metrics[f'{task_name}_loss'] = task_losses[task_name] / task_counts[task_name]
                
                # 任务指标
                if task_predictions[task_name]:
                    task_metrics = self.compute_metrics(
                        np.array(task_predictions[task_name]),
                        np.array(task_targets[task_name]),
                        task_name
                    )
                    metrics.update(task_metrics)
        
        return metrics
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_configs.keys()}
        task_predictions = {task: [] for task in self.task_configs.keys()}
        task_targets = {task: [] for task in self.task_configs.keys()}
        task_counts = {task: 0 for task in self.task_configs.keys()}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                trials = batch['trial'].to(self.device)
                labels = batch['label'].to(self.device)
                task_names = batch['task_name']
                weights = batch['weight'].to(self.device)
                
                # 按任务分组处理
                for task_name in self.task_configs.keys():
                    task_mask = [tn == task_name for tn in task_names]
                    if not any(task_mask):
                        continue
                    
                    task_indices = torch.tensor(task_mask, device=self.device)
                    task_trials = trials[task_indices]
                    task_labels = labels[task_indices]
                    task_weights_batch = weights[task_indices]
                    
                    if len(task_trials) == 0:
                        continue
                    
                    # 前向传播
                    outputs = self.model(task_trials, task_name=task_name)
                    logits = outputs['logits']
                    
                    # 损失
                    ce_loss = self.criterion(logits, task_labels)
                    weighted_loss = ce_loss * task_weights_batch.mean()
                    
                    # 记录
                    task_losses[task_name] += weighted_loss.item()
                    task_counts[task_name] += 1
                    
                    # 预测
                    preds = torch.argmax(logits, dim=1)
                    task_predictions[task_name].extend(preds.cpu().numpy())
                    task_targets[task_name].extend(task_labels.cpu().numpy())
        
        # 计算指标
        total_loss = sum(task_losses.values()) / max(sum(task_counts.values()), 1)
        metrics = {'total_loss': total_loss}
        
        for task_name in self.task_configs.keys():
            if task_counts[task_name] > 0:
                metrics[f'{task_name}_loss'] = task_losses[task_name] / task_counts[task_name]
                
                if task_predictions[task_name]:
                    task_metrics = self.compute_metrics(
                        np.array(task_predictions[task_name]),
                        np.array(task_targets[task_name]),
                        task_name
                    )
                    metrics.update(task_metrics)
        
        return metrics
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader = None,
              save_dir: str = 'checkpoints/supervised') -> None:
        """执行监督训练"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader)
            self.train_metrics.append(train_metrics)
            
            # 验证
            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)
                self.val_metrics.append(val_metrics)
                
                # 计算平均准确率
                val_accs = [v for k, v in val_metrics.items() if k.endswith('_accuracy')]
                avg_val_acc = np.mean(val_accs) if val_accs else 0.0
                
                # 打印指标
                print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_metrics['total_loss']:.4f}, "
                      f"Val Acc: {avg_val_acc:.4f}")
                
                # 早停和模型保存
                if avg_val_acc > self.best_val_acc:
                    self.best_val_acc = avg_val_acc
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_metrics': self.train_metrics,
                        'val_metrics': self.val_metrics,
                        'best_val_acc': self.best_val_acc
                    }, save_dir / 'best_supervised_model.pth')
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"早停：{self.early_stopping_patience}轮内验证性能无提升")
                        break
            else:
                print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics
                }, save_dir / f'supervised_checkpoint_epoch_{epoch + 1}.pth')
        
        # 保存训练曲线
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir: Path):
        """绘制训练曲线"""
        if not self.val_metrics:
            return
        
        # 提取指标名称
        metric_names = set()
        for metrics in self.train_metrics:
            metric_names.update(metrics.keys())
        
        metric_names = sorted([m for m in metric_names if not m.endswith('_loss')])
        
        # 绘制
        n_metrics = len(metric_names)
        if n_metrics == 0:
            return
        
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            
            # 训练指标
            train_values = [m.get(metric_name, 0) for m in self.train_metrics]
            ax.plot(train_values, label='Train', color='blue')
            
            # 验证指标
            val_values = [m.get(metric_name, 0) for m in self.val_metrics]
            ax.plot(val_values, label='Validation', color='red')
            
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'supervised_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_multitask_data(data_configs: Dict[str, str]) -> Dict[str, Dict]:
    """加载多任务数据"""
    data_dict = {}
    
    for task_name, data_path in data_configs.items():
        print(f"加载 {task_name} 数据: {data_path}")
        
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            trials = data['trials']  # (n_trials, n_channels, n_samples)
            labels = data['labels']  # (n_trials,)
            
            data_dict[task_name] = {
                'trials': trials,
                'labels': labels
            }
            
            print(f"  {task_name}: {len(trials)}样本, {len(np.unique(labels))}类")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
    
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="EEG有监督联合训练")
    parser.add_argument('--config', type=str, default='configs/supervised_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_configs', type=str, nargs='+', required=True,
                       help='数据配置，格式：task_name:data_path')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='预训练模型路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints/supervised',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    # 新增覆盖参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='覆盖配置中的batch_size')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='覆盖配置中的max_epochs')
    
    args = parser.parse_args()
    
    # 解析数据配置
    data_configs = {}
    for config_str in args.data_configs:
        task_name, data_path = config_str.split(':')
        data_configs[task_name] = data_path
    
    # 加载配置（指定utf-8以避免Windows编码问题）
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
                'sfreq': 250.0,
                'task_configs': {
                    'task1': {'d_output': 4, 'top_k': 2},
                    'task2': {'d_output': 2, 'top_k': 2}
                }
            },
            'training': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'max_epochs': 200,
                'early_stopping_patience': 20,
                'scheduler_type': 'cosine',
                'label_smoothing': 0.1,
                'task_weights': None
            },
            'augmentation': {
                'enabled': True,
                'augment_prob': 0.3,
                'augmentation_ratio': 0.2
            }
        }
    
    # 覆盖训练超参（如有）
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    print("加载多任务数据...")
    data_dict = load_multitask_data(data_configs)
    
    # 数据分割
    train_data_dict = {}
    val_data_dict = {}
    
    for task_name, data in data_dict.items():
        trials = data['trials']
        labels = data['labels']
        
        n_trials = len(trials)
        train_size = int(0.8 * n_trials)
        indices = np.random.permutation(n_trials)
        
        train_data_dict[task_name] = {
            'trials': trials[indices[:train_size]],
            'labels': labels[indices[:train_size]]
        }
        
        val_data_dict[task_name] = {
            'trials': trials[indices[train_size:]],
            'labels': labels[indices[train_size:]]
        }
    
    # 创建简化增广器
    augmentation_pipeline = None
    if config['augmentation']['enabled']:
        augmentation_pipeline = SimpleEEGAugmentation(aug_prob=config['augmentation']['augment_prob'])
    
    # CPU环境下DataLoader更安全设置
    is_cpu = (args.device == 'cpu')
    num_workers = 0 if is_cpu else 4
    pin_memory = False if is_cpu else True
    
    # 创建数据集
    train_dataset = MultiTaskEEGDataset(
        train_data_dict,
        task_weights=config['training'].get('task_weights'),
        augmentation_pipeline=augmentation_pipeline,
        augment_prob=config['augmentation']['augment_prob']
    )
    
    val_dataset = MultiTaskEEGDataset(
        val_data_dict,
        task_weights=config['training'].get('task_weights')
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 创建模型
    print("创建模型...")
    # 仅保留提供了数据的任务
    if 'task_configs' in config['model']:
        task_cfg = {k: v for k, v in config['model']['task_configs'].items() if k in data_configs}
        if task_cfg:
            config['model']['task_configs'] = task_cfg
    model = create_mi_net(**config['model'])
    
    # 加载预训练权重
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"加载预训练模型: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    def _to_float(x, default):
        try:
            return float(x)
        except Exception:
            return float(default)
    def _to_int(x, default):
        try:
            return int(x)
        except Exception:
            return int(default)

    trainer_kwargs = {
        'learning_rate': _to_float(config['training'].get('learning_rate', 1e-3), 1e-3),
        'weight_decay': _to_float(config['training'].get('weight_decay', 1e-4), 1e-4),
        'scheduler_type': config['training'].get('scheduler_type', 'cosine'),
        'max_epochs': _to_int(config['training'].get('max_epochs', 200), 200),
        'early_stopping_patience': _to_int(config['training'].get('early_stopping_patience', 20), 20),
        'label_smoothing': _to_float(config['training'].get('label_smoothing', 0.1), 0.1),
    }
    trainer = SupervisedTrainer(
        model=model,
        task_configs=config['model']['task_configs'],
        device=args.device,
        **trainer_kwargs
    )
    
    # 开始训练
    print("开始有监督联合训练...")
    trainer.train(train_dataloader, val_dataloader, args.save_dir)
    
    print("训练完成!")


if __name__ == "__main__":
    main()
