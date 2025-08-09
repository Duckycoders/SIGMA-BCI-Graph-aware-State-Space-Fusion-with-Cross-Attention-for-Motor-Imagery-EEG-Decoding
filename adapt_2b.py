#!/usr/bin/env python3
"""
跨数据集适配脚本：2a→2b
实现few-shot适配、通道映射、知识蒸馏
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


class ChannelMapper(nn.Module):
    """通道映射器：处理不同数据集的通道差异"""
    
    def __init__(self,
                 source_channels: List[str],
                 target_channels: List[str],
                 mapping_method: str = 'learned',
                 dropout: float = 0.1):
        """
        初始化通道映射器
        
        Args:
            source_channels: 源通道列表
            target_channels: 目标通道列表
            mapping_method: 映射方法 ('learned', 'interpolation', 'selection')
            dropout: Dropout概率
        """
        super().__init__()
        
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.mapping_method = mapping_method
        self.n_source = len(source_channels)
        self.n_target = len(target_channels)
        
        if mapping_method == 'learned':
            # 可学习的线性映射
            self.channel_mapping = nn.Linear(self.n_source, self.n_target, bias=False)
            
        elif mapping_method == 'interpolation':
            # 基于空间位置的插值映射
            self.mapping_matrix = self._create_interpolation_matrix()
            self.register_buffer('fixed_mapping', self.mapping_matrix)
            
        elif mapping_method == 'selection':
            # 选择最相似的通道
            self.channel_indices = self._find_similar_channels()
            
        self.dropout = nn.Dropout(dropout)
        
        print(f"通道映射器初始化:")
        print(f"  源通道: {self.n_source} ({source_channels})")
        print(f"  目标通道: {self.n_target} ({target_channels})")
        print(f"  映射方法: {mapping_method}")
    
    def _create_interpolation_matrix(self) -> torch.Tensor:
        """创建插值映射矩阵"""
        # 简化的插值映射：基于通道名称相似性
        mapping_matrix = torch.zeros(self.n_target, self.n_source)
        
        for i, target_ch in enumerate(self.target_channels):
            # 查找最相似的源通道
            similarities = []
            for j, source_ch in enumerate(self.source_channels):
                if target_ch == source_ch:
                    similarities.append(1.0)
                elif target_ch in source_ch or source_ch in target_ch:
                    similarities.append(0.8)
                else:
                    # 基于名称的简单相似性
                    common_chars = set(target_ch.lower()) & set(source_ch.lower())
                    similarity = len(common_chars) / max(len(target_ch), len(source_ch))
                    similarities.append(similarity)
            
            # 归一化相似性作为权重
            similarities = torch.FloatTensor(similarities)
            if similarities.sum() > 0:
                similarities = similarities / similarities.sum()
            else:
                # 如果没有相似性，使用均匀权重
                similarities = torch.ones(self.n_source) / self.n_source
            
            mapping_matrix[i] = similarities
        
        return mapping_matrix
    
    def _find_similar_channels(self) -> List[int]:
        """找到相似通道的索引"""
        indices = []
        
        for target_ch in self.target_channels:
            best_idx = 0
            best_similarity = 0
            
            for j, source_ch in enumerate(self.source_channels):
                if target_ch == source_ch:
                    best_idx = j
                    break
                elif target_ch in source_ch or source_ch in target_ch:
                    if 0.8 > best_similarity:
                        best_similarity = 0.8
                        best_idx = j
            
            indices.append(best_idx)
        
        return indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据 (batch, n_source_channels, n_samples)
            
        Returns:
            映射后的数据 (batch, n_target_channels, n_samples)
        """
        if self.mapping_method == 'learned':
            # 可学习映射
            # 转置以便应用线性层
            x_transposed = x.transpose(1, 2)  # (batch, n_samples, n_source_channels)
            mapped = self.channel_mapping(x_transposed)  # (batch, n_samples, n_target_channels)
            output = mapped.transpose(1, 2)  # (batch, n_target_channels, n_samples)
            
        elif self.mapping_method == 'interpolation':
            # 插值映射
            output = torch.matmul(self.fixed_mapping.to(x.device), x)
            
        elif self.mapping_method == 'selection':
            # 通道选择
            output = x[:, self.channel_indices, :]
            
        return self.dropout(output)


class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7):
        """
        初始化知识蒸馏损失
        
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            targets: 真实标签
            
        Returns:
            蒸馏损失
        """
        # 温度缩放的软标签
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 蒸馏损失
        distill_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 组合损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


class DomainAdaptationDataset(Dataset):
    """域适配数据集"""
    
    def __init__(self,
                 source_data: Dict,
                 target_data: Dict,
                 source_channels: List[str],
                 target_channels: List[str],
                 adaptation_ratio: float = 0.1):
        """
        初始化域适配数据集
        
        Args:
            source_data: 源域数据 {'trials': array, 'labels': array}
            target_data: 目标域数据
            source_channels: 源域通道
            target_channels: 目标域通道
            adaptation_ratio: 目标域数据使用比例（few-shot）
        """
        self.source_data = source_data
        self.target_data = target_data
        self.source_channels = source_channels
        self.target_channels = target_channels
        
        # 选择少量目标域数据
        n_target_samples = len(target_data['trials'])
        n_adapt_samples = max(1, int(n_target_samples * adaptation_ratio))
        
        # 确保每个类别至少有一个样本
        target_labels = target_data['labels']
        unique_labels = np.unique(target_labels)
        
        adapt_indices = []
        for label in unique_labels:
            label_indices = np.where(target_labels == label)[0]
            n_per_class = max(1, n_adapt_samples // len(unique_labels))
            selected = np.random.choice(label_indices, 
                                      min(n_per_class, len(label_indices)), 
                                      replace=False)
            adapt_indices.extend(selected)
        
        self.adapt_trials = target_data['trials'][adapt_indices]
        self.adapt_labels = target_data['labels'][adapt_indices]
        
        print(f"域适配数据集:")
        print(f"  源域样本: {len(source_data['trials'])}")
        print(f"  目标域总样本: {n_target_samples}")
        print(f"  适配样本: {len(adapt_indices)}")
        print(f"  适配比例: {len(adapt_indices)/n_target_samples:.3f}")
    
    def __len__(self):
        return len(self.adapt_trials)
    
    def __getitem__(self, idx):
        trial = self.adapt_trials[idx]
        label = self.adapt_labels[idx]
        
        return {
            'trial': torch.FloatTensor(trial),
            'label': torch.LongTensor([label])[0]
        }


class DomainAdapter:
    """域适配器"""
    
    def __init__(self,
                 source_model: nn.Module,
                 target_task_config: Dict,
                 source_channels: List[str],
                 target_channels: List[str],
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 adaptation_epochs: int = 50,
                 use_knowledge_distillation: bool = True):
        """
        初始化域适配器
        
        Args:
            source_model: 预训练的源域模型
            target_task_config: 目标任务配置
            source_channels: 源域通道
            target_channels: 目标域通道
            device: 设备
            learning_rate: 学习率
            adaptation_epochs: 适配轮数
            use_knowledge_distillation: 是否使用知识蒸馏
        """
        self.device = device
        self.adaptation_epochs = adaptation_epochs
        self.use_knowledge_distillation = use_knowledge_distillation
        
        # 源模型（冻结）
        self.source_model = source_model.to(device)
        for param in self.source_model.parameters():
            param.requires_grad = False
        self.source_model.eval()
        
        # 通道映射器
        self.channel_mapper = ChannelMapper(
            source_channels=source_channels,
            target_channels=target_channels,
            mapping_method='learned'
        ).to(device)
        
        # 目标模型（复制源模型架构）
        self.target_model = create_mi_net(
            dataset_type='bnci2b',
            n_channels=len(target_channels),
            task_configs={'motor_imagery': target_task_config}
        ).to(device)
        
        # 初始化目标模型权重（从源模型复制兼容部分）
        self._initialize_target_model()
        
        # 冻结主干网络，只训练适配器和分类头
        self._freeze_backbone()
        
        # 优化器（只优化可训练参数）
        trainable_params = list(self.channel_mapper.parameters()) + \
                          [p for p in self.target_model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # 损失函数
        if use_knowledge_distillation:
            self.criterion = KnowledgeDistillationLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"域适配器初始化:")
        print(f"  源域通道: {len(source_channels)}")
        print(f"  目标域通道: {len(target_channels)}")
        print(f"  可训练参数: {sum(p.numel() for p in trainable_params):,}")
        print(f"  使用知识蒸馏: {use_knowledge_distillation}")
    
    def _initialize_target_model(self):
        """初始化目标模型权重"""
        # 复制兼容的权重
        source_state = self.source_model.state_dict()
        target_state = self.target_model.state_dict()
        
        for name, param in target_state.items():
            if name in source_state:
                source_param = source_state[name]
                if param.shape == source_param.shape:
                    param.data.copy_(source_param.data)
                    print(f"  复制权重: {name}")
    
    def _freeze_backbone(self):
        """冻结主干网络"""
        # 冻结除MoE和分类头之外的所有参数
        for name, param in self.target_model.named_parameters():
            if 'moe' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print("冻结主干网络，只训练MoE和分类头")
    
    def adapt_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """适配一个epoch"""
        self.target_model.train()
        self.channel_mapper.train()
        
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for batch in tqdm(dataloader, desc="Adapting"):
            trials = batch['trial'].to(self.device)  # 目标域格式
            labels = batch['label'].to(self.device)
            
            # 通道映射：目标域→源域格式
            mapped_trials = self.channel_mapper(trials)
            
            self.optimizer.zero_grad()
            
            if self.use_knowledge_distillation:
                # 教师模型预测（源域）
                with torch.no_grad():
                    teacher_outputs = self.source_model(mapped_trials, task_name='motor_imagery')
                    teacher_logits = teacher_outputs['logits']
                
                # 学生模型预测（目标域）
                student_outputs = self.target_model(trials, task_name='motor_imagery')
                student_logits = student_outputs['logits']
                
                # 知识蒸馏损失
                loss = self.criterion(student_logits, teacher_logits, labels)
                
            else:
                # 标准交叉熵损失
                outputs = self.target_model(trials, task_name='motor_imagery')
                loss = self.criterion(outputs['logits'], labels)
                student_logits = outputs['logits']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in [self.channel_mapper, self.target_model] if p.requires_grad], 
                max_norm=1.0
            )
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            acc = accuracy_score(labels.cpu().numpy(), 
                               torch.argmax(student_logits, dim=1).cpu().numpy())
            total_acc += acc
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估目标域性能"""
        self.target_model.eval()
        self.channel_mapper.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                trials = batch['trial'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.target_model(trials, task_name='motor_imagery')
                preds = torch.argmax(outputs['logits'], dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'kappa': kappa
        }
    
    def adapt(self, 
              adapt_dataloader: DataLoader,
              test_dataloader: DataLoader,
              save_dir: str = 'checkpoints/adaptation') -> None:
        """执行域适配"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_acc = 0.0
        adaptation_history = []
        
        # 基线性能（适配前）
        print("评估适配前性能...")
        baseline_metrics = self.evaluate(test_dataloader)
        print(f"基线性能: Acc={baseline_metrics['accuracy']:.4f}, "
              f"F1={baseline_metrics['f1']:.4f}, Kappa={baseline_metrics['kappa']:.4f}")
        
        for epoch in range(self.adaptation_epochs):
            print(f"\n适配 Epoch {epoch + 1}/{self.adaptation_epochs}")
            
            # 适配训练
            adapt_metrics = self.adapt_epoch(adapt_dataloader)
            
            # 评估
            test_metrics = self.evaluate(test_dataloader)
            
            # 记录
            epoch_metrics = {
                'epoch': epoch + 1,
                'adapt_loss': adapt_metrics['loss'],
                'adapt_acc': adapt_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_kappa': test_metrics['kappa']
            }
            adaptation_history.append(epoch_metrics)
            
            print(f"适配损失: {adapt_metrics['loss']:.4f}, "
                  f"适配精度: {adapt_metrics['accuracy']:.4f}")
            print(f"测试精度: {test_metrics['accuracy']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}, Kappa: {test_metrics['kappa']:.4f}")
            
            # 保存最佳模型
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'target_model_state_dict': self.target_model.state_dict(),
                    'channel_mapper_state_dict': self.channel_mapper.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'adaptation_history': adaptation_history
                }, save_dir / 'best_adapted_model.pth')
            
            # 更新学习率
            self.scheduler.step(test_metrics['accuracy'])
        
        # 保存适配历史
        self._plot_adaptation_curves(adaptation_history, save_dir)
        
        print(f"\n域适配完成!")
        print(f"最佳测试精度: {best_acc:.4f}")
        improvement = best_acc - baseline_metrics['accuracy']
        print(f"性能提升: {improvement:.4f} ({improvement/baseline_metrics['accuracy']*100:.1f}%)")
    
    def _plot_adaptation_curves(self, history: List[Dict], save_dir: Path):
        """绘制适配曲线"""
        epochs = [h['epoch'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失和适配精度
        ax1.plot(epochs, [h['adapt_loss'] for h in history], 'b-', label='适配损失')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, [h['adapt_acc'] for h in history], 'r-', label='适配精度')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失', color='b')
        ax1_twin.set_ylabel('精度', color='r')
        ax1.set_title('适配过程')
        ax1.grid(True)
        
        # 测试指标
        ax2.plot(epochs, [h['test_acc'] for h in history], 'g-', label='测试精度')
        ax2.plot(epochs, [h['test_f1'] for h in history], 'b-', label='测试F1')
        ax2.plot(epochs, [h['test_kappa'] for h in history], 'm-', label='测试Kappa')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('指标值')
        ax2.set_title('测试性能')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'adaptation_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_adaptation_data(source_path: str, target_path: str) -> Tuple[Dict, Dict]:
    """加载适配数据"""
    # 加载源域数据
    source_data = np.load(source_path)
    source_trials = source_data['trials']
    source_labels = source_data['labels']
    
    # 加载目标域数据
    target_data = np.load(target_path)
    target_trials = target_data['trials']
    target_labels = target_data['labels']
    
    return {
        'trials': source_trials,
        'labels': source_labels
    }, {
        'trials': target_trials,
        'labels': target_labels
    }


def main():
    parser = argparse.ArgumentParser(description="2a→2b域适配")
    parser.add_argument('--source_model_path', type=str, required=True,
                       help='源域预训练模型路径')
    parser.add_argument('--source_data_path', type=str, required=True,
                       help='源域数据路径')
    parser.add_argument('--target_data_path', type=str, required=True,
                       help='目标域数据路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints/adaptation',
                       help='保存目录')
    parser.add_argument('--adaptation_ratio', type=float, default=0.1,
                       help='目标域数据使用比例')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='适配学习率')
    parser.add_argument('--adaptation_epochs', type=int, default=50,
                       help='适配轮数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    print("加载适配数据...")
    source_data, target_data = load_adaptation_data(
        args.source_data_path, args.target_data_path
    )
    
    print(f"源域数据: {source_data['trials'].shape}")
    print(f"目标域数据: {target_data['trials'].shape}")
    
    # 定义通道映射
    source_channels = [f'Ch{i}' for i in range(source_data['trials'].shape[1])]  # 假设BNCI 2a
    target_channels = ['C3', 'Cz', 'C4']  # BNCI 2b双极导联
    
    # 目标域数据分割
    n_target = len(target_data['trials'])
    test_size = int(0.5 * n_target)  # 50%用于测试
    indices = np.random.permutation(n_target)
    
    target_adapt_data = {
        'trials': target_data['trials'][indices[test_size:]],
        'labels': target_data['labels'][indices[test_size:]]
    }
    
    target_test_data = {
        'trials': target_data['trials'][indices[:test_size]],
        'labels': target_data['labels'][indices[:test_size]]
    }
    
    # 创建适配数据集
    adapt_dataset = DomainAdaptationDataset(
        source_data=source_data,
        target_data=target_adapt_data,
        source_channels=source_channels,
        target_channels=target_channels,
        adaptation_ratio=args.adaptation_ratio
    )
    
    test_dataset = DomainAdaptationDataset(
        source_data=source_data,
        target_data=target_test_data,
        source_channels=source_channels,
        target_channels=target_channels,
        adaptation_ratio=1.0  # 使用所有测试数据
    )
    
    # 创建数据加载器
    adapt_dataloader = DataLoader(
        adapt_dataset, batch_size=8, shuffle=True, num_workers=2
    )
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    
    # 加载源域模型
    print("加载源域模型...")
    checkpoint = torch.load(args.source_model_path, map_location='cpu')
    
    source_model = create_mi_net('bnci2a')
    source_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建域适配器
    adapter = DomainAdapter(
        source_model=source_model,
        target_task_config={'d_output': 2, 'top_k': 2},  # BNCI 2b是2类
        source_channels=source_channels,
        target_channels=target_channels,
        device=args.device,
        learning_rate=args.learning_rate,
        adaptation_epochs=args.adaptation_epochs,
        use_knowledge_distillation=True
    )
    
    # 执行域适配
    print("开始域适配...")
    adapter.adapt(adapt_dataloader, test_dataloader, args.save_dir)
    
    print("域适配完成!")


if __name__ == "__main__":
    main()

