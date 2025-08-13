#!/usr/bin/env python3
"""
模型评估脚本
支持LOSO、跨数据集、跨日评估
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
from scipy import stats
import json

from models.mi_net import create_mi_net, MINet


class EvalDataset(Dataset):
    """评估数据集"""
    
    def __init__(self, trials: np.ndarray, labels: np.ndarray):
        self.trials = torch.FloatTensor(trials)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return {
            'trial': self.trials[idx],
            'label': self.labels[idx]
        }


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_single_task(self, 
                            dataloader: DataLoader, 
                            task_name: str = 'motor_imagery') -> Dict[str, float]:
        """评估单个任务"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                trials = batch['trial'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(trials, task_name=task_name)
                logits = outputs['logits']
                probs = outputs['predictions']
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        # 计算置信区间
        n_samples = len(all_labels)
        acc_ci = self._compute_confidence_interval(acc, n_samples)
        f1_ci = self._compute_confidence_interval(f1, n_samples)
        kappa_ci = self._compute_confidence_interval(kappa, n_samples)
        
        return {
            'accuracy': acc,
            'f1_macro': f1,
            'cohen_kappa': kappa,
            'accuracy_ci': acc_ci,
            'f1_ci': f1_ci,
            'kappa_ci': kappa_ci,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def _compute_confidence_interval(self, metric: float, n_samples: int, 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # 使用正态分布近似
        std_error = np.sqrt(metric * (1 - metric) / n_samples)
        margin = z_score * std_error
        
        return (max(0, metric - margin), min(1, metric + margin))
    
    def loso_evaluation(self, 
                       subjects_data: Dict[int, Dict],
                       task_name: str = 'motor_imagery') -> Dict:
        """留一受试者交叉验证 (LOSO)"""
        print("执行LOSO评估...")
        
        subject_results = {}
        all_accuracies = []
        all_f1s = []
        all_kappas = []
        
        subjects = list(subjects_data.keys())
        
        for test_subject in subjects:
            print(f"\n测试受试者 {test_subject}")
            
            # 准备测试数据
            test_data = subjects_data[test_subject]
            test_dataset = EvalDataset(test_data['trials'], test_data['labels'])
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # 评估
            results = self.evaluate_single_task(test_dataloader, task_name)
            
            subject_results[test_subject] = results
            all_accuracies.append(results['accuracy'])
            all_f1s.append(results['f1_macro'])
            all_kappas.append(results['cohen_kappa'])
            
            print(f"受试者 {test_subject}: Acc={results['accuracy']:.4f}, "
                  f"F1={results['f1_macro']:.4f}, Kappa={results['cohen_kappa']:.4f}")
        
        # 统计汇总
        loso_summary = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'mean_f1': np.mean(all_f1s),
            'std_f1': np.std(all_f1s),
            'mean_kappa': np.mean(all_kappas),
            'std_kappa': np.std(all_kappas),
            'subject_results': subject_results,
            'all_accuracies': all_accuracies
        }
        
        print(f"\nLOSO汇总:")
        print(f"平均准确率: {loso_summary['mean_accuracy']:.4f} ± {loso_summary['std_accuracy']:.4f}")
        print(f"平均F1: {loso_summary['mean_f1']:.4f} ± {loso_summary['std_f1']:.4f}")
        print(f"平均Kappa: {loso_summary['mean_kappa']:.4f} ± {loso_summary['std_kappa']:.4f}")
        
        return loso_summary
    
    def cross_dataset_evaluation(self, 
                                source_data: Dict,
                                target_data: Dict,
                                source_task: str = 'motor_imagery',
                                target_task: str = 'motor_imagery') -> Dict:
        """跨数据集评估"""
        print("执行跨数据集评估...")
        
        # 源域评估
        source_dataset = EvalDataset(source_data['trials'], source_data['labels'])
        source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=False)
        source_results = self.evaluate_single_task(source_dataloader, source_task)
        
        # 目标域评估
        target_dataset = EvalDataset(target_data['trials'], target_data['labels'])
        target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=False)
        target_results = self.evaluate_single_task(target_dataloader, target_task)
        
        # 计算性能差异
        acc_drop = source_results['accuracy'] - target_results['accuracy']
        f1_drop = source_results['f1_macro'] - target_results['f1_macro']
        kappa_drop = source_results['cohen_kappa'] - target_results['kappa']
        
        cross_dataset_results = {
            'source_results': source_results,
            'target_results': target_results,
            'accuracy_drop': acc_drop,
            'f1_drop': f1_drop,
            'kappa_drop': kappa_drop
        }
        
        print(f"源域性能: Acc={source_results['accuracy']:.4f}")
        print(f"目标域性能: Acc={target_results['accuracy']:.4f}")
        print(f"性能下降: {acc_drop:.4f}")
        
        return cross_dataset_results
    
    def cross_session_evaluation(self, 
                                sessions_data: Dict[int, Dict],
                                task_name: str = 'motor_imagery') -> Dict:
        """跨会话评估"""
        print("执行跨会话评估...")
        
        session_results = {}
        sessions = sorted(sessions_data.keys())
        
        for i, train_session in enumerate(sessions[:-1]):
            for test_session in sessions[i+1:]:
                print(f"\n训练会话 {train_session} -> 测试会话 {test_session}")
                
                # 测试数据
                test_data = sessions_data[test_session]
                test_dataset = EvalDataset(test_data['trials'], test_data['labels'])
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # 评估
                results = self.evaluate_single_task(test_dataloader, task_name)
                
                session_pair = f"S{train_session}->S{test_session}"
                session_results[session_pair] = results
                
                print(f"{session_pair}: Acc={results['accuracy']:.4f}")
        
        return session_results


def plot_confusion_matrix(labels: np.ndarray, 
                         predictions: np.ndarray,
                         class_names: List[str],
                         save_path: str):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loso_results(loso_results: Dict, save_path: str):
    """绘制LOSO结果"""
    subjects = list(loso_results['subject_results'].keys())
    accuracies = [loso_results['subject_results'][s]['accuracy'] for s in subjects]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(subjects)), accuracies, alpha=0.7)
    plt.axhline(y=loso_results['mean_accuracy'], color='red', linestyle='--', 
               label=f"平均: {loso_results['mean_accuracy']:.4f}")
    
    plt.xlabel('受试者')
    plt.ylabel('准确率')
    plt.title('LOSO评估结果')
    plt.xticks(range(len(subjects)), [f'S{s}' for s in subjects])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def statistical_significance_test(results1: List[float], 
                                results2: List[float]) -> Dict:
    """统计显著性检验"""
    # Wilcoxon签秩检验
    statistic, p_value = stats.wilcoxon(results1, results2)
    
    # 效应大小 (Cohen's d)
    pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
    cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
    
    return {
        'wilcoxon_statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }


def load_evaluation_data(data_path: str) -> Dict:
    """加载评估数据"""
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        return {
            'trials': data['trials'],
            'labels': data['labels'],
            'subjects': data.get('subjects', None),
            'sessions': data.get('sessions', None)
        }
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='评估数据路径')
    parser.add_argument('--eval_type', type=str, 
                       choices=['single', 'loso', 'cross_dataset', 'cross_session'],
                       default='single', help='评估类型')
    parser.add_argument('--target_data_path', type=str, default=None,
                       help='目标数据集路径（跨数据集评估）')
    parser.add_argument('--save_dir', type=str, default='results/evaluation',
                       help='结果保存目录')
    parser.add_argument('--task_name', type=str, default='motor_imagery',
                       help='任务名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    # 根据数据推断模型配置
    data = load_evaluation_data(args.data_path)
    n_channels = data['trials'].shape[1]
    n_samples = data['trials'].shape[2]
    n_classes = len(np.unique(data['labels']))
    
    # 使用与训练时相同的配置
    model = create_mi_net(
        dataset_type='bnci2a',  # 使用标准配置
        n_channels=n_channels,
        n_samples=n_samples,
        use_graph_conv=False,   # 禁用图卷积
        use_mamba_branch=False, # 禁用Mamba
        task_configs={args.task_name: {'d_output': n_classes, 'top_k': 2}}
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, args.device)
    
    # 执行评估
    if args.eval_type == 'single':
        # 单次评估
        dataset = EvalDataset(data['trials'], data['labels'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        results = evaluator.evaluate_single_task(dataloader, args.task_name)
        
        print(f"评估结果:")
        print(f"准确率: {results['accuracy']:.4f} ± {results['accuracy_ci']}")
        print(f"F1分数: {results['f1_macro']:.4f} ± {results['f1_ci']}")
        print(f"Cohen's κ: {results['cohen_kappa']:.4f} ± {results['kappa_ci']}")
        
        # 绘制混淆矩阵
        class_names = [f"Class {i}" for i in range(n_classes)]
        plot_confusion_matrix(
            results['labels'], results['predictions'], 
            class_names, save_dir / 'confusion_matrix.png'
        )
        
        # 保存结果
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'cohen_kappa': results['cohen_kappa'],
                'accuracy_ci': results['accuracy_ci'],
                'f1_ci': results['f1_ci'],
                'kappa_ci': results['kappa_ci']
            }, f, indent=2)
    
    elif args.eval_type == 'loso':
        # LOSO评估
        if data['subjects'] is None:
            raise ValueError("LOSO评估需要受试者信息")
        
        # 按受试者组织数据
        subjects_data = {}
        unique_subjects = np.unique(data['subjects'])
        
        for subject in unique_subjects:
            subject_mask = data['subjects'] == subject
            subjects_data[subject] = {
                'trials': data['trials'][subject_mask],
                'labels': data['labels'][subject_mask]
            }
        
        loso_results = evaluator.loso_evaluation(subjects_data, args.task_name)
        
        # 绘制LOSO结果
        plot_loso_results(loso_results, save_dir / 'loso_results.png')
        
        # 保存结果
        with open(save_dir / 'loso_results.json', 'w') as f:
            json.dump({
                'mean_accuracy': loso_results['mean_accuracy'],
                'std_accuracy': loso_results['std_accuracy'],
                'mean_f1': loso_results['mean_f1'],
                'std_f1': loso_results['std_f1'],
                'mean_kappa': loso_results['mean_kappa'],
                'std_kappa': loso_results['std_kappa'],
                'all_accuracies': loso_results['all_accuracies']
            }, f, indent=2)
    
    elif args.eval_type == 'cross_dataset':
        # 跨数据集评估
        if args.target_data_path is None:
            raise ValueError("跨数据集评估需要目标数据集路径")
        
        target_data = load_evaluation_data(args.target_data_path)
        
        cross_results = evaluator.cross_dataset_evaluation(
            source_data=data,
            target_data=target_data,
            source_task=args.task_name,
            target_task=args.task_name
        )
        
        # 保存结果
        with open(save_dir / 'cross_dataset_results.json', 'w') as f:
            json.dump({
                'source_accuracy': cross_results['source_results']['accuracy'],
                'target_accuracy': cross_results['target_results']['accuracy'],
                'accuracy_drop': cross_results['accuracy_drop'],
                'f1_drop': cross_results['f1_drop'],
                'kappa_drop': cross_results['kappa_drop']
            }, f, indent=2)
    
    print(f"评估完成，结果保存到: {save_dir}")


if __name__ == "__main__":
    main()

