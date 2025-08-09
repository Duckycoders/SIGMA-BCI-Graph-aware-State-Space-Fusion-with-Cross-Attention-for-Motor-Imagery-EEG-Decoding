#!/usr/bin/env python3
"""
消融实验脚本
测试各模块对模型性能的贡献
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

from models.mi_net import create_mi_net
from eval import ModelEvaluator, EvalDataset
from torch.utils.data import DataLoader


def run_ablation_experiment(data_path: str, 
                           save_dir: str = 'results/ablation',
                           device: str = 'cuda',
                           n_trials_limit: int = None) -> Dict:
    """运行消融实验"""
    
    # 加载数据
    data = np.load(data_path)
    trials = data['trials']
    labels = data['labels']
    
    # 限制试次数（用于快速测试）
    if n_trials_limit and len(trials) > n_trials_limit:
        indices = np.random.choice(len(trials), n_trials_limit, replace=False)
        trials = trials[indices]
        labels = labels[indices]
    
    n_channels = trials.shape[1]
    n_samples = trials.shape[2]
    n_classes = len(np.unique(labels))
    
    print(f"消融实验数据: {trials.shape}, {n_classes}类")
    
    # 创建数据集
    dataset = EvalDataset(trials, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 定义消融配置
    ablation_configs = {
        'full_model': {
            'description': '完整模型',
            'module_switches': {}
        },
        'no_filterbank': {
            'description': '无FilterBank',
            'module_switches': {'filterbank': False}
        },
        'no_graph_conv': {
            'description': '无图卷积',
            'module_switches': {'graph_conv': False}
        },
        's4_only': {
            'description': '仅S4分支',
            'module_switches': {'mamba_branch': False}
        },
        'mamba_only': {
            'description': '仅Mamba分支',
            'module_switches': {'s4_branch': False}
        },
        'no_fusion': {
            'description': '无跨注意力融合',
            'module_switches': {'fusion': False}
        },
        'no_moe': {
            'description': '无MoE',
            'module_switches': {'moe': False}
        },
        'minimal': {
            'description': '最小模型 (无FB+无Graph+无MoE)',
            'module_switches': {
                'filterbank': False,
                'graph_conv': False,
                'moe': False
            }
        }
    }
    
    results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\n测试配置: {config['description']}")
        
        try:
            # 创建模型
            model = create_mi_net(
                n_channels=n_channels,
                n_samples=n_samples,
                task_configs={'motor_imagery': {'d_output': n_classes, 'top_k': 2}},
                module_switches=config['module_switches']
            )
            
            # 随机初始化权重（消融实验用）
            # 在实际应用中，应该加载预训练权重
            
            # 评估
            evaluator = ModelEvaluator(model, device)
            metrics = evaluator.evaluate_single_task(dataloader)
            
            # 记录结果
            results[config_name] = {
                'description': config['description'],
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'cohen_kappa': metrics['cohen_kappa'],
                'n_parameters': sum(p.numel() for p in model.parameters())
            }
            
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  参数量: {results[config_name]['n_parameters']:,}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results[config_name] = {
                'description': config['description'],
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'cohen_kappa': 0.0,
                'n_parameters': 0,
                'error': str(e)
            }
    
    return results


def plot_ablation_results(results: Dict, save_dir: Path):
    """绘制消融实验结果"""
    
    # 准备数据
    configs = []
    accuracies = []
    f1_scores = []
    kappa_scores = []
    param_counts = []
    
    for config_name, result in results.items():
        if 'error' not in result:
            configs.append(result['description'])
            accuracies.append(result['accuracy'])
            f1_scores.append(result['f1_macro'])
            kappa_scores.append(result['cohen_kappa'])
            param_counts.append(result['n_parameters'])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Configuration': configs,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'Cohen_Kappa': kappa_scores,
        'Parameters': param_counts
    })
    
    # 按准确率排序
    df = df.sort_values('Accuracy', ascending=True)
    
    # 绘制性能对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准确率对比
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df['Configuration'], df['Accuracy'], alpha=0.7)
    ax1.set_xlabel('准确率')
    ax1.set_title('消融实验 - 准确率对比')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')
    
    # F1分数对比
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df['Configuration'], df['F1-Score'], alpha=0.7, color='orange')
    ax2.set_xlabel('F1分数')
    ax2.set_title('消融实验 - F1分数对比')
    ax2.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')
    
    # Cohen's κ对比
    ax3 = axes[1, 0]
    bars3 = ax3.barh(df['Configuration'], df['Cohen_Kappa'], alpha=0.7, color='green')
    ax3.set_xlabel("Cohen's κ")
    ax3.set_title("消融实验 - Cohen's κ对比")
    ax3.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')
    
    # 参数量对比
    ax4 = axes[1, 1]
    bars4 = ax4.barh(df['Configuration'], df['Parameters']/1e6, alpha=0.7, color='red')
    ax4.set_xlabel('参数量 (M)')
    ax4.set_title('消融实验 - 参数量对比')
    ax4.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}M', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存表格
    df.to_csv(save_dir / 'ablation_results.csv', index=False)
    
    return df


def analyze_module_contribution(results: Dict) -> Dict:
    """分析各模块的贡献度"""
    
    full_model_acc = results.get('full_model', {}).get('accuracy', 0)
    
    contributions = {}
    
    # 计算各模块移除后的性能下降
    module_impacts = {
        'FilterBank': results.get('no_filterbank', {}).get('accuracy', 0),
        'GraphConv': results.get('no_graph_conv', {}).get('accuracy', 0),
        'Mamba分支': results.get('s4_only', {}).get('accuracy', 0),
        'S4分支': results.get('mamba_only', {}).get('accuracy', 0),
        'Cross-Attention': results.get('no_fusion', {}).get('accuracy', 0),
        'MoE': results.get('no_moe', {}).get('accuracy', 0)
    }
    
    for module_name, reduced_acc in module_impacts.items():
        if reduced_acc > 0:
            contribution = full_model_acc - reduced_acc
            contributions[module_name] = {
                'absolute_contribution': contribution,
                'relative_contribution': contribution / full_model_acc if full_model_acc > 0 else 0,
                'performance_drop': -contribution  # 负值表示性能下降
            }
    
    return contributions


def plot_module_contributions(contributions: Dict, save_dir: Path):
    """绘制模块贡献度分析"""
    
    modules = list(contributions.keys())
    abs_contributions = [contributions[m]['absolute_contribution'] for m in modules]
    rel_contributions = [contributions[m]['relative_contribution'] * 100 for m in modules]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绝对贡献度
    bars1 = ax1.bar(modules, abs_contributions, alpha=0.7)
    ax1.set_ylabel('准确率贡献 (绝对值)')
    ax1.set_title('模块贡献度分析 - 绝对贡献')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 相对贡献度
    bars2 = ax2.bar(modules, rel_contributions, alpha=0.7, color='orange')
    ax2.set_ylabel('准确率贡献 (%)')
    ax2.set_title('模块贡献度分析 - 相对贡献')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'module_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据路径')
    parser.add_argument('--save_dir', type=str, default='results/ablation',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--n_trials_limit', type=int, default=None,
                       help='限制试次数（用于快速测试）')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始消融实验...")
    
    # 运行消融实验
    results = run_ablation_experiment(
        args.data_path,
        args.save_dir,
        args.device,
        args.n_trials_limit
    )
    
    # 保存原始结果
    with open(save_dir / 'ablation_raw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 分析和可视化
    print("\n生成分析报告...")
    
    # 绘制结果对比
    df = plot_ablation_results(results, save_dir)
    print(f"结果表格保存到: {save_dir / 'ablation_results.csv'}")
    
    # 分析模块贡献
    contributions = analyze_module_contribution(results)
    plot_module_contributions(contributions, save_dir)
    
    # 生成汇总报告
    print("\n=== 消融实验汇总 ===")
    
    if 'full_model' in results and 'error' not in results['full_model']:
        full_acc = results['full_model']['accuracy']
        print(f"完整模型准确率: {full_acc:.4f}")
        
        # 找出最重要的模块
        if contributions:
            most_important = max(contributions.items(), 
                               key=lambda x: x[1]['absolute_contribution'])
            print(f"最重要模块: {most_important[0]} "
                  f"(贡献: {most_important[1]['absolute_contribution']:.4f})")
        
        # 效率分析
        minimal_result = results.get('minimal', {})
        if 'error' not in minimal_result:
            minimal_acc = minimal_result['accuracy']
            minimal_params = minimal_result['n_parameters']
            full_params = results['full_model']['n_parameters']
            
            print(f"最小模型准确率: {minimal_acc:.4f}")
            print(f"性能差异: {full_acc - minimal_acc:.4f}")
            print(f"参数压缩比: {full_params / minimal_params:.1f}x")
    
    print(f"\n消融实验完成，结果保存到: {save_dir}")


if __name__ == "__main__":
    main()

