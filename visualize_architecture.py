#!/usr/bin/env python3
"""
MI-Net架构图绘制程序
基于代码分析生成学术专业级架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, Arrow
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# 设置学术论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class MINetArchitectureVisualizer:
    """MI-Net架构可视化器"""
    
    def __init__(self, figsize=(16, 12)):
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.ax.set_xlim(0, 22)
        self.ax.set_ylim(0, 16)
        self.ax.axis('off')
        
        # 定义颜色方案（学术风格）
        self.colors = {
            'input': '#E8F4FD',           # 浅蓝色 - 输入
            'preprocessing': '#FFF2CC',    # 浅黄色 - 预处理
            'feature_extraction': '#E1F5FE', # 浅青色 - 特征提取
            'sequence_modeling': '#F3E5F5', # 浅紫色 - 序列建模
            'fusion': '#FFECB3',          # 浅橙色 - 融合
            'classification': '#C8E6C9',   # 浅绿色 - 分类
            'attention': '#FFCDD2',       # 浅红色 - 注意力
            'moe': '#D1C4E9',             # 浅紫色 - MoE
            'border': '#424242',          # 深灰色 - 边框
            'text': '#212121',            # 深灰色 - 文字
            'arrow': '#1976D2'            # 蓝色 - 箭头
        }
        
        # 模块位置配置
        self.positions = {}
        self._setup_positions()
        
    def _setup_positions(self):
        """设置各模块的位置"""
        self.positions = {
            'input': (3, 14),
            'filterbank': (3, 12),
            'graph_conv': (3, 10),
            's4_branch': (7, 8),
            'mamba_branch': (11, 8),
            'cross_attention': (9, 6),
            'multiscale_fusion': (9, 4.5),
            'global_pool': (9, 3),
            'moe_adapter': (13, 3),
            'classifier': (16, 3),
            'output': (18, 3)
        }
    
    def draw_module(self, name: str, position: Tuple[float, float], 
                   size: Tuple[float, float], color: str, 
                   title: str, details: List[str] = None, 
                   shape: str = 'rectangle', side_text: List[str] = None):
        """绘制模块"""
        x, y = position
        w, h = size
        
        if shape == 'rectangle':
            # 绘制主体矩形
            rect = FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor=self.colors['border'],
                linewidth=1.2,
                alpha=0.9
            )
            self.ax.add_patch(rect)
        elif shape == 'ellipse':
            ellipse = Ellipse(
                (x, y), w, h,
                facecolor=color,
                edgecolor=self.colors['border'],
                linewidth=1.2,
                alpha=0.9
            )
            self.ax.add_patch(ellipse)
        
        # 添加标题（只在模块内部显示标题）
        self.ax.text(x, y, title, 
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color=self.colors['text'])
        
        # 添加侧边详细信息
        if side_text:
            for i, detail in enumerate(side_text):
                self.ax.text(x + w/2 + 0.3, y + 0.3 - i*0.3, detail,
                           ha='left', va='center',
                           fontsize=9,
                           color=self.colors['text'],
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', 
                                   edgecolor='lightgray',
                                   alpha=0.8))
        
        # 如果没有侧边文字，则在模块内部添加简短信息
        if details and not side_text:
            for i, detail in enumerate(details):
                self.ax.text(x, y - 0.3 - i*0.25, detail,
                           ha='center', va='center',
                           fontsize=8,
                           color=self.colors['text'])
    
    def draw_arrow(self, start: Tuple[float, float], 
                  end: Tuple[float, float],
                  label: str = None, 
                  style: str = 'single',
                  curvature: float = 0):
        """绘制连接箭头"""
        x1, y1 = start
        x2, y2 = end
        
        if curvature == 0:
            # 直线箭头
            arrow = mpatches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->' if style == 'single' else '<->',
                mutation_scale=15,
                color=self.colors['arrow'],
                linewidth=1.5,
                alpha=0.8
            )
        else:
            # 曲线箭头
            arrow = mpatches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->' if style == 'single' else '<->',
                mutation_scale=15,
                color=self.colors['arrow'],
                linewidth=1.5,
                alpha=0.8,
                connectionstyle=f"arc3,rad={curvature}"
            )
        
        self.ax.add_patch(arrow)
        
        # 添加标签
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if curvature != 0:
                mid_y += curvature * 0.5
            self.ax.text(mid_x, mid_y + 0.15, label,
                        ha='center', va='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='white', 
                                edgecolor='none',
                                alpha=0.8))
    
    def draw_data_flow(self):
        """绘制数据流向"""
        # 主要数据流
        flows = [
            ('input', 'filterbank', '(B, C, T)'),
            ('filterbank', 'graph_conv', '(B, F, C, T)'),
            ('graph_conv', 's4_branch', '(B, F, T, D)'),
            ('graph_conv', 'mamba_branch', '(B, F, T, D)'),
            ('s4_branch', 'cross_attention', '(B, F, T, D_s4)'),
            ('mamba_branch', 'cross_attention', '(B, F, T, D_mamba)'),
            ('cross_attention', 'multiscale_fusion', '(B, F, T, D)'),
            ('multiscale_fusion', 'global_pool', '(B, T, D)'),
            ('global_pool', 'moe_adapter', '(B, D)'),
            ('moe_adapter', 'classifier', '(B, D)'),
            ('classifier', 'output', '(B, Classes)')
        ]
        
        for start_name, end_name, shape_label in flows:
            if start_name in self.positions and end_name in self.positions:
                start_pos = self.positions[start_name]
                end_pos = self.positions[end_name]
                
                # 调整连接点
                if start_name == 'graph_conv' and end_name in ['s4_branch', 'mamba_branch']:
                    # 分叉连接
                    curvature = 0.2 if end_name == 's4_branch' else -0.2
                    self.draw_arrow(start_pos, end_pos, shape_label, curvature=curvature)
                elif start_name in ['s4_branch', 'mamba_branch'] and end_name == 'cross_attention':
                    # 汇聚连接
                    curvature = 0.2 if start_name == 's4_branch' else -0.2
                    self.draw_arrow(start_pos, end_pos, shape_label, curvature=curvature)
                else:
                    self.draw_arrow(start_pos, end_pos, shape_label)
    
    def draw_complete_architecture(self):
        """绘制完整架构"""
        # 1. 输入模块
        self.draw_module(
            'input', self.positions['input'], (2.5, 1.0),
            self.colors['input'], 'EEG Input',
            side_text=['Raw EEG Signal', 'Shape: (B, 22, 1000)']
        )
        
        # 2. FilterBank模块
        self.draw_module(
            'filterbank', self.positions['filterbank'], (2.5, 1.0),
            self.colors['preprocessing'], 'FilterBank',
            side_text=['4 Frequency Bands', 'μ(4-8Hz), α(8-14Hz)', 'β(14-30Hz), γ(30-45Hz)']
        )
        
        # 3. 图卷积模块
        self.draw_module(
            'graph_conv', self.positions['graph_conv'], (2.5, 1.0),
            self.colors['feature_extraction'], 'Graph Conv',
            side_text=['Spatial Modeling', 'Electrode Connectivity', 'GCN/GAT Layers']
        )
        
        # 4. S4分支
        self.draw_module(
            's4_branch', self.positions['s4_branch'], (2.5, 1.0),
            self.colors['sequence_modeling'], 'S4 Branch',
            side_text=['State Space Model', 'Long Sequence', 'HiPPO Matrix', '2 Layers']
        )
        
        # 5. Mamba分支
        self.draw_module(
            'mamba_branch', self.positions['mamba_branch'], (2.5, 1.0),
            self.colors['sequence_modeling'], 'Mamba Branch',
            side_text=['Selective SSM', 'Dynamic Selection', 'Parallel Scanning', '2 Layers']
        )
        
        # 6. 跨注意力融合
        self.draw_module(
            'cross_attention', self.positions['cross_attention'], (3.0, 1.0),
            self.colors['attention'], 'Cross-Attention',
            side_text=['S4 ↔ Mamba Interaction', 'Bidirectional Attention', 'Multi-Head (8 heads)']
        )
        
        # 7. 多尺度融合
        self.draw_module(
            'multiscale_fusion', self.positions['multiscale_fusion'], (2.5, 0.8),
            self.colors['fusion'], 'Multi-Scale Fusion',
            side_text=['Frequency Band Fusion', 'Attention Weighting']
        )
        
        # 8. 全局池化
        self.draw_module(
            'global_pool', self.positions['global_pool'], (2.5, 0.8),
            self.colors['preprocessing'], 'Global Pool',
            side_text=['Adaptive Avg Pool', 'Temporal Aggregation']
        )
        
        # 9. MoE适配器
        self.draw_module(
            'moe_adapter', self.positions['moe_adapter'], (2.5, 1.0),
            self.colors['moe'], 'MoE Adapter',
            side_text=['4 Experts', 'Top-2 Selection', 'Load Balancing']
        )
        
        # 10. 分类器
        self.draw_module(
            'classifier', self.positions['classifier'], (2.5, 0.8),
            self.colors['classification'], 'Classifier',
            side_text=['Task-Specific Head', 'Multi-Task Support']
        )
        
        # 11. 输出
        self.draw_module(
            'output', self.positions['output'], (2.0, 0.8),
            self.colors['input'], 'Output',
            side_text=['Class Probabilities', 'Shape: (B, Classes)']
        )
        
        # 绘制数据流
        self.draw_data_flow()
        
        # 添加侧边注释
        self._add_side_annotations()
        
        # 添加标题和说明
        self._add_title_and_legend()
    
    def _add_side_annotations(self):
        """添加侧边技术说明"""
        # 左下角技术栈和训练策略说明
        tech_stack = [
            "Key Technologies:",
            "• FilterBank: Multi-band filtering",
            "• Graph Conv: Spatial modeling",
            "• S4/Mamba: Sequence modeling",
            "• Cross-Attention: Branch fusion",
            "• MoE: Multi-task learning",
            "",
            "Training Strategies:",
            "• Self-Supervised Pre-training",
            "• Multi-task Joint Training", 
            "• Domain Adaptation"
        ]
        
        for i, text in enumerate(tech_stack):
            weight = 'bold' if text.endswith(':') else 'normal'
            self.ax.text(0.5, 5.5 - i*0.35, text, 
                        fontsize=9, fontweight=weight,
                        color=self.colors['text'])
        
        # 右侧性能指标
        performance = [
            "Performance:",
            "• BNCI 2a: 70-85% (LOSO)",
            "• BNCI 2b: 75-90% (LOSO)",
            "• EEG-MMI: 65-80% (LOSO)",
            "",
            "Evaluation:",
            "• Leave-One-Subject-Out",
            "• Cross-Dataset Transfer",
            "• Statistical Significance",
            "",
            "Datasets:",
            "• BNCI2014-001 (22ch)",
            "• BNCI2014-004 (3ch)", 
            "• PhysionetMI (64ch)"
        ]
        
        for i, text in enumerate(performance):
            weight = 'bold' if text.endswith(':') else 'normal'
            self.ax.text(18.5, 12 - i*0.3, text,
                        fontsize=8, fontweight=weight,
                        color=self.colors['text'])
    
    def _add_title_and_legend(self):
        """添加标题和图例"""
        # 主标题
        self.ax.text(11, 15.5, 'MI-Net: Motor Imagery EEG Decoding Architecture',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color=self.colors['text'])
        
        # 副标题
        self.ax.text(11, 15, 'FilterBank + Graph Conv + S4/Mamba + Cross-Attention + MoE',
                    ha='center', va='center',
                    fontsize=12, style='italic',
                    color=self.colors['text'])
        
        # 图例
        legend_elements = [
            mpatches.Patch(color=self.colors['input'], label='Input/Output'),
            mpatches.Patch(color=self.colors['preprocessing'], label='Preprocessing'),
            mpatches.Patch(color=self.colors['feature_extraction'], label='Feature Extraction'),
            mpatches.Patch(color=self.colors['sequence_modeling'], label='Sequence Modeling'),
            mpatches.Patch(color=self.colors['attention'], label='Attention Mechanism'),
            mpatches.Patch(color=self.colors['fusion'], label='Feature Fusion'),
            mpatches.Patch(color=self.colors['moe'], label='Mixture of Experts'),
            mpatches.Patch(color=self.colors['classification'], label='Classification')
        ]
        
        self.ax.legend(handles=legend_elements, 
                      loc='upper right', 
                      bbox_to_anchor=(0.98, 0.98),
                      fontsize=9,
                      framealpha=0.9)
        
        # 添加维度说明
        dim_text = ("B=Batch, C=Channels, T=Time, F=Frequency Bands, D=Feature Dimension")
        self.ax.text(11, 0.5, dim_text,
                    ha='center', va='center',
                    fontsize=9, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='lightgray', 
                             alpha=0.7))
    
    def save_figure(self, filename: str = 'mi_net_architecture.png'):
        """保存图像"""
        plt.savefig(filename, 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        print(f"架构图已保存为: {filename}")





def main():
    """主函数"""
    print("正在生成MI-Net架构图...")
    
    # 创建主架构图
    visualizer = MINetArchitectureVisualizer()
    visualizer.draw_complete_architecture()
    visualizer.save_figure('mi_net_architecture_professional.png')
    
    print("架构图生成完成！")
    plt.show()


if __name__ == "__main__":
    main()
