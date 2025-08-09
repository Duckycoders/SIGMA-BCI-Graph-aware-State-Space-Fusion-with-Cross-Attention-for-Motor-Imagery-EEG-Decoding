# MI-Net: 运动想象EEG解码深度学习框架

一个用于运动想象脑电信号解码的完整深度学习框架，整合了FilterBank、图卷积、S4/Mamba序列建模、跨注意力融合和混合专家(MoE)等先进技术。

## 🌟 主要特性

### 📊 支持数据集
- **EEG-MMI** (PhysioNet): 64通道，160Hz，4类运动想象
- **BNCI2014-001** (BCI Competition IV 2a): 22通道，250Hz，4类运动想象  
- **BNCI2014-004** (BCI Competition IV 2b): 3双极导联，250Hz，2类运动想象

### 🏗️ 先进架构
```
Input EEG → FilterBank → GraphConv → [S4分支, Mamba分支] → Cross-Attention融合 → MoE Adapter → 分类头
```

#### 核心模块
- **FilterBank**: 并行多频带滤波 (μ、α、β、γ波段)
- **GraphConv**: 基于PyTorch Geometric的电极空间关系建模
- **S4分支**: 结构化状态空间模型，长序列依赖建模
- **Mamba分支**: 选择性状态空间模型（支持自动回退到S4）
- **Cross-Attention**: S4↔Mamba双向注意力融合
- **MoE Adapter**: 混合专家机制，支持多任务学习

### 🚀 训练策略
- **自监督预训练**: 时间/通道遮罩重建
- **多任务联合训练**: 跨数据集共享表示学习
- **域适配**: 2a→2b few-shot适配，知识蒸馏
- **数据增广**: 基于Braindecode的丰富增广技术

### 📈 评估体系
- **LOSO**: 留一受试者交叉验证
- **跨数据集**: 2a/EEG-MMI → 2b性能评估
- **跨会话**: 不同session间泛化能力
- **消融实验**: 各模块贡献度分析
- **可解释性**: 注意力权重、频带重要性、通道贡献

## 🔧 快速开始

### 环境安装

#### 方式1: GPU环境 (CUDA 12.1)
```bash
# 使用预配置脚本
chmod +x install_cuda.sh
./install_cuda.sh
```

#### 方式2: CPU环境
```bash
# 使用CPU脚本
chmod +x install_cpu.sh  
./install_cpu.sh
```

#### 方式3: Conda环境
```bash
conda env create -f environment.yml
conda activate bci-eeg
```

#### Mamba依赖问题
如果遇到`mamba-ssm`安装失败：
```bash
# 常见解决方案
pip install packaging ninja
pip install "mamba-ssm[causal-conv1d]" --no-cache-dir

# 如果仍然失败，框架会自动回退到S4-only模式
```

### 10分钟快速验证

运行最小示例，验证整个流水线：
```bash
chmod +x run_minimal.sh
./run_minimal.sh
```

这将：
1. 下载少量EEG-MMI数据 (2个受试者)
2. 转换为BIDS格式
3. 预处理和增广
4. 3轮自监督预训练
5. 3轮有监督训练  
6. 评估和可视化

预期输出：
- 训练损失收敛
- 准确率 > 0.25 (优于4类随机分类)
- 生成混淆矩阵和训练曲线

## 📚 完整使用流程

### 1. 数据获取

#### EEG-MMI数据集
```bash
python scripts/download_eegmmi.py \
    --data_dir data/eegmmi \
    --subjects 1 2 3 4 5 \
    --runs 3 7 11 \
    --check
```

#### BNCI数据集
```bash
# 下载2a和2b数据集
python scripts/download_bnci.py \
    --data_dir data/bnci \
    --dataset both \
    --subjects 1 2 3 4 5 \
    --check
```

### 2. BIDS格式转换

```bash
# EEG-MMI
python scripts/to_bids.py \
    --dataset eegmmi \
    --raw_data_dir data/eegmmi \
    --bids_root data/bids_eegmmi \
    --validate

# BNCI 2a
python scripts/to_bids.py \
    --dataset bnci2a \
    --raw_data_dir data/bnci/bnci2014_001 \
    --bids_root data/bids_bnci2a \
    --validate

# BNCI 2b  
python scripts/to_bids.py \
    --dataset bnci2b \
    --raw_data_dir data/bnci/bnci2014_004 \
    --bids_root data/bids_bnci2b \
    --validate
```

### 3. 自监督预训练

```bash
python train_ssl.py \
    --data_path data/processed/bnci2a_trials.npz \
    --config configs/ssl_config.yaml \
    --save_dir checkpoints/ssl \
    --device cuda
```

配置参数 (`configs/ssl_config.yaml`):
- `mask_ratio: 0.5` - 遮罩比例
- `mask_type: 'time_block'` - 遮罩类型
- `max_epochs: 100` - 训练轮数

### 4. 多任务监督训练

```bash
python train_supervised.py \
    --data_configs \
        bnci2a:data/processed/bnci2a_trials.npz \
        eegmmi:data/processed/eegmmi_trials.npz \
    --pretrained_path checkpoints/ssl/best_ssl_model.pth \
    --config configs/supervised_config.yaml \
    --save_dir checkpoints/supervised \
    --device cuda
```

### 5. 跨数据集适配 (2a→2b)

```bash
python adapt_2b.py \
    --source_model_path checkpoints/supervised/best_supervised_model.pth \
    --source_data_path data/processed/bnci2a_trials.npz \
    --target_data_path data/processed/bnci2b_trials.npz \
    --adaptation_ratio 0.1 \
    --save_dir checkpoints/adaptation \
    --device cuda
```

### 6. 模型评估

#### LOSO评估
```bash
python eval.py \
    --model_path checkpoints/supervised/best_supervised_model.pth \
    --data_path data/processed/bnci2a_trials.npz \
    --eval_type loso \
    --save_dir results/loso \
    --device cuda
```

#### 跨数据集评估
```bash
python eval.py \
    --model_path checkpoints/supervised/best_supervised_model.pth \
    --data_path data/processed/bnci2a_trials.npz \
    --target_data_path data/processed/bnci2b_trials.npz \
    --eval_type cross_dataset \
    --save_dir results/cross_dataset \
    --device cuda
```

### 7. 消融实验

```bash
python ablate.py \
    --data_path data/processed/bnci2a_trials.npz \
    --save_dir results/ablation \
    --device cuda
```

## 📁 项目结构

```
BCI/
├── README.md                 # 项目说明
├── requirements.txt          # Python依赖
├── environment.yml          # Conda环境
├── install_cuda.sh          # GPU安装脚本
├── install_cpu.sh           # CPU安装脚本
├── run_minimal.sh           # 最小示例脚本
│
├── configs/                 # 配置文件
│   ├── ssl_config.yaml      # 自监督预训练配置
│   └── supervised_config.yaml # 监督训练配置
│
├── scripts/                 # 数据处理脚本
│   ├── download_eegmmi.py   # EEG-MMI数据下载
│   ├── download_bnci.py     # BNCI数据下载
│   └── to_bids.py          # BIDS格式转换
│
├── eeg/                     # EEG处理模块
│   ├── __init__.py
│   ├── preprocess.py        # 预处理和FilterBank
│   └── augment.py          # 数据增广
│
├── models/                  # 模型架构
│   ├── __init__.py
│   ├── filterbank.py        # FilterBank模块
│   ├── graph.py            # 图卷积模块
│   ├── s4_branch.py        # S4分支
│   ├── mamba_branch.py     # Mamba分支
│   ├── fusion.py           # 跨注意力融合
│   ├── moe_adapter.py      # MoE适配器
│   └── mi_net.py           # 完整MI-Net
│
├── explain/                 # 可解释性分析
│   ├── attn_maps.py        # 注意力可视化
│   ├── band_importance.py  # 频带重要性
│   └── spatial_importance.py # 空间重要性
│
├── train_ssl.py            # 自监督预训练
├── train_supervised.py     # 监督联合训练
├── adapt_2b.py            # 域适配脚本
├── eval.py                # 评估脚本
└── ablate.py              # 消融实验
```

## ⚙️ 配置说明

### 模型架构配置

在配置文件中可调整各模块：

```yaml
model:
  # 数据配置
  n_channels: 22
  n_samples: 1000
  sfreq: 250.0
  
  # FilterBank配置
  filter_bands: [[4, 8], [8, 14], [14, 30], [30, 45]]
  use_adaptive_filterbank: false
  
  # 图卷积配置  
  use_graph_conv: true
  graph_hidden_dims: [64, 128]
  
  # S4分支配置
  use_s4_branch: true
  s4_d_model: 128
  s4_n_layers: 2
  
  # Mamba分支配置
  use_mamba_branch: true
  mamba_d_model: 128
  mamba_n_layers: 2
  
  # 融合配置
  fusion_method: 'cross_attention'
  
  # MoE配置
  use_moe: true
  moe_n_experts: 4
  moe_top_k: 2
```

### 消融实验配置

通过`module_switches`控制模块开关：

```python
# 只使用S4分支
model = create_mi_net(
    'bnci2a',
    module_switches={'mamba_branch': False}
)

# 不使用FilterBank
model = create_mi_net(
    'bnci2a', 
    module_switches={'filterbank': False}
)

# 不使用图卷积
model = create_mi_net(
    'bnci2a',
    module_switches={'graph_conv': False}
)
```

## 📊 性能指标

模型评估包含以下指标：
- **准确率 (Accuracy)**: 分类准确率
- **宏平均F1**: 多类平衡F1分数
- **Cohen's κ**: 考虑随机一致性的性能指标
- **95%置信区间**: 统计显著性验证

### 典型性能范围
- **BNCI 2a**: 70-85% (LOSO)
- **BNCI 2b**: 75-90% (LOSO)
- **EEG-MMI**: 65-80% (LOSO)
- **跨数据集**: 通常有5-15%性能下降

## 🔍 可解释性分析

### 注意力权重可视化
```python
from explain.attn_maps import visualize_attention

# 可视化跨注意力权重
visualize_attention(
    model, data, 
    save_path='attention_maps.png'
)
```

### 频带重要性分析
```python
from explain.band_importance import analyze_band_importance

# 分析各频带贡献
importance = analyze_band_importance(
    model, data,
    bands=['μ', 'α', 'β', 'γ']
)
```

### 空间重要性分析
```python
from explain.spatial_importance import plot_channel_importance

# 绘制通道重要性头皮图
plot_channel_importance(
    model, data,
    save_path='channel_importance.png'
)
```

## 🐛 常见问题解决

### Mamba安装失败
```bash
# 解决方案1: 更新编译工具
pip install --upgrade setuptools wheel

# 解决方案2: 使用预编译版本
pip install mamba-ssm --find-links https://github.com/state-spaces/mamba/releases

# 解决方案3: 自动回退
# 框架会自动检测并回退到S4-only模式
```

### PyTorch Geometric安装问题
```bash
# 确保PyTorch版本匹配
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安装对应版本的PyG
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### 内存不足
```bash
# 减少批次大小
--batch_size 16

# 使用梯度累积
--gradient_accumulation_steps 2

# 启用混合精度训练
--use_amp true
```

### 数据加载错误
```bash
# 检查数据格式
python -c "import numpy as np; data=np.load('data.npz'); print(list(data.keys()))"

# 验证BIDS格式
python scripts/to_bids.py --validate
```

## 📈 实验复现

### 论文实验设置
1. **数据集**: BNCI2014-001, BNCI2014-004, EEG-MMI
2. **评估**: 10折LOSO交叉验证
3. **指标**: Accuracy, Macro-F1, Cohen's κ
4. **预处理**: 0.5-100Hz滤波, 1-4s试次窗口
5. **增广**: 30%概率应用Braindecode增广

### 消融实验
- [ ] 完整模型 vs 单分支模型
- [ ] 有无FilterBank对比
- [ ] 不同融合方法对比  
- [ ] MoE vs 标准分类头
- [ ] 预训练 vs 随机初始化

### 预期结果
运行完整实验应得到：
- 训练收敛曲线
- LOSO性能报告
- 跨数据集泛化分析
- 注意力权重可视化
- 消融实验对比表

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 克隆项目
git clone <repo-url>
cd BCI

# 安装开发依赖
pip install -e .
pip install pytest black flake8

# 运行测试
pytest tests/

# 代码格式化
black .
flake8 .
```

### 贡献方向
- [ ] 新的序列建模架构
- [ ] 更多数据集支持
- [ ] 新的数据增广方法
- [ ] 模型压缩和加速
- [ ] 实时解码接口
- [ ] 可解释性方法

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📚 引用

如果这个项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{minet2024,
  title={MI-Net: A Deep Learning Framework for Motor Imagery EEG Decoding},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/BCI}
}
```

## 📧 联系

- 作者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目主页: [https://github.com/your-username/BCI]

---

⭐ 如果这个项目对您有帮助，请给个Star！

