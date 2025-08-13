# BCI项目 - Google Colab使用指南

## 🎯 为什么使用Google Colab？

在Windows CPU环境下，Mamba-SSM无法安装（需要CUDA编译器）。Google Colab提供免费的GPU环境，可以完整运行本项目的所有功能：

- ✅ **Mamba分支**: 选择性状态空间模型
- ✅ **图卷积**: PyTorch Geometric图神经网络
- ✅ **GPU加速**: 训练速度提升10-100倍
- ✅ **免费使用**: 每天12小时免费GPU时间

## 🚀 快速开始

### 步骤1: 上传代码到GitHub

1. 在GitHub创建新仓库（如：`BCI-EEG-Decoding`）
2. 将当前项目代码上传：

```bash
# 在项目目录下
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/BCI-EEG-Decoding.git
git push -u origin main
```

### 步骤2: 在Google Colab中运行

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `colab_train.ipynb` 文件
3. 修改notebook中的GitHub仓库地址
4. 设置GPU运行时：
   - 点击 `运行时` → `更改运行时类型`
   - 硬件加速器选择 `GPU` (T4/V100)
5. 按顺序运行所有cell

## 📊 完整功能对比

| 功能 | Windows CPU | Google Colab GPU |
|------|-------------|------------------|
| FilterBank | ✅ | ✅ |
| 图卷积 | ✅ (已修复) | ✅ |
| S4分支 | ✅ | ✅ |
| Mamba分支 | ❌ (需要CUDA) | ✅ |
| 跨注意力融合 | ✅ | ✅ |
| MoE专家混合 | ✅ | ✅ |
| 训练速度 | 慢 (CPU) | 快 (GPU) |

## 🔧 Colab环境优势

### 自动安装所有依赖
```python
# 在Colab中自动安装：
- torch (GPU版本)
- torch-geometric 
- mamba-ssm ✅
- causal-conv1d ✅
- 所有其他依赖
```

### GPU加速训练
- **CPU训练**: 每个epoch ~2-3分钟
- **GPU训练**: 每个epoch ~10-30秒
- **总提升**: 5-10倍速度提升

### 更大的数据集
```python
# CPU环境限制
subjects = [1]  # 只能处理1个受试者

# GPU环境可以处理更多
subjects = [1, 2, 3, 4, 5]  # 处理多个受试者
batch_size = 64  # 更大的batch size
max_epochs = 100  # 更多训练轮数
```

## 📁 项目文件说明

```
BCI/
├── colab_setup.py          # Colab环境自动设置脚本
├── colab_train.ipynb       # Colab训练notebook
├── README_COLAB.md         # 本文件
├── models/                 # 模型代码（完整功能）
├── configs/               # 配置文件
├── scripts/               # 数据处理脚本
└── requirements.txt       # 依赖列表
```

## 🎯 预期结果

使用完整功能（Mamba + 图卷积）在GPU上训练，预期性能：

- **准确率**: 70-85% (vs 25%随机基线)
- **F1分数**: 0.65-0.80
- **Cohen's κ**: 0.50-0.75
- **训练时间**: 10-30分钟（完整流程）

## 🔍 故障排除

### 常见问题

1. **GPU配额用完**
   - 解决：等待12小时重置，或使用Colab Pro

2. **内存不足**
   - 解决：减小batch_size到16或32

3. **GitHub克隆失败**
   - 解决：检查仓库地址是否正确，确保仓库是public

### 监控训练进度

```python
# 在Colab中实时查看
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 训练过程中会自动显示：
# - 损失曲线
# - 准确率变化
# - GPU使用情况
```

## 💡 进阶使用

### 1. 多数据集训练
```python
# 在Colab中可以同时使用多个数据集
datasets = ['bnci2a', 'bnci2b', 'eegmmi']
```

### 2. 超参数调优
```python
# GPU环境下可以快速测试不同参数
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]
```

### 3. 消融实验
```python
# 完整的消融实验对比
ablation_configs = [
    {'mamba': True, 'graph': True},   # 完整模型
    {'mamba': False, 'graph': True},  # 无Mamba
    {'mamba': True, 'graph': False},  # 无图卷积
    {'mamba': False, 'graph': False}  # 基础模型
]
```

## 📞 技术支持

如果在Colab中遇到问题：

1. 检查GPU是否启用：`!nvidia-smi`
2. 检查依赖安装：运行 `colab_setup.py`
3. 查看详细错误日志
4. 重启运行时并重新运行

---

🎉 **祝您在Google Colab中训练愉快！完整的Mamba + 图卷积功能等着您！**
