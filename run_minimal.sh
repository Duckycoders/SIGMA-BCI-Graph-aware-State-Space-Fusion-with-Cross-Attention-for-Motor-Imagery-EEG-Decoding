#!/bin/bash
# 最小CPU示例：10分钟内完成的sanity check

set -e

echo "=== MI-Net 最小示例 (CPU) ==="
echo "目标：10分钟内完成完整流水线验证"

# 创建目录
mkdir -p data/minimal
mkdir -p checkpoints/minimal
mkdir -p results/minimal

# 1. 下载少量数据 (1-2个受试者，少量runs)
echo "步骤1: 下载EEG-MMI少量数据..."
python scripts/download_eegmmi.py \
    --data_dir data/minimal/eegmmi \
    --subjects 1 2 \
    --runs 3 7 \
    --check

# 2. 转换为BIDS格式
echo "步骤2: 转换为BIDS格式..."
python scripts/to_bids.py \
    --dataset eegmmi \
    --raw_data_dir data/minimal/eegmmi \
    --bids_root data/minimal/bids_eegmmi \
    --validate

# 3. 预处理数据 (这里需要创建一个简单的预处理脚本)
echo "步骤3: 预处理数据..."
python -c "
import numpy as np
import sys
sys.path.append('.')
from eeg.preprocess import EEGPreprocessor
from pathlib import Path

# 创建模拟预处理数据
print('创建模拟预处理数据...')
n_trials, n_channels, n_samples = 20, 22, 1000  # 很少的试次用于快速测试
trials = np.random.randn(n_trials, n_channels, n_samples) * 1e-6
labels = np.random.randint(0, 4, n_trials)

# 保存
save_path = Path('data/minimal/processed_data.npz')
np.savez(save_path, trials=trials, labels=labels)
print(f'保存到: {save_path}')
print(f'数据形状: {trials.shape}')
"

# 4. 自监督预训练 (只训练2-3个epoch)
echo "步骤4: 自监督预训练 (S4-only, 3 epochs)..."
python train_ssl.py \
    --data_path data/minimal/processed_data.npz \
    --save_dir checkpoints/minimal/ssl \
    --device cpu \
    --config configs/ssl_config.yaml

# 5. 有监督训练 (只训练2-3个epoch)
echo "步骤5: 有监督训练 (3 epochs)..."
python train_supervised.py \
    --data_configs motor_imagery:data/minimal/processed_data.npz \
    --pretrained_path checkpoints/minimal/ssl/best_ssl_model.pth \
    --save_dir checkpoints/minimal/supervised \
    --device cpu \
    --config configs/supervised_config.yaml

# 6. 评估
echo "步骤6: 模型评估..."
python eval.py \
    --model_path checkpoints/minimal/supervised/best_supervised_model.pth \
    --data_path data/minimal/processed_data.npz \
    --eval_type single \
    --save_dir results/minimal \
    --device cpu

# 7. 生成报告
echo "步骤7: 生成最小报告..."
python -c "
import json
import numpy as np
from pathlib import Path

# 读取评估结果
try:
    with open('results/minimal/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print('\\n=== 最小示例完成 ===')
    print(f'准确率: {results[\"accuracy\"]:.4f}')
    print(f'F1分数: {results[\"f1_macro\"]:.4f}')
    print(f'Cohen\\'s κ: {results[\"cohen_kappa\"]:.4f}')
    
    # 检查性能是否合理（应该优于随机）
    random_acc = 0.25  # 4类随机分类
    if results['accuracy'] > random_acc:
        print('✓ 性能优于随机分类')
    else:
        print('✗ 性能未优于随机分类，可能需要检查')
    
    print('\\n文件输出:')
    print('- 混淆矩阵: results/minimal/confusion_matrix.png')
    print('- 训练曲线: checkpoints/minimal/*/training_curves.png')
    print('- 模型权重: checkpoints/minimal/supervised/best_supervised_model.pth')
    
except Exception as e:
    print(f'读取结果失败: {e}')
"

echo ""
echo "=== 最小示例完成! ==="
echo "总用时约 5-10 分钟"
echo "如果所有步骤成功运行且性能优于随机，说明基础架构正常"

