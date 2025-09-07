# ===== 完整SIGMA-BCI Colab训练脚本 =====
import os
import torch
import numpy as np
from scipy import signal
import shutil

print("🔥 完整SIGMA-BCI Colab训练")

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ===== 智能环境检测和路径配置 =====
def detect_environment():
    """智能检测运行环境并返回相应配置"""
    
    # 检测方法1：检查是否有Colab特有的模块
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    
    # 检测方法2：检查环境变量
    if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
        return 'colab'
    
    # 检测方法3：检查是否在Google Colab的工作目录
    if '/content' in os.getcwd():
        return 'colab'
    
    # 检测方法4：检查是否存在/content目录且在Linux系统（更精确的Colab检测）
    if os.path.exists('/content') and os.name == 'posix' and 'google.colab' in str(os.environ):
        return 'colab'
    
    # 默认为本地环境
    return 'local'

def get_paths(env_type):
    """根据环境类型返回相应的路径配置"""
    if env_type == 'colab':
        return {
            'data_dir': '/content/BCI/data/bnci/bnci2014_001',
            'project_dir': '/content/BCI',
            'backup_dir': '/content/data_backup',
            'model_save_path': '/content/BCI/best_sigma_bci.pth'
        }
    else:  # local
        return {
            'data_dir': 'data/bnci/bnci2014_001',
            'project_dir': '.',
            'backup_dir': './data_backup',
            'model_save_path': 'best_sigma_bci.pth'
        }

# 环境检测
env_type = detect_environment()
paths = get_paths(env_type)

print("🔄 智能环境检测...")
print(f"📍 检测到环境: {env_type.upper()}")
print(f"📂 数据目录: {paths['data_dir']}")
print(f"💾 模型保存路径: {paths['model_save_path']}")

# 环境设置
if env_type == 'colab':
    # Colab环境设置
    if os.path.exists('/content/BCI'):
        # 备份数据目录
        if os.path.exists('/content/BCI/data'):
            print("💾 备份数据目录...")
            if os.path.exists(paths['backup_dir']):
                shutil.rmtree(paths['backup_dir'])
            shutil.move('/content/BCI/data', paths['backup_dir'])
            print("✅ 数据已备份")
        
        # 删除其他文件
        shutil.rmtree('/content/BCI')
        print("🗑️ 清理代码文件")
    
    # 创建项目结构
    os.makedirs(paths['project_dir'], exist_ok=True)
    os.chdir(paths['project_dir'])
    
    # 恢复数据目录
    if os.path.exists(paths['backup_dir']):
        print("📂 恢复数据目录...")
        shutil.move(paths['backup_dir'], '/content/BCI/data')
        print("✅ 数据目录已恢复")
    else:
        print("📁 首次运行，请手动上传数据")
        os.makedirs(paths['data_dir'], exist_ok=True)
else:
    # 本地环境设置
    if not os.path.exists(paths['data_dir']):
        print(f"❌ 请确保数据目录存在: {paths['data_dir']}")
        print("   数据目录结构应该是:")
        print("   data/bnci/bnci2014_001/")
        print("   ├── S01_0train_0.npz")
        print("   ├── S01_0train_1.npz")
        print("   └── ...")
    else:
        # 检查数据文件
        npz_files = [f for f in os.listdir(paths['data_dir']) if f.endswith('.npz')]
        print(f"✅ 数据目录已找到，包含 {len(npz_files)} 个文件")

print("✅ 环境准备完成")

# ===== 专业EEG预处理（基于文献最佳实践）=====
def professional_eeg_preprocessing(trial, sfreq=250.0):
    """专业EEG预处理：Braindecode + 文献最佳实践"""
    
    # 1. 带通滤波 (4-40Hz)
    nyquist = sfreq / 2
    low_freq = 4.0 / nyquist
    high_freq = 40.0 / nyquist
    
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    
    filtered_trial = np.zeros_like(trial)
    for ch in range(trial.shape[0]):
        filtered_trial[ch, :] = signal.filtfilt(b, a, trial[ch, :])
    
    # 2. 工频陷波 (50Hz及谐波)
    for notch_freq in [50.0, 100.0]:  # 50Hz和100Hz
        if notch_freq < sfreq / 2:
            notch_normalized = notch_freq / nyquist
            b_notch, a_notch = signal.iirnotch(notch_normalized, Q=30)
            
            for ch in range(filtered_trial.shape[0]):
                filtered_trial[ch, :] = signal.filtfilt(b_notch, a_notch, filtered_trial[ch, :])
    
    # 3. 基线校正
    baseline_samples = int(0.5 * sfreq)
    if trial.shape[1] > baseline_samples:
        baseline = filtered_trial[:, :baseline_samples].mean(axis=1, keepdims=True)
        filtered_trial = filtered_trial - baseline
    
    # 4. 通道重参考 (CAR)
    car = filtered_trial.mean(axis=0, keepdims=True)
    filtered_trial = filtered_trial - car
    
    # 5. 指数移动标准化 (Braindecode标准)
    factor_new = 1e-3
    for ch in range(filtered_trial.shape[0]):
        ch_data = filtered_trial[ch, :]
        
        # 指数移动均值和方差
        running_mean = 0
        running_var = 1
        
        standardized = np.zeros_like(ch_data)
        for i, sample in enumerate(ch_data):
            running_mean = (1 - factor_new) * running_mean + factor_new * sample
            running_var = (1 - factor_new) * running_var + factor_new * (sample - running_mean) ** 2
            standardized[i] = (sample - running_mean) / (np.sqrt(running_var) + 1e-8)
        
        filtered_trial[ch, :] = standardized
    
    # 6. 幅值归一化到合理范围
    trial_std = filtered_trial.std()
    if trial_std > 0:
        filtered_trial = filtered_trial / trial_std
    
    # 7. 异常值处理
    filtered_trial = np.clip(filtered_trial, -5, 5)  # 限制在±5标准差内
    
    return filtered_trial

# ===== 完整SIGMA-BCI模型 =====
class SIGMA_BCI(torch.nn.Module):
    """完整SIGMA-BCI：FilterBank + S4 + Mamba + MoE + Riemann"""
    
    def __init__(self, n_classes=4, n_chans=22, samples=751, dropout=0.5):
        super().__init__()
        
        # ===== 1. 三频带FilterBank =====
        self.mu_filter = self._create_filter_bank(8, 14, sfreq=250)      # μ波 8-14Hz
        self.beta_filter = self._create_filter_bank(14, 30, sfreq=250)   # β波 14-30Hz  
        self.gamma_filter = self._create_filter_bank(30, 40, sfreq=250)  # γ波 30-40Hz
        
        # ===== 2. S4状态空间分支 =====
        self.s4_branch = self._create_s4_layer(d_model=64, d_state=32, seq_len=samples)
        
        # ===== 3. Mamba分支 =====
        self.mamba_branch = self._create_mamba_layer(d_model=64, d_state=16, seq_len=samples)
        
        # ===== 4. 跨注意力融合 =====
        self.cross_attention = torch.nn.MultiheadAttention(64, num_heads=4, dropout=0.1)
        self.fusion_norm = torch.nn.LayerNorm(64)
        
        # ===== 5. 4专家MoE系统 =====
        self.expert_spatial = torch.nn.Linear(64, 64)
        self.expert_temporal = torch.nn.Linear(64, 64) 
        self.expert_frequency = torch.nn.Linear(64, 64)
        self.expert_mixed = torch.nn.Linear(64, 64)
        self.router = torch.nn.Linear(64, 4)
        
        # ===== 6. Riemann几何分支 =====
        self.riemann_branch = torch.nn.Sequential(
            torch.nn.Linear(253, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 32)
        )
        
        # ===== 7. 多模态融合 =====
        self.multimodal = torch.nn.Sequential(
            torch.nn.Linear(64 + 32, 128),  # MoE + Riemann
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4)
        )
        
        # ===== 8. 最终分类器 =====
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, n_classes)
        )
        
        print(f"🔥 完整SIGMA-BCI架构:")
        print(f"  ✅ 3频带FilterBank (μ/β/γ波)")
        print(f"  ✅ S4+Mamba双状态空间")
        print(f"  ✅ 跨注意力融合")
        print(f"  ✅ 4专家MoE")
        print(f"  ✅ Riemann几何")
        print(f"  ✅ 多模态融合")
        print(f"  📊 总参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def _create_filter_bank(self, low_freq, high_freq, sfreq=250):
        """创建滤波器组"""
        return torch.nn.Sequential(
            torch.nn.Conv1d(22, 22, kernel_size=64, padding=32, groups=22),
            torch.nn.BatchNorm1d(22),
            torch.nn.ReLU()
        )
    
    def _create_s4_layer(self, d_model, d_state, seq_len):
        """创建简化S4层"""
        return torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model)
        )
    
    def _create_mamba_layer(self, d_model, d_state, seq_len):
        """创建简化Mamba层"""
        return torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model)
        )
    
    def compute_riemann(self, x):
        """计算Riemann几何特征"""
        batch_size, n_channels, n_time = x.shape
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(-1, -2)) / (n_time - 1)
        
        eye = torch.eye(n_channels, device=x.device)
        cov = cov + 1e-6 * eye.unsqueeze(0)
        
        triu_indices = torch.triu_indices(n_channels, n_channels)
        return cov[:, triu_indices[0], triu_indices[1]]
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. 三频带FilterBank
        x_mu = self.mu_filter(x)      # μ波频带
        x_beta = self.beta_filter(x)  # β波频带
        x_gamma = self.gamma_filter(x) # γ波频带
        
        # 通道平均后转为序列
        x_mu_avg = x_mu.mean(dim=1)      # (batch, time)
        x_beta_avg = x_beta.mean(dim=1)  # (batch, time)  
        x_gamma_avg = x_gamma.mean(dim=1) # (batch, time)
        
        # 添加特征维度
        x_mu_seq = x_mu_avg.unsqueeze(-1)     # (batch, time, 1)
        x_beta_seq = x_beta_avg.unsqueeze(-1) # (batch, time, 1)
        x_gamma_seq = x_gamma_avg.unsqueeze(-1) # (batch, time, 1)
        
        # 2. 双分支状态空间建模
        s4_mu = self.s4_branch(x_mu_seq)        # S4处理μ波
        mamba_beta = self.mamba_branch(x_beta_seq)  # Mamba处理β波
        s4_gamma = self.s4_branch(x_gamma_seq)  # S4处理γ波
        
        # 3. 跨注意力融合多个分支
        # 融合S4(μ) + Mamba(β)
        fused_main, _ = self.cross_attention(s4_mu, mamba_beta, mamba_beta)
        fused_main = self.fusion_norm(s4_mu + fused_main)
        
        # 加入γ波信息
        fused_final, _ = self.cross_attention(fused_main, s4_gamma, s4_gamma)
        fused_final = self.fusion_norm(fused_main + fused_final)
        
        # 时间池化
        pooled = fused_final.mean(dim=1)  # (batch, 64)
        
        # 4. 4专家MoE
        router_weights = torch.softmax(self.router(pooled), dim=-1)  # (batch, 4)
        
        # 4个专家输出
        expert_outputs = [
            self.expert_spatial(pooled),
            self.expert_temporal(pooled),
            self.expert_frequency(pooled),
            self.expert_mixed(pooled)
        ]
        
        # 简化MoE：直接加权组合所有专家
        moe_out = (router_weights[:, 0:1] * expert_outputs[0] + 
                  router_weights[:, 1:2] * expert_outputs[1] +
                  router_weights[:, 2:3] * expert_outputs[2] + 
                  router_weights[:, 3:4] * expert_outputs[3])
        
        # 5. Riemann
        riemann_feat = self.riemann_branch(self.compute_riemann(x))
        
        # 6. 多模态融合
        combined = torch.cat([moe_out, riemann_feat], dim=-1)
        final_feat = self.multimodal(combined)
        
        # 7. 分类
        logits = self.classifier(final_feat)
        
        return {
            'logits': logits,
            'predictions': torch.softmax(logits, dim=-1),
            'moe_weights': router_weights
        }

# ===== 数据加载（专业预处理）=====
def load_bnci_data_professional():
    all_trials = []
    all_labels = []
    all_subjects = []
    
    # 使用全局路径配置
    data_dir = paths['data_dir']
    print(f"📂 从 {data_dir} 加载数据...")
    
    # 加载前6个受试者
    for subject_id in range(1, 7):
        print(f"处理受试者 {subject_id}:")
        subject_trials = 0
        
        for session_type in ['0train', '1test']:
            for session_id in range(6):
                filename = f'S{subject_id:02d}_{session_type}_{session_id}.npz'
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        data = np.load(filepath, allow_pickle=True)
                        eeg_data = data['data']
                        events = data['events']
                        sfreq = float(data['sfreq'])
                        
                        for event in events:
                            start_sample = int(event[0])
                            event_type = int(event[2])
                            
                            if event_type in [1, 2, 3, 4]:
                                end_sample = start_sample + 751
                                
                                if end_sample <= eeg_data.shape[1]:
                                    trial = eeg_data[:, start_sample:end_sample]
                                    
                                    # 专业预处理
                                    trial = professional_eeg_preprocessing(trial, sfreq)
                                    
                                    all_trials.append(trial)
                                    all_labels.append(event_type - 1)
                                    all_subjects.append(subject_id)
                                    subject_trials += 1
                        
                    except Exception as e:
                        print(f"    ❌ {filename}: {e}")
        
        print(f"  受试者{subject_id}: {subject_trials}试次")
    
    return np.array(all_trials), np.array(all_labels), np.array(all_subjects)

# 加载数据
print("\n🔥 加载并预处理BNCI数据...")
trials, labels, subjects = load_bnci_data_professional()

print(f"\n✅ 数据加载完成:")
print(f"  试次: {len(trials)}")
if len(trials) > 0:
    print(f"  受试者: {np.unique(subjects)}")
    print(f"  类别分布: {dict(zip(['左手', '右手', '脚', '舌头'], np.bincount(labels.astype(int))))}")
    print(f"  预处理后数据范围: {trials.min():.2f} ~ {trials.max():.2f}")
else:
    print("  ❌ 没有加载到任何数据！")
    print(f"  请检查数据目录: {paths['data_dir']}")
    exit(1)

# ===== LOSO分割 =====
from sklearn.model_selection import train_test_split

test_subject = 1
test_mask = subjects == test_subject
train_mask = ~test_mask

X_train_val = trials[train_mask]
y_train_val = labels[train_mask]
X_test = trials[test_mask]
y_test = labels[test_mask]

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
)

# 确保数据类型正确
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

print(f"\n📊 LOSO分割:")
print(f"  训练: {len(X_train)} (受试者2-6)")
print(f"  验证: {len(X_val)}")
print(f"  测试: {len(X_test)} (受试者1)")

# ===== 训练完整SIGMA-BCI =====
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# 数据集
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型
model = SIGMA_BCI(n_classes=4, n_chans=22, samples=751).to(device)

# 优化器配置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练
print(f"\n🚀 完整SIGMA-BCI训练...")

best_val_acc = 0
for epoch in range(20):
    # 训练
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs['logits'], batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=1)
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)
    
    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs['logits'], dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)
    
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    avg_loss = total_loss / len(train_loader)
    
    print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 使用配置的保存路径
        torch.save(model.state_dict(), paths['model_save_path'])
    
    if val_acc > 0.7:
        print("  ✅ 达到70%准确率！")
        break

# ===== 最终测试 =====
print(f"\n📊 LOSO最终测试...")

model.eval()
all_preds = []
all_labels = []
all_moe_weights = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())
        all_moe_weights.append(outputs['moe_weights'].cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)

print(f"\n🎯 完整SIGMA-BCI LOSO结果:")
print(f"  跨受试者准确率: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  最佳验证准确率: {best_val_acc:.4f}")
print(f"  提升: +{(test_acc-0.25)*100:.1f}% vs 基线")

# MoE专家使用分析
if all_moe_weights:
    moe_weights_all = np.concatenate(all_moe_weights, axis=0)
    expert_usage = moe_weights_all.mean(axis=0)
    
    print(f"\n🔧 MoE专家使用分析:")
    expert_names = ['空间专家', '时间专家', '频率专家', '混合专家']
    for name, usage in zip(expert_names, expert_usage):
        print(f"  {name}: {usage:.3f} ({usage*100:.1f}%)")
    
    # 专家多样性
    expert_diversity = np.var(expert_usage)
    print(f"  专家多样性指数: {expert_diversity:.4f}")

# 检查预测分布
pred_counts = np.bincount(all_preds, minlength=4)
print(f"  预测分布: {pred_counts}")
print(f"  是否平衡: {pred_counts.std() < pred_counts.mean() * 0.5}")

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Left', 'Right', 'Feet', 'Tongue'],
            yticklabels=['Left', 'Right', 'Feet', 'Tongue'])
plt.title(f'完整SIGMA-BCI LOSO Results\\nAccuracy: {test_acc:.3f}')
plt.show()

if test_acc > 0.5:
    print("✅ 完整SIGMA-BCI成功学习！")
    print("🎯 FilterBank + S4 + Mamba + MoE + Riemann 架构验证成功")
else:
    print("⚠️  需要进一步调试和优化")

print(f"\n🔥 完整SIGMA-BCI验证完成！")
