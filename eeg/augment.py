#!/usr/bin/env python3
"""
EEG数据增广模块
基于Braindecode实现各种数据增广技术
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Union
from braindecode.augmentation import (
    TimeReverse, SignFlip, FTSurrogate, SmoothTimeMask, 
    FrequencyShift, ChannelsShuffle, ChannelsDropout,
    GaussianNoise, IdentityTransform
)
from braindecode.augmentation.base import AugmentedDataLoader, Transform
import random


class MixUp(Transform):
    """Mixup数据增广"""
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability=probability)
        self.alpha = alpha
        
    def operation(self, X: torch.Tensor, y: torch.Tensor = None):
        if X.dim() == 3:  # (batch, channels, time)
            batch_size = X.shape[0]
            
            # 生成混合权重
            lam = np.random.beta(self.alpha, self.alpha)
            
            # 随机排列索引
            indices = torch.randperm(batch_size)
            
            # 混合数据
            X_mixed = lam * X + (1 - lam) * X[indices]
            
            if y is not None:
                y_mixed = (y, y[indices], lam)
                return X_mixed, y_mixed
            
            return X_mixed
        else:
            return X


class EEGAugmentationPipeline:
    """EEG数据增广管线"""
    
    def __init__(self, 
                 transforms_config: Dict = None,
                 random_state: int = 42,
                 sfreq: float = 250.0):
        """
        初始化增广管线
        
        Args:
            transforms_config: 增广配置字典
            random_state: 随机种子
            sfreq: 采样率（用于频移等操作）
        """
        self.random_state = random_state
        self.sfreq = sfreq
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        if transforms_config is None:
            transforms_config = self.get_default_config()
            
        self.transforms_config = transforms_config
        self.transforms = self._build_transforms()
        
        print(f"EEG增广管线初始化:")
        for name, config in transforms_config.items():
            if config.get('enabled', True):
                print(f"  {name}: 概率={config.get('probability', 0.5)}")
    
    def get_default_config(self) -> Dict:
        """获取默认增广配置"""
        return {
            'time_reverse': {
                'enabled': True,
                'probability': 0.3,
            },
            'sign_flip': {
                'enabled': True, 
                'probability': 0.3,
            },
            'ft_surrogate': {
                'enabled': True,
                'probability': 0.2,
                'phase_noise_magnitude': 1.0,
            },
            'smooth_time_mask': {
                'enabled': False,  # 默认禁用以避免版本不兼容
                'probability': 0.3,
                'mask_len_samples': 50,
                'n_masks': 2,
            },
            'frequency_shift': {
                'enabled': False,  # 默认禁用以避免版本不兼容
                'probability': 0.2,
                'max_delta_freq': 2.0,
            },
            'channels_shuffle': {
                'enabled': False,  # 默认禁用以避免版本不兼容
                'probability': 0.2,
                'n_channels': 3,
            },
            'channels_dropout': {
                'enabled': False,  # 默认禁用以避免版本不兼容
                'probability': 0.2,
                'n_channels': 2,
            },
            'gaussian_noise': {
                'enabled': True,
                'probability': 0.3,
                'std': 0.01,
            },
            'mixup': {
                'enabled': True,
                'probability': 0.3,
                'alpha': 0.2,
            }
        }
    
    def _build_transforms(self) -> List[Transform]:
        """构建增广变换列表"""
        transforms = []
        
        for name, config in self.transforms_config.items():
            if not config.get('enabled', True):
                continue
                
            if name == 'time_reverse':
                transform = TimeReverse(
                    probability=config.get('probability', 0.3)
                )
            elif name == 'sign_flip':
                transform = SignFlip(
                    probability=config.get('probability', 0.3)
                )
            elif name == 'ft_surrogate':
                transform = FTSurrogate(
                    probability=config.get('probability', 0.2),
                    phase_noise_magnitude=config.get('phase_noise_magnitude', 1.0)
                )
            elif name == 'smooth_time_mask':
                # braindecode 1.0.0不支持n_masks参数
                transform = SmoothTimeMask(
                    config.get('probability', 0.3),
                    mask_len_samples=config.get('mask_len_samples', 50)
                )
            elif name == 'frequency_shift':
                # 需要提供采样率；签名 (probability, sfreq, max_delta_freq)
                transform = FrequencyShift(
                    config.get('probability', 0.2),
                    self.sfreq,
                    config.get('max_delta_freq', 2.0)
                )
            elif name == 'channels_shuffle':
                # 某些版本无n_channels参数，这里直接跳过以保证运行
                continue
            elif name == 'channels_dropout':
                # 某些版本无n_channels参数，这里直接跳过以保证运行
                continue
            elif name == 'gaussian_noise':
                transform = GaussianNoise(
                    probability=config.get('probability', 0.3),
                    std=config.get('std', 0.01)
                )
            elif name == 'mixup':
                transform = MixUp(
                    probability=config.get('probability', 0.3),
                    alpha=config.get('alpha', 0.2)
                )
            else:
                print(f"警告: 未知的增广类型 {name}")
                continue
                
            transforms.append(transform)
        
        return transforms
    
    def apply_single_trial(self, trial: np.ndarray, 
                          apply_probability: float = 0.5) -> np.ndarray:
        """
        对单个试次应用增广
        
        Args:
            trial: 单个试次数据，形状为 (n_channels, n_samples)
            apply_probability: 应用增广的概率
            
        Returns:
            增广后的试次数据
        """
        if np.random.random() > apply_probability:
            return trial
            
        # 转换为tensor
        trial_tensor = torch.FloatTensor(trial).unsqueeze(0)  # (1, n_channels, n_samples)
        
        # 随机选择一个增广变换
        if self.transforms:
            transform = np.random.choice(self.transforms)
            try:
                trial_aug = transform(trial_tensor)
                if isinstance(trial_aug, tuple):  # Mixup可能返回元组
                    trial_aug = trial_aug[0]
                return trial_aug.squeeze(0).numpy()
            except Exception as e:
                print(f"增广失败: {e}")
                return trial
        
        return trial
    
    def apply_batch(self, trials: np.ndarray, 
                   labels: np.ndarray = None,
                   augmentation_ratio: float = 0.5) -> tuple:
        """
        对批次数据应用增广
        
        Args:
            trials: 试次数据，形状为 (n_trials, n_channels, n_samples)
            labels: 标签数组
            augmentation_ratio: 增广比例（相对于原始数据）
            
        Returns:
            (增广后的试次, 增广后的标签)
        """
        n_trials = len(trials)
        n_aug_trials = int(n_trials * augmentation_ratio)
        
        if n_aug_trials == 0:
            return trials, labels
        
        print(f"生成 {n_aug_trials} 个增广试次 (原始: {n_trials})")
        
        # 随机选择要增广的试次
        aug_indices = np.random.choice(n_trials, size=n_aug_trials, replace=True)
        
        aug_trials = []
        aug_labels = []
        
        for idx in aug_indices:
            trial = trials[idx]
            aug_trial = self.apply_single_trial(trial, apply_probability=1.0)
            aug_trials.append(aug_trial)
            
            if labels is not None:
                aug_labels.append(labels[idx])
        
        aug_trials = np.array(aug_trials)
        
        # 合并原始和增广数据
        combined_trials = np.concatenate([trials, aug_trials], axis=0)
        
        if labels is not None:
            aug_labels = np.array(aug_labels)
            combined_labels = np.concatenate([labels, aug_labels], axis=0)
            return combined_trials, combined_labels
        
        return combined_trials, None
    
    def apply_filterbank_batch(self, trials_fb: np.ndarray,
                             labels: np.ndarray = None,
                             augmentation_ratio: float = 0.5) -> tuple:
        """
        对FilterBank数据应用增广
        
        Args:
            trials_fb: FilterBank试次数据，形状为 (n_bands, n_trials, n_channels, n_samples)
            labels: 标签数组
            augmentation_ratio: 增广比例
            
        Returns:
            (增广后的FilterBank试次, 增广后的标签)
        """
        n_bands, n_trials, n_channels, n_samples = trials_fb.shape
        n_aug_trials = int(n_trials * augmentation_ratio)
        
        if n_aug_trials == 0:
            return trials_fb, labels
        
        print(f"对FilterBank数据生成 {n_aug_trials} 个增广试次")
        
        # 为每个频带独立进行增广
        aug_trials_fb = np.zeros((n_bands, n_aug_trials, n_channels, n_samples))
        aug_indices = np.random.choice(n_trials, size=n_aug_trials, replace=True)
        
        for band_idx in range(n_bands):
            for aug_idx, orig_idx in enumerate(aug_indices):
                trial = trials_fb[band_idx, orig_idx]  # (n_channels, n_samples)
                aug_trial = self.apply_single_trial(trial, apply_probability=1.0)
                aug_trials_fb[band_idx, aug_idx] = aug_trial
        
        # 合并原始和增广数据
        combined_trials_fb = np.concatenate([trials_fb, aug_trials_fb], axis=1)
        
        if labels is not None:
            aug_labels = labels[aug_indices]
            combined_labels = np.concatenate([labels, aug_labels], axis=0)
            return combined_trials_fb, combined_labels
        
        return combined_trials_fb, None
    
    def create_augmented_dataset(self, trials: np.ndarray, 
                               labels: np.ndarray,
                               trials_fb: np.ndarray = None,
                               augmentation_ratio: float = 0.5,
                               shuffle: bool = True) -> Dict:
        """
        创建增广数据集
        
        Args:
            trials: 原始试次数据
            labels: 标签
            trials_fb: FilterBank试次数据（可选）
            augmentation_ratio: 增广比例
            shuffle: 是否打乱数据
            
        Returns:
            包含增广数据的字典
        """
        print(f"\n创建增广数据集...")
        print(f"原始数据: {len(trials)} 个试次")
        print(f"增广比例: {augmentation_ratio}")
        
        # 增广常规数据
        aug_trials, aug_labels = self.apply_batch(
            trials, labels, augmentation_ratio
        )
        
        result = {
            'trials': aug_trials,
            'labels': aug_labels,
        }
        
        # 增广FilterBank数据（如果提供）
        if trials_fb is not None:
            aug_trials_fb, _ = self.apply_filterbank_batch(
                trials_fb, labels, augmentation_ratio
            )
            result['trials_filterbank'] = aug_trials_fb
        
        # 打乱数据
        if shuffle:
            n_total = len(aug_trials)
            shuffle_indices = np.random.permutation(n_total)
            result['trials'] = result['trials'][shuffle_indices]
            result['labels'] = result['labels'][shuffle_indices]
            
            if 'trials_filterbank' in result:
                result['trials_filterbank'] = result['trials_filterbank'][:, shuffle_indices]
        
        print(f"增广后数据: {len(result['trials'])} 个试次")
        if 'trials_filterbank' in result:
            print(f"FilterBank形状: {result['trials_filterbank'].shape}")
        
        return result


def create_augmentation_pipeline(config: Dict = None, 
                               mode: str = 'training',
                               sfreq: float = 250.0) -> EEGAugmentationPipeline:
    """
    创建增广管线的便捷函数
    
    Args:
        config: 增广配置
        mode: 模式 ('training', 'light', 'heavy')
        sfreq: 采样率
        
    Returns:
        增广管线对象
    """
    if config is None:
        pipeline = EEGAugmentationPipeline()
        base_config = pipeline.get_default_config()
        
        if mode == 'light':
            # 轻量增广：降低概率
            for key in base_config:
                base_config[key]['probability'] *= 0.5
        elif mode == 'heavy':
            # 重度增广：提高概率
            for key in base_config:
                base_config[key]['probability'] = min(0.8, base_config[key]['probability'] * 1.5)
        
        config = base_config
    
    return EEGAugmentationPipeline(config, sfreq=sfreq)


if __name__ == "__main__":
    # 测试增广管线
    print("EEG增广管线测试")
    
    # 创建模拟数据
    n_trials, n_channels, n_samples = 10, 22, 250
    trials = np.random.randn(n_trials, n_channels, n_samples) * 1e-6
    labels = np.random.randint(0, 4, n_trials)
    
    # 创建增广管线
    aug_pipeline = create_augmentation_pipeline(mode='training', sfreq=250.0)
    
    # 应用增广
    aug_result = aug_pipeline.create_augmented_dataset(
        trials=trials,
        labels=labels,
        augmentation_ratio=0.5
    )
    
    print(f"\n测试完成!")
    print(f"原始数据: {trials.shape}")
    print(f"增广后数据: {aug_result['trials'].shape}")
    print(f"标签分布: {np.bincount(aug_result['labels'])}")
