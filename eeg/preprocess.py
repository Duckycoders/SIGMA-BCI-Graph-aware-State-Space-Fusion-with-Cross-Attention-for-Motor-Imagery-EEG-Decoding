#!/usr/bin/env python3
"""
EEG数据预处理模块
包含滤波、FilterBank、试次裁剪、标准化等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import mne
from mne import Epochs
from mne.filter import filter_data
from braindecode.preprocessing import exponential_moving_standardize
from scipy.signal import iirnotch, filtfilt
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """EEG预处理器"""
    
    def __init__(self, 
                 sfreq: float = 250.0,
                 low_freq: float = 0.5,
                 high_freq: float = 100.0,
                 notch_freq: Union[float, List[float]] = [50.0, 60.0],
                 filter_bands: List[Tuple[float, float]] = None,
                 trial_window: Tuple[float, float] = (1.0, 4.0),
                 baseline: Optional[Tuple[float, float]] = None,
                 standardize: bool = True,
                 standardize_factor: float = 1e-3):
        """
        初始化预处理器
        
        Args:
            sfreq: 采样频率
            low_freq: 高通滤波截止频率
            high_freq: 低通滤波截止频率  
            notch_freq: 工频陷波频率
            filter_bands: FilterBank频带列表，格式[(low1,high1), (low2,high2), ...]
            trial_window: 试次裁剪窗口，相对于事件onset的时间窗 (start, end)
            baseline: 基线校正窗口，None表示不进行基线校正
            standardize: 是否进行指数滑动标准化
            standardize_factor: 标准化因子
        """
        self.sfreq = sfreq
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq if isinstance(notch_freq, list) else [notch_freq]
        
        # 默认FilterBank频带：μ波、低β、高β、γ波
        if filter_bands is None:
            self.filter_bands = [(4, 8), (8, 14), (14, 30), (30, 45)]
        else:
            self.filter_bands = filter_bands
            
        self.trial_window = trial_window
        self.baseline = baseline
        self.standardize = standardize
        self.standardize_factor = standardize_factor
        
        print(f"EEG预处理器初始化:")
        print(f"  采样频率: {self.sfreq} Hz")
        print(f"  滤波范围: {self.low_freq}-{self.high_freq} Hz")
        print(f"  陷波频率: {self.notch_freq} Hz") 
        print(f"  FilterBank频带: {self.filter_bands}")
        print(f"  试次窗口: {self.trial_window[0]}-{self.trial_window[1]} s")
        
    def apply_basic_filters(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        应用基础滤波：带通滤波 + 工频陷波
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            滤波后的Raw对象
        """
        print(f"  应用基础滤波: {self.low_freq}-{self.high_freq} Hz")
        
        # 创建副本避免修改原数据
        raw_filt = raw.copy()
        
        # 带通滤波
        raw_filt.filter(l_freq=self.low_freq, h_freq=self.high_freq, 
                       picks='eeg', verbose=False)
        
        # 工频陷波
        for notch_f in self.notch_freq:
            if notch_f < self.sfreq / 2:  # 确保陷波频率在奈奎斯特频率内
                print(f"  应用陷波滤波: {notch_f} Hz")
                raw_filt.notch_filter(freqs=notch_f, picks='eeg', verbose=False)
        
        return raw_filt
    
    def apply_filterbank(self, data: np.ndarray) -> np.ndarray:
        """
        应用FilterBank并行子带滤波
        
        Args:
            data: EEG数据，形状为 (n_channels, n_samples)
            
        Returns:
            FilterBank输出，形状为 (n_bands, n_channels, n_samples)
        """
        print(f"  应用FilterBank: {len(self.filter_bands)}个频带")
        
        n_channels, n_samples = data.shape
        n_bands = len(self.filter_bands)
        
        # 初始化输出数组
        filtered_data = np.zeros((n_bands, n_channels, n_samples))
        
        for i, (low, high) in enumerate(self.filter_bands):
            print(f"    频带{i+1}: {low}-{high} Hz")
            
            # 对每个通道应用子带滤波
            for ch in range(n_channels):
                filtered_data[i, ch, :] = filter_data(
                    data[ch, :], 
                    sfreq=self.sfreq,
                    l_freq=low, 
                    h_freq=high,
                    verbose=False
                )
        
        return filtered_data
    
    def extract_trials(self, raw: mne.io.Raw, events: np.ndarray, 
                      event_id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取试次数据
        
        Args:
            raw: 滤波后的Raw对象
            events: 事件数组
            event_id: 事件ID字典
            
        Returns:
            trials: 试次数据，形状为 (n_trials, n_channels, n_samples)
            labels: 标签数组
        """
        print(f"  提取试次: 窗口 {self.trial_window[0]}-{self.trial_window[1]}s")
        
        # 创建Epochs对象
        epochs = Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=self.trial_window[0],
            tmax=self.trial_window[1],
            baseline=self.baseline,
            picks='eeg',
            preload=True,
            verbose=False
        )
        
        # 获取数据和标签
        trials = epochs.get_data()  # (n_trials, n_channels, n_samples)
        labels = epochs.events[:, -1]  # 事件代码
        
        print(f"    提取到 {len(trials)} 个试次")
        print(f"    数据形状: {trials.shape}")
        
        return trials, labels
    
    def apply_standardization(self, data: np.ndarray) -> np.ndarray:
        """
        应用指数滑动标准化
        
        Args:
            data: 输入数据，形状可以是 (n_trials, n_channels, n_samples) 
                 或 (n_bands, n_trials, n_channels, n_samples)
                 
        Returns:
            标准化后的数据
        """
        if not self.standardize:
            return data
            
        print(f"  应用指数滑动标准化 (factor={self.standardize_factor})")
        
        # 处理不同维度的输入
        if data.ndim == 3:  # (n_trials, n_channels, n_samples)
            standardized_data = np.zeros_like(data)
            n_trials, n_channels, n_samples = data.shape
            
            for trial in range(n_trials):
                for ch in range(n_channels):
                    standardized_data[trial, ch, :] = exponential_moving_standardize(
                        data[trial, ch, :], factor_new=self.standardize_factor
                    )
                    
        elif data.ndim == 4:  # (n_bands, n_trials, n_channels, n_samples)
            standardized_data = np.zeros_like(data)
            n_bands, n_trials, n_channels, n_samples = data.shape
            
            for band in range(n_bands):
                for trial in range(n_trials):
                    for ch in range(n_channels):
                        standardized_data[band, trial, ch, :] = exponential_moving_standardize(
                            data[band, trial, ch, :], factor_new=self.standardize_factor
                        )
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")
            
        return standardized_data
    
    def interpolate_bad_channels(self, raw: mne.io.Raw, 
                               bad_channels: List[str] = None) -> mne.io.Raw:
        """
        插值坏通道
        
        Args:
            raw: Raw对象
            bad_channels: 坏通道列表，None则自动检测
            
        Returns:
            插值后的Raw对象
        """
        raw_interp = raw.copy()
        
        if bad_channels is None:
            # 简单的坏通道检测：基于信号幅度
            data = raw_interp.get_data(picks='eeg')
            std_per_ch = np.std(data, axis=1)
            threshold = 3 * np.median(std_per_ch)
            
            bad_ch_indices = np.where(std_per_ch > threshold)[0]
            bad_channels = [raw_interp.ch_names[i] for i in bad_ch_indices 
                          if raw_interp.ch_names[i] in raw_interp.info['ch_names']]
        
        if bad_channels:
            print(f"  插值坏通道: {bad_channels}")
            raw_interp.info['bads'] = bad_channels
            raw_interp.interpolate_bads(reset_bads=True, verbose=False)
        
        return raw_interp
    
    def process_raw(self, raw: mne.io.Raw, events: np.ndarray, 
                   event_id: Dict[str, int], 
                   bad_channels: List[str] = None,
                   apply_filterbank: bool = True) -> Dict[str, np.ndarray]:
        """
        完整预处理流程
        
        Args:
            raw: 原始MNE Raw对象
            events: 事件数组
            event_id: 事件ID字典
            bad_channels: 坏通道列表
            apply_filterbank: 是否应用FilterBank
            
        Returns:
            包含处理结果的字典：
            - 'trials': 试次数据
            - 'labels': 标签
            - 'trials_filterbank': FilterBank试次数据（如果应用）
            - 'ch_names': 通道名
            - 'sfreq': 采样频率
        """
        print(f"\n开始EEG预处理...")
        print(f"原始数据: {len(raw.ch_names)}通道, {raw.info['sfreq']}Hz, {raw.times[-1]:.1f}s")
        
        # 1. 插值坏通道
        raw_clean = self.interpolate_bad_channels(raw, bad_channels)
        
        # 2. 基础滤波
        raw_filtered = self.apply_basic_filters(raw_clean)
        
        # 3. 提取试次
        trials, labels = self.extract_trials(raw_filtered, events, event_id)
        
        # 4. 标准化
        trials_std = self.apply_standardization(trials)
        
        result = {
            'trials': trials_std,
            'labels': labels,
            'ch_names': [ch for ch in raw_filtered.ch_names 
                        if ch in mne.pick_info(raw_filtered.info, sel='eeg')['ch_names']],
            'sfreq': raw_filtered.info['sfreq'],
            'event_id': event_id
        }
        
        # 5. FilterBank处理（可选）
        if apply_filterbank:
            print(f"\n应用FilterBank处理...")
            
            # 对每个试次应用FilterBank
            n_trials, n_channels, n_samples = trials_std.shape
            n_bands = len(self.filter_bands)
            
            trials_fb = np.zeros((n_bands, n_trials, n_channels, n_samples))
            
            for trial_idx in range(n_trials):
                # 对单个试次应用FilterBank
                trial_fb = self.apply_filterbank(trials_std[trial_idx])  # (n_bands, n_channels, n_samples)
                trials_fb[:, trial_idx, :, :] = trial_fb
            
            # 标准化FilterBank输出
            trials_fb_std = self.apply_standardization(trials_fb)
            result['trials_filterbank'] = trials_fb_std
            result['filter_bands'] = self.filter_bands
        
        print(f"\n预处理完成!")
        print(f"输出试次形状: {result['trials'].shape}")
        if 'trials_filterbank' in result:
            print(f"FilterBank输出形状: {result['trials_filterbank'].shape}")
            
        return result


def create_electrode_positions(ch_names: List[str]) -> Dict[str, np.ndarray]:
    """
    创建电极位置信息（用于图构建）
    
    Args:
        ch_names: 通道名列表
        
    Returns:
        电极位置字典，键为通道名，值为3D坐标
    """
    # 标准10-20系统电极位置（简化版）
    standard_pos = {
        'Fp1': [-0.3, 0.9, 0.3], 'Fp2': [0.3, 0.9, 0.3],
        'F3': [-0.6, 0.6, 0.6], 'F4': [0.6, 0.6, 0.6], 'Fz': [0, 0.7, 0.7],
        'C3': [-0.7, 0, 0.7], 'C4': [0.7, 0, 0.7], 'Cz': [0, 0, 1],
        'P3': [-0.6, -0.6, 0.6], 'P4': [0.6, -0.6, 0.6], 'Pz': [0, -0.7, 0.7],
        'O1': [-0.3, -0.9, 0.3], 'O2': [0.3, -0.9, 0.3],
        'F7': [-0.8, 0.4, 0.4], 'F8': [0.8, 0.4, 0.4],
        'T7': [-0.9, 0, 0.4], 'T8': [0.9, 0, 0.4],
        'P7': [-0.8, -0.4, 0.4], 'P8': [0.8, -0.4, 0.4],
        'FC1': [-0.4, 0.3, 0.8], 'FC2': [0.4, 0.3, 0.8],
        'CP1': [-0.4, -0.3, 0.8], 'CP2': [0.4, -0.3, 0.8],
    }
    
    # 为BNCI 2b数据集添加双极导联位置
    if 'C3-C4' in ch_names or len(ch_names) == 3:
        standard_pos.update({
            'C3-C4': [0, 0, 0.8],  # 双极导联的虚拟位置
            'C3': [-0.7, 0, 0.7],
            'C4': [0.7, 0, 0.7]
        })
    
    # 只返回存在的通道
    positions = {}
    for ch in ch_names:
        if ch in standard_pos:
            positions[ch] = np.array(standard_pos[ch])
        else:
            # 为未知通道分配随机位置
            positions[ch] = np.random.randn(3) * 0.1
    
    return positions


if __name__ == "__main__":
    # 测试预处理器
    print("EEG预处理器测试")
    
    # 创建模拟数据
    n_channels, n_samples = 22, 1000
    sfreq = 250.0
    
    info = mne.create_info(
        ch_names=[f'C{i}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # 模拟EEG信号
    data = np.random.randn(n_channels, n_samples) * 1e-6
    raw = mne.io.RawArray(data, info)
    
    # 模拟事件
    events = np.array([[100, 0, 1], [300, 0, 2], [500, 0, 1], [700, 0, 2]])
    event_id = {'left': 1, 'right': 2}
    
    # 初始化预处理器
    preprocessor = EEGPreprocessor()
    
    # 运行预处理
    result = preprocessor.process_raw(raw, events, event_id)
    
    print(f"\n测试完成!")
    print(f"处理后试次形状: {result['trials'].shape}")
    if 'trials_filterbank' in result:
        print(f"FilterBank形状: {result['trials_filterbank'].shape}")
