#!/usr/bin/env python3
"""
下载BCI Competition IV Dataset 2a/2b和EEG-MMI数据集
使用MOABB库进行数据获取
"""

import os
import argparse
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2014_001, BNCI2014_004, PhysionetMI
import mne


def download_bnci_2a(data_dir: str, subjects: list = None):
    """
    下载BNCI2014-001 (BCI Competition IV 2a)
    
    Args:
        data_dir: 数据保存目录
        subjects: 受试者列表，默认为[1, 2]
    """
    if subjects is None:
        subjects = [1, 2]  # 默认下载前2个受试者
    
    data_dir = Path(data_dir) / "bnci2014_001"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载BNCI2014-001 (2a)数据集到: {data_dir}")
    print(f"受试者: {subjects}")
    
    dataset = BNCI2014_001()
    
    # BNCI2a触发码映射
    event_mapping = {'769': 769, '770': 770, '771': 771, '772': 772}
    
    for subject in subjects:
        print(f"\n下载受试者 {subject:02d}...")
        
        # 获取受试者数据
        subject_data = dataset.get_data(subjects=[subject])
        
        # 保存数据信息
        for session_name, session_data in subject_data[subject].items():
            print(f"  会话: {session_name}")
            for run_name, raw in session_data.items():
                try:
                    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, ecg=False, emg=False, misc=False)
                    data = raw.get_data(picks=eeg_picks)
                    ch_names = np.array([raw.ch_names[i] for i in eeg_picks], dtype=object)
                    data_shape = data.shape
                except Exception:
                    data = raw.get_data(picks='eeg')
                    ch_names = np.array([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg'], dtype=object)
                    data_shape = data.shape
                print(f"    Run: {run_name}, 形状: {data_shape}")
                
                # 从annotations提取events，映射到标准触发码
                try:
                    events, _ = mne.events_from_annotations(raw, event_id=event_mapping)
                except Exception:
                    # 备用：尝试从stim通道提取
                    events = mne.find_events(raw, verbose=False)
                
                # 保存为numpy格式便于后续处理
                save_path = data_dir / f"S{subject:02d}_{session_name}_{run_name}.npz"
                np.savez(save_path, 
                        data=data,
                        events=events,
                        ch_names=ch_names,
                        sfreq=float(raw.info['sfreq']))
                print(f"    保存到: {save_path}")
    
    # 创建数据集描述
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("BNCI2014-001 (BCI Competition IV Dataset 2a)\n")
        f.write("=" * 50 + "\n\n")
        f.write("任务: 4类运动想象 (左手、右手、脚、舌头)\n")
        f.write("采样率: 250 Hz\n")
        f.write("通道数: 22 (EEG)\n")
        f.write("受试者数: 9\n")
        f.write("会话数: 2 (训练+测试)\n")
        f.write("每类试次: 72 (训练) + 72 (测试)\n\n")
        f.write(f"已下载受试者: {subjects}\n")


def download_bnci_2b(data_dir: str, subjects: list = None):
    """
    下载BNCI2014-004 (BCI Competition IV 2b)
    
    Args:
        data_dir: 数据保存目录  
        subjects: 受试者列表，默认为[1, 2]
    """
    if subjects is None:
        subjects = [1, 2]
    
    data_dir = Path(data_dir) / "bnci2014_004"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载BNCI2014-004 (2b)数据集到: {data_dir}")
    print(f"受试者: {subjects}")
    
    dataset = BNCI2014_004()
    
    for subject in subjects:
        print(f"\n下载受试者 {subject:02d}...")
        
        # 获取受试者数据
        subject_data = dataset.get_data(subjects=[subject])
        
        # 保存数据信息
        for session_name, session_data in subject_data[subject].items():
            print(f"  会话: {session_name}")
            for run_name, raw in session_data.items():
                try:
                    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, ecg=False, emg=False, misc=False)
                    # 仅保留C3/Cz/C4如果存在
                    pick_names = [raw.ch_names[i] for i in eeg_picks]
                    wanted = [i for i, ch in zip(eeg_picks, pick_names) if ch in ['C3', 'Cz', 'C4']]
                    use_picks = wanted if wanted else eeg_picks
                    data = raw.get_data(picks=use_picks)
                    ch_names = np.array([raw.ch_names[i] for i in use_picks], dtype=object)
                    data_shape = data.shape
                except Exception:
                    data = raw.get_data(picks='eeg')
                    ch_names = np.array([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg'], dtype=object)
                    data_shape = data.shape
                print(f"    Run: {run_name}, 形状: {data_shape}")
                
                # 提取events
                try:
                    events, _ = mne.events_from_annotations(raw)
                except Exception:
                    events = mne.find_events(raw, verbose=False)
                
                # 保存为numpy格式
                save_path = data_dir / f"S{subject:02d}_{session_name}_{run_name}.npz"
                np.savez(save_path,
                        data=data,
                        events=events, 
                        ch_names=ch_names,
                        sfreq=float(raw.info['sfreq']))
                print(f"    保存到: {save_path}")
    
    # 创建数据集描述
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("BNCI2014-004 (BCI Competition IV Dataset 2b)\n")
        f.write("=" * 50 + "\n\n")
        f.write("任务: 2类运动想象 (左手、右手)\n")
        f.write("采样率: 250 Hz\n")
        f.write("通道数: 3 (双极导联: C3, Cz, C4)\n")
        f.write("受试者数: 9\n")
        f.write("会话数: 5\n")
        f.write("每会话试次: 120 (左手60 + 右手60)\n\n")
        f.write(f"已下载受试者: {subjects}\n")


def download_eegmmi(data_dir: str, subjects: list = None):
    """
    下载EEG-MMI (PhysionetMI)数据集
    
    Args:
        data_dir: 数据保存目录
        subjects: 受试者列表，默认为[1, 2, 3, 4, 5]
    """
    if subjects is None:
        subjects = [1, 2, 3, 4, 5]  # 默认下载前5个受试者
    
    data_dir = Path(data_dir) / "physionet_mi"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载EEG-MMI (PhysionetMI)数据集到: {data_dir}")
    print(f"受试者: {subjects}")
    
    dataset = PhysionetMI()
    
    for subject in subjects:
        print(f"\n下载受试者 {subject:03d}...")
        
        try:
            # 获取受试者数据
            subject_data = dataset.get_data(subjects=[subject])
            
            # 保存数据信息
            for session_name, session_data in subject_data[subject].items():
                print(f"  会话: {session_name}")
                for run_name, raw in session_data.items():
                    try:
                        eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, ecg=False, emg=False, misc=False)
                        data = raw.get_data(picks=eeg_picks)
                        ch_names = np.array([raw.ch_names[i] for i in eeg_picks], dtype=object)
                        data_shape = data.shape
                    except Exception:
                        data = raw.get_data(picks='eeg')
                        ch_names = np.array([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg'], dtype=object)
                        data_shape = data.shape
                    print(f"    Run: {run_name}, 形状: {data_shape}")
                    
                    # 提取events
                    try:
                        events, _ = mne.events_from_annotations(raw)
                    except Exception:
                        events = mne.find_events(raw, verbose=False)
                    
                    # 保存为numpy格式
                    save_path = data_dir / f"S{subject:03d}_{session_name}_{run_name}.npz"
                    np.savez(save_path,
                            data=data,
                            events=events,
                            ch_names=ch_names,
                            sfreq=float(raw.info['sfreq']))
                    print(f"    保存到: {save_path}")
        
        except Exception as e:
            print(f"  下载受试者 {subject:03d} 失败: {e}")
            continue
    
    # 创建数据集描述
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("EEG-MMI (PhysionetMI) Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write("任务: 4类运动想象 (左拳、右拳、双拳、双脚)\n")
        f.write("采样率: 160 Hz\n")
        f.write("通道数: 64 (EEG)\n")
        f.write("受试者数: 109\n")
        f.write("会话数: 多个\n")
        f.write("任务类型: 运动想象和运动执行\n\n")
        f.write(f"已下载受试者: {subjects}\n")


def check_bnci_data(data_dir: str):
    """检查下载的BNCI数据"""
    data_dir = Path(data_dir)
    
    for dataset_name in ["bnci2014_001", "bnci2014_004"]:
        dataset_dir = data_dir / dataset_name
        if dataset_dir.exists():
            print(f"\n{dataset_name}:")
            npz_files = list(dataset_dir.glob("*.npz"))
            print(f"  发现 {len(npz_files)} 个数据文件")
            
            for npz_file in sorted(npz_files):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    print(f"  {npz_file.name}: {data['data'].shape}, {float(data['sfreq'])}Hz")
                except Exception as e:
                    print(f"  {npz_file.name}: 读取错误 - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载BNCI数据集")
    parser.add_argument("--data_dir", type=str, default="data/bnci",
                       help="数据保存目录")
    parser.add_argument("--dataset", type=str, choices=["2a", "2b", "eegmmi", "both", "all"], 
                       default="both", help="要下载的数据集")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2],
                       help="要下载的受试者列表")
    parser.add_argument("--check", action="store_true",
                       help="检查已下载的数据")
    
    args = parser.parse_args()
    
    if args.check:
        check_bnci_data(args.data_dir)
    else:
        if args.dataset in ["2a", "both", "all"]:
            download_bnci_2a(args.data_dir, args.subjects)
        
        if args.dataset in ["2b", "both", "all"]:
            download_bnci_2b(args.data_dir, args.subjects)
            
        if args.dataset in ["eegmmi", "all"]:
            download_eegmmi(args.data_dir, args.subjects)
            
        check_bnci_data(args.data_dir)
