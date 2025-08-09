#!/usr/bin/env python3
"""
将下载的EEG数据转换为BIDS格式
支持EEG-MMI、BNCI2014-001、BNCI2014-004数据集
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
from mne.datasets import eegbci
import json


def create_events_from_annotations(raw, dataset_type="eegmmi"):
    """从MNE Raw对象的annotations创建events数组"""
    if dataset_type == "eegmmi":
        # EEG-MMI事件映射
        event_mapping = {
            'T0': 0,  # 实验开始
            'T1': 1,  # 左手握拳想象开始
            'T2': 2,  # 右手握拳想象开始
            'T3': 3,  # 双脚想象开始
            'T4': 4   # 双手握拳想象开始
        }
    elif dataset_type == "bnci2a":
        # BNCI 2a事件映射
        event_mapping = {
            '769': 1,  # 左手
            '770': 2,  # 右手
            '771': 3,  # 脚
            '772': 4,  # 舌头
            '276': 0,  # 眼睛闭合
            '277': 0,  # 眼睛睁开
            '1023': 0, # 拒绝试次
            '1072': 0, # 未知
        }
    elif dataset_type == "bnci2b":
        # BNCI 2b事件映射
        event_mapping = {
            '769': 1,  # 左手
            '770': 2,  # 右手
            '276': 0,  # 眼睛闭合
            '277': 0,  # 眼睛睁开
            '1023': 0, # 拒绝试次
            '1072': 0, # 未知
        }
    else:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    
    # 从annotations提取events
    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
        events, event_id = mne.events_from_annotations(raw, event_id=event_mapping)
    else:
        # 如果没有annotations，尝试从stim通道提取
        events = mne.find_events(raw)
        event_id = event_mapping
    
    return events, event_id


def convert_eegmmi_to_bids(raw_data_dir: str, bids_root: str):
    """将EEG-MMI数据转换为BIDS格式"""
    raw_data_dir = Path(raw_data_dir)
    bids_root = Path(bids_root)
    
    print(f"转换EEG-MMI数据: {raw_data_dir} -> {bids_root}")
    
    # 创建数据集描述
    make_dataset_description(
        path=bids_root,
        name="EEG Motor Movement/Imagery Dataset",
        authors=["Schalk, G.", "McFarland, D.J.", "Hinterberger, T.", "Birbaumer, N.", "Wolpaw, J.R."],
        data_license="PDDL",
        description="EEG recordings of motor movement and imagery tasks"
    )
    
    # 查找所有EDF文件
    edf_files = list(raw_data_dir.glob("**/*.edf"))
    print(f"发现 {len(edf_files)} 个EDF文件")
    
    for edf_file in sorted(edf_files):
        try:
            print(f"\n处理: {edf_file.name}")
            
            # 解析文件名获取受试者和run信息
            # 文件名格式: S001R03.edf (受试者001, run03)
            fname_parts = edf_file.stem  # 去掉.edf
            subject_id = fname_parts[1:4]  # S001 -> 001
            run_id = fname_parts[5:7]      # R03 -> 03
            
            # 读取原始数据
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # 设置通道类型
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch.startswith(('C', 'F', 'P', 'O', 'T'))})
            
            # 设置montage
            try:
                raw.set_montage('standard_1020', verbose=False)
            except:
                print(f"    警告: 无法设置montage")
            
            # 创建events
            events, event_id = create_events_from_annotations(raw, "eegmmi")
            
            # 确定任务类型
            task_mapping = {
                '03': 'motor-imagery',  # 左手想象
                '04': 'motor-imagery',  # 右手想象  
                '05': 'motor-imagery',  # 脚想象
                '06': 'motor-imagery',  # 双手想象
                '07': 'motor-imagery',  # 左手想象
                '08': 'motor-imagery',  # 右手想象
                '09': 'motor-imagery',  # 脚想象
                '10': 'motor-imagery',  # 双手想象
                '11': 'motor-imagery',  # 左手想象
                '12': 'motor-imagery',  # 右手想象
                '13': 'motor-imagery',  # 脚想象
                '14': 'motor-imagery',  # 双手想象
            }
            task = task_mapping.get(run_id, 'motor-imagery')
            
            # 创建BIDS路径
            bids_path = BIDSPath(
                subject=subject_id,
                task=task,
                run=run_id,
                datatype='eeg',
                root=bids_root
            )
            
            # 写入BIDS格式
            write_raw_bids(
                raw=raw,
                bids_path=bids_path,
                events_data=events,
                event_id=event_id,
                overwrite=True,
                verbose=False
            )
            
            print(f"    成功转换: subject-{subject_id}, run-{run_id}")
            
        except Exception as e:
            print(f"    错误: {e}")
            continue


def convert_bnci_to_bids(raw_data_dir: str, bids_root: str, dataset_type: str):
    """将BNCI数据转换为BIDS格式"""
    raw_data_dir = Path(raw_data_dir)
    bids_root = Path(bids_root)
    
    dataset_name = f"BNCI2014-{dataset_type[-2:]}"
    print(f"转换{dataset_name}数据: {raw_data_dir} -> {bids_root}")
    
    # 创建数据集描述
    make_dataset_description(
        path=bids_root,
        name=f"BCI Competition IV Dataset {dataset_type[-2:]}",
        authors=["Tangermann, M.", "Müller, K.-R.", "Aertsen, A.", "Birbaumer, N."],
        data_license="CC0",
        description=f"EEG motor imagery data from BCI Competition IV {dataset_type[-2:]}"
    )
    
    # 查找所有NPZ文件
    npz_files = list(raw_data_dir.glob("*.npz"))
    print(f"发现 {len(npz_files)} 个NPZ文件")
    
    for npz_file in sorted(npz_files):
        try:
            print(f"\n处理: {npz_file.name}")
            
            # 加载数据
            data_dict = np.load(npz_file)
            data = data_dict['data']
            events = data_dict['events'] 
            ch_names = data_dict['ch_names'].tolist()
            sfreq = float(data_dict['sfreq'])
            
            # 解析文件名
            # 格式: S01_session_T_run_0.npz
            parts = npz_file.stem.split('_')
            subject_id = parts[0][1:]  # S01 -> 01
            session = parts[1]
            run = parts[3]
            
            # 创建MNE Raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 设置montage（如果可能）
            try:
                if dataset_type == "2b":
                    # 2b只有3个双极通道，创建简单montage
                    montage_dict = {
                        'C3': [-0.5, 0, 0],
                        'Cz': [0, 0, 0], 
                        'C4': [0.5, 0, 0]
                    }
                    # 只为存在的通道设置位置
                    existing_montage = {ch: pos for ch, pos in montage_dict.items() if ch in ch_names}
                    if existing_montage:
                        montage = mne.channels.make_dig_montage(existing_montage)
                        raw.set_montage(montage, verbose=False)
                else:
                    raw.set_montage('standard_1020', verbose=False)
            except:
                print(f"    警告: 无法设置montage")
            
            # 创建BIDS路径
            bids_path = BIDSPath(
                subject=subject_id,
                session=session,
                task='motor-imagery',
                run=run,
                datatype='eeg',
                root=bids_root
            )
            
            # 创建事件映射
            if dataset_type == "2a":
                event_id = {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}
            else:  # 2b
                event_id = {'left_hand': 1, 'right_hand': 2}
            
            # 写入BIDS格式
            write_raw_bids(
                raw=raw,
                bids_path=bids_path,
                events_data=events,
                event_id=event_id,
                overwrite=True,
                verbose=False
            )
            
            print(f"    成功转换: subject-{subject_id}, session-{session}, run-{run}")
            
        except Exception as e:
            print(f"    错误: {e}")
            continue


def validate_bids(bids_root: str):
    """验证BIDS格式的正确性"""
    bids_root = Path(bids_root)
    
    print(f"\n验证BIDS目录: {bids_root}")
    
    # 检查必需文件
    required_files = ["dataset_description.json"]
    for req_file in required_files:
        if (bids_root / req_file).exists():
            print(f"  ✓ {req_file}")
        else:
            print(f"  ✗ 缺失: {req_file}")
    
    # 检查受试者目录
    subject_dirs = list(bids_root.glob("sub-*"))
    print(f"  发现 {len(subject_dirs)} 个受试者目录")
    
    for sub_dir in sorted(subject_dirs):
        print(f"    {sub_dir.name}:")
        eeg_dir = sub_dir / "eeg"
        if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*.eeg")) + list(eeg_dir.glob("*.vhdr")) + list(eeg_dir.glob("*.fif"))
            events_files = list(eeg_dir.glob("*events.tsv"))
            channels_files = list(eeg_dir.glob("*channels.tsv"))
            
            print(f"      EEG文件: {len(eeg_files)}")
            print(f"      Events文件: {len(events_files)}")
            print(f"      Channels文件: {len(channels_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换EEG数据为BIDS格式")
    parser.add_argument("--dataset", type=str, choices=["eegmmi", "bnci2a", "bnci2b"], 
                       required=True, help="数据集类型")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                       help="原始数据目录")
    parser.add_argument("--bids_root", type=str, required=True,
                       help="BIDS输出目录")
    parser.add_argument("--validate", action="store_true",
                       help="验证BIDS格式")
    
    args = parser.parse_args()
    
    if args.dataset == "eegmmi":
        convert_eegmmi_to_bids(args.raw_data_dir, args.bids_root)
    elif args.dataset == "bnci2a":
        convert_bnci_to_bids(args.raw_data_dir, args.bids_root, "2a")
    elif args.dataset == "bnci2b":
        convert_bnci_to_bids(args.raw_data_dir, args.bids_root, "2b")
    
    if args.validate:
        validate_bids(args.bids_root)
