#!/usr/bin/env python3
"""
从多数据集原始NPZ文件聚合生成训练所需的 trials/labels NPZ 文件
支持BNCI2014-001 (2a), BNCI2014-004 (2b), EEG-MMI (PhysionetMI)

输入NPZ字段: data (n_channels, n_samples), events (n_events, 3), ch_names, sfreq
输出NPZ字段: trials (n_trials, n_channels, n_samples), labels (n_trials,)
"""

import os
import sys
from pathlib import Path
# 将项目根目录加入sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import mne
from typing import Dict, List

from eeg.preprocess import EEGPreprocessor


# 数据集事件码映射
BNCI2A_TRIGGER_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}  # 左手、右手、脚、舌头
BNCI2B_TRIGGER_TO_CLASS = {1: 0, 2: 1}  # 左手、右手
EEGMMI_TRIGGER_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}  # 左拳、右拳、双拳、双脚


def collect_bnci2a_trials(src_dir: str,
                          subjects: list,
                          trial_window=(1.0, 4.0),
                          sfreq_default: float = 250.0,
                          apply_filterbank: bool = False):
    src_path = Path(src_dir)
    all_trials = []
    all_labels = []

    preproc = EEGPreprocessor(
        sfreq=sfreq_default,
        low_freq=0.5,
        high_freq=100.0,
        notch_freq=[50.0],
        trial_window=trial_window,
        standardize=True,
    )

    # 使用1..4事件码
    event_id = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}

    for subj in subjects:
        pattern = f"S{subj:02d}_*.npz"
        npz_files = sorted(src_path.glob(pattern))
        if not npz_files:
            print(f"受试者{subj:02d}未找到NPZ文件，跳过")
            continue

        print(f"\n处理受试者 {subj:02d}，文件数: {len(npz_files)}")
        for f in npz_files:
            try:
                data_dict = np.load(f, allow_pickle=True)
                data = data_dict['data']  # (n_channels, n_samples)
                events = data_dict['events']  # (n_events, 3)
                ch_names = data_dict['ch_names'].tolist()
                sfreq = float(data_dict['sfreq']) if 'sfreq' in data_dict else sfreq_default

                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                raw = mne.io.RawArray(data, info, verbose=False)

                # 预处理 + 试次提取
                trials, labels = preproc.extract_trials(raw, events, event_id)

                # 标签从触发码映射为0..3
                labels = np.vectorize(BNCI2A_TRIGGER_TO_CLASS.get)(labels)

                # 标准化
                trials = preproc.apply_standardization(trials)

                all_trials.append(trials)
                all_labels.append(labels)

                print(f"  {f.name}: trials={trials.shape}, labels分布={np.unique(labels, return_counts=True)}")
            except Exception as e:
                print(f"  处理失败 {f.name}: {e}")

    if not all_trials:
        raise RuntimeError("未收集到任何试次，请检查源目录与受试者列表")

    trials = np.concatenate(all_trials, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return trials, labels


def main():
    parser = argparse.ArgumentParser(description="准备BNCI2a训练数据 (trials/labels)")
    parser.add_argument('--src_dir', type=str, required=True, help='源目录，例如 data/bnci/bnci2014_001')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1], help='受试者列表')
    parser.add_argument('--out', type=str, required=True, help='输出路径，例如 data/processed/bnci2a_s01.npz')
    parser.add_argument('--tmin', type=float, default=1.0)
    parser.add_argument('--tmax', type=float, default=4.0)

    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    trials, labels = collect_bnci2a_trials(
        src_dir=args.src_dir,
        subjects=args.subjects,
        trial_window=(args.tmin, args.tmax),
    )

    np.savez(args.out, trials=trials, labels=labels)
    print(f"\n已保存: {args.out}")
    print(f"trials: {trials.shape}, labels: {labels.shape}, 类别: {np.unique(labels)}")


def prepare_multi_dataset(bnci2a_dir: str, bnci2b_dir: str, eegmmi_dir: str, 
                         output_dir: str, subjects_per_dataset: Dict[str, List[int]]):
    """
    准备多数据集联合训练数据
    
    Args:
        bnci2a_dir: BNCI 2a数据目录
        bnci2b_dir: BNCI 2b数据目录  
        eegmmi_dir: EEG-MMI数据目录
        output_dir: 输出目录
        subjects_per_dataset: 每个数据集的被试列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_datasets = {}
    
    # 处理BNCI 2a数据集
    if bnci2a_dir and Path(bnci2a_dir).exists():
        print("处理BNCI 2a数据集...")
        trials, labels = collect_bnci2a_trials(
            bnci2a_dir, 
            subjects_per_dataset.get('bnci2a', [1, 2, 3]),
            trial_window=(1.0, 4.0)
        )
        all_datasets['bnci2a'] = {'trials': trials, 'labels': labels}
        print(f"BNCI 2a: {trials.shape}, 标签分布: {np.bincount(labels)}")
    
    # 处理BNCI 2b数据集 (需要适配3通道到22通道)
    if bnci2b_dir and Path(bnci2b_dir).exists():
        print("\n处理BNCI 2b数据集...")
        # 由于BNCI 2b只有3通道，我们需要特殊处理
        # 这里暂时跳过，专注于BNCI 2a + EEG-MMI
        print("BNCI 2b处理暂时跳过（通道数不匹配）")
    
    # 处理EEG-MMI数据集 (需要适配64通道到22通道)
    if eegmmi_dir and Path(eegmmi_dir).exists():
        print("\n处理EEG-MMI数据集...")
        # EEG-MMI有64通道，需要选择与BNCI 2a相似的通道
        # 这里暂时跳过，专注于BNCI 2a多被试
        print("EEG-MMI处理暂时跳过（通道数不匹配）")
    
    # 保存多数据集文件
    for dataset_name, data in all_datasets.items():
        output_file = output_path / f"{dataset_name}_multi_subjects.npz"
        np.savez(output_file, 
                trials=data['trials'], 
                labels=data['labels'])
        print(f"\n保存{dataset_name}: {output_file}")
        print(f"  形状: {data['trials'].shape}")
        print(f"  标签分布: {np.bincount(data['labels'])}")


def main_multi():
    """多数据集处理主函数"""
    parser = argparse.ArgumentParser(description="多数据集预处理")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                       help="处理模式：single=单被试，multi=多数据集")
    parser.add_argument("--src", required=True, help="源数据目录")
    parser.add_argument("--out", required=True, help="输出文件路径")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1],
                       help="要处理的被试列表")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # 调用原有单被试处理，但需要适配参数
        import sys
        # 重新构造sys.argv来调用原main函数
        original_argv = sys.argv
        sys.argv = ['prepare_bnci2a_npz.py', '--src_dir', args.src, '--out', args.out, 
                   '--subjects'] + [str(s) for s in args.subjects]
        main()
        sys.argv = original_argv
    else:
        # 多数据集处理
        subjects_config = {
            'bnci2a': args.subjects,
            'bnci2b': args.subjects,
            'eegmmi': args.subjects
        }
        prepare_multi_dataset(
            bnci2a_dir=args.src + "/bnci2014_001",
            bnci2b_dir=args.src + "/bnci2014_004", 
            eegmmi_dir=args.src + "/physionet_mi",
            output_dir=Path(args.out).parent,
            subjects_per_dataset=subjects_config
        )


if __name__ == '__main__':
    main_multi()
