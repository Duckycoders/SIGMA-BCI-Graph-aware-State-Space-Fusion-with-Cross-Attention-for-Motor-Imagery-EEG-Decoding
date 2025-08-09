#!/usr/bin/env python3
"""
从 scripts/download_bnci.py 下载的BNCI2014-001 (2a) 原始NPZ文件
聚合生成训练所需的 trials/labels NPZ 文件。

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

from eeg.preprocess import EEGPreprocessor


# 当前保存的events第三列为1,2,3,4
BNCI2A_TRIGGER_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}


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


if __name__ == '__main__':
    main()
