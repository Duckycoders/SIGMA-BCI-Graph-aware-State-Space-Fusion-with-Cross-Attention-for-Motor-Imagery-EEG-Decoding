#!/usr/bin/env python3
"""
下载EEG Motor Movement/Imagery Dataset (PhysioNet EEGBCI)
使用MNE的EEGBCI fetcher进行数据获取
"""

import os
import argparse
from pathlib import Path
import mne
from mne.datasets import eegbci
import numpy as np


def download_eegmmi(data_dir: str, subjects: list = None, runs: list = None):
    """
    下载EEG-MMI数据集
    
    Args:
        data_dir: 数据保存目录
        subjects: 受试者列表，默认为[1, 2]（用于快速测试）
        runs: 任务run列表，默认为[3, 7, 11]（左手、右手、双脚想象）
    """
    if subjects is None:
        subjects = [1, 2]  # 默认下载前2个受试者用于测试
    if runs is None:
        runs = [3, 7, 11]  # 运动想象任务：左手(3,7,11), 右手(4,8,12), 脚(5,9,13), 拳头(6,10,14)
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载EEG-MMI数据集到: {data_dir}")
    print(f"受试者: {subjects}")
    print(f"任务runs: {runs}")
    
    for subject in subjects:
        print(f"\n下载受试者 {subject:03d}...")
        
        # 下载指定runs的数据
        raw_fnames = eegbci.load_data(subject, runs, path=data_dir, update_path=False)
        
        print(f"受试者 {subject:03d} 下载完成，文件数: {len(raw_fnames)}")
        for fname in raw_fnames:
            print(f"  - {fname}")
    
    print(f"\nEEG-MMI数据集下载完成！")
    
    # 创建数据集描述文件
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("EEG Motor Movement/Imagery Dataset (PhysioNet EEGBCI)\n")
        f.write("=" * 50 + "\n\n")
        f.write("数据来源: https://physionet.org/content/eegmmidb/\n")
        f.write("采样率: 160 Hz\n")
        f.write("通道数: 64 (10-20系统)\n")
        f.write("受试者数: 109\n\n")
        f.write("任务说明:\n")
        f.write("- Run 3,7,11: 左手握拳想象\n")
        f.write("- Run 4,8,12: 右手握拳想象\n") 
        f.write("- Run 5,9,13: 双脚想象\n")
        f.write("- Run 6,10,14: 双手握拳想象\n\n")
        f.write(f"已下载受试者: {subjects}\n")
        f.write(f"已下载runs: {runs}\n")


def check_download(data_dir: str):
    """检查下载的数据"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    edf_files = list(data_dir.glob("**/*.edf"))
    print(f"\n发现 {len(edf_files)} 个EDF文件:")
    for edf_file in sorted(edf_files):
        # 快速检查文件
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            print(f"  {edf_file.name}: {raw.info['nchan']}通道, {raw.info['sfreq']}Hz, {raw.times[-1]:.1f}s")
        except Exception as e:
            print(f"  {edf_file.name}: 读取错误 - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载EEG-MMI数据集")
    parser.add_argument("--data_dir", type=str, default="data/eegmmi", 
                       help="数据保存目录")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2],
                       help="要下载的受试者列表")
    parser.add_argument("--runs", type=int, nargs="+", default=[3, 7, 11],
                       help="要下载的run列表")
    parser.add_argument("--check", action="store_true",
                       help="检查已下载的数据")
    
    args = parser.parse_args()
    
    if args.check:
        check_download(args.data_dir)
    else:
        download_eegmmi(args.data_dir, args.subjects, args.runs)
        check_download(args.data_dir)
