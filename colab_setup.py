#!/usr/bin/env python3
"""
Google Colab环境设置脚本
用于在Colab中快速设置BCI项目环境
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ 成功: {description}")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ 失败: {description}")
        if result.stderr:
            print(result.stderr)
        return False
    return True

def setup_colab_environment():
    """设置Colab环境"""
    print("🚀 开始设置Google Colab环境...")
    
    # 1. 更新pip
    run_command("pip install --upgrade pip", "更新pip")
    
    # 2. 安装PyTorch (GPU版本)
    run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "安装PyTorch GPU版本"
    )
    
    # 3. 安装PyTorch Geometric
    run_command(
        "pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html",
        "安装PyTorch Geometric"
    )
    
    # 4. 安装Mamba-SSM依赖
    run_command("pip install packaging ninja", "安装Mamba依赖")
    
    # 5. 安装Mamba-SSM
    run_command("pip install mamba-ssm", "安装Mamba-SSM")
    
    # 6. 安装其他依赖
    run_command("pip install -r requirements.txt", "安装项目依赖")
    
    # 7. 检查CUDA是否可用
    import torch
    print(f"\n🔍 环境检查:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 8. 测试关键模块导入
    try:
        import torch_geometric
        print("✅ PyTorch Geometric导入成功")
    except ImportError:
        print("❌ PyTorch Geometric导入失败")
    
    try:
        from mamba_ssm import Mamba
        print("✅ Mamba-SSM导入成功")
    except ImportError:
        print("❌ Mamba-SSM导入失败")
    
    print("\n🎉 Colab环境设置完成！")

if __name__ == "__main__":
    setup_colab_environment()
