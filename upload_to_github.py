#!/usr/bin/env python3
"""
快速上传BCI项目到GitHub的辅助脚本
"""

import subprocess
import sys
import os
from pathlib import Path

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
        return True
    else:
        print(f"❌ 失败: {description}")
        if result.stderr:
            print(result.stderr)
        return False

def check_git_installed():
    """检查Git是否安装"""
    result = subprocess.run("git --version", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Git已安装: {result.stdout.strip()}")
        return True
    else:
        print("❌ Git未安装，请先安装Git")
        return False

def setup_git_repo():
    """设置Git仓库"""
    print("🚀 开始设置Git仓库...")
    
    # 检查是否已经是Git仓库
    if Path('.git').exists():
        print("📁 检测到现有Git仓库")
        choice = input("是否要重新初始化？(y/N): ").lower()
        if choice == 'y':
            run_command("rm -rf .git", "删除现有Git仓库")
            run_command("git init", "初始化Git仓库")
        else:
            print("保持现有Git仓库")
    else:
        run_command("git init", "初始化Git仓库")
    
    # 设置用户信息（如果未设置）
    result = subprocess.run("git config user.name", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        name = input("请输入您的Git用户名: ")
        run_command(f'git config user.name "{name}"', "设置用户名")
    
    result = subprocess.run("git config user.email", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        email = input("请输入您的Git邮箱: ")
        run_command(f'git config user.email "{email}"', "设置邮箱")

def add_and_commit():
    """添加文件并提交"""
    print("\n📝 添加文件到Git...")
    
    # 添加所有文件
    run_command("git add .", "添加所有文件")
    
    # 检查状态
    run_command("git status", "检查Git状态")
    
    # 提交
    commit_message = input("\n请输入提交信息 (默认: 'Initial BCI project commit'): ")
    if not commit_message:
        commit_message = "Initial BCI project commit"
    
    run_command(f'git commit -m "{commit_message}"', "提交更改")

def setup_remote_and_push():
    """设置远程仓库并推送"""
    print("\n🌐 设置远程GitHub仓库...")
    
    print("请在GitHub上创建一个新仓库，然后提供仓库地址")
    print("例如: https://github.com/YOUR_USERNAME/BCI-EEG-Decoding.git")
    
    repo_url = input("\n请输入GitHub仓库地址: ")
    if not repo_url:
        print("❌ 未提供仓库地址，跳过推送")
        return False
    
    # 添加远程仓库
    run_command(f"git remote add origin {repo_url}", "添加远程仓库")
    
    # 设置主分支
    run_command("git branch -M main", "设置主分支为main")
    
    # 推送到GitHub
    print("\n🚀 推送到GitHub...")
    success = run_command("git push -u origin main", "推送到GitHub")
    
    if success:
        print(f"\n🎉 成功！您的项目已上传到: {repo_url}")
        print("\n下一步:")
        print("1. 打开Google Colab: https://colab.research.google.com/")
        print("2. 上传colab_train.ipynb文件")
        print(f"3. 修改notebook中的仓库地址为: {repo_url}")
        print("4. 设置GPU运行时并运行所有cell")
        return True
    else:
        print("❌ 推送失败，请检查仓库地址和权限")
        return False

def main():
    """主函数"""
    print("🎯 BCI项目GitHub上传助手")
    print("=" * 50)
    
    # 检查Git
    if not check_git_installed():
        return
    
    # 检查项目文件
    required_files = ['models', 'configs', 'scripts', 'requirements.txt', 'colab_setup.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保在BCI项目根目录下运行此脚本")
        return
    
    print("✅ 项目文件检查通过")
    
    try:
        # 设置Git仓库
        setup_git_repo()
        
        # 添加和提交文件
        add_and_commit()
        
        # 推送到GitHub
        setup_remote_and_push()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
