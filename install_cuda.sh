#!/bin/bash
# CUDA 12.1 + GPU installation script
set -e

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "Installing PyTorch Geometric with CUDA support..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Attempting to install Mamba (optional)..."
pip install "mamba-ssm[causal-conv1d]" || echo "WARNING: Mamba installation failed. Will fallback to S4-only mode."

echo "Installation complete! GPU support enabled."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

