#!/bin/bash
# CPU-only installation script
set -e

echo "Installing PyTorch CPU version..."
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo "Installing PyTorch Geometric CPU version..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "CPU installation complete!"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

