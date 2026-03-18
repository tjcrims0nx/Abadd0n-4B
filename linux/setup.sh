#!/bin/bash
# Abadd0n Linux/WSL Setup Script

set -e

echo "========================================"
echo "  Abadd0n - Linux/WSL Setup"
echo "========================================"

# Detect if running in WSL
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    echo "  Detected: WSL (Windows Subsystem for Linux)"
    IS_WSL=true
else
    echo "  Detected: Native Linux"
    IS_WSL=false
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install CPU version (default)
echo ""
echo "[2/3] Installing dependencies (CPU)..."
pip install -r linux/requirements.txt

# Optional: GPU support
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "  NVIDIA GPU detected!"
    read -p "  Install PyTorch with CUDA support? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo "  ✓ CUDA support enabled"
    fi
elif [[ "$IS_WSL" == "true" ]]; then
    echo ""
    echo "  WSL detected - for GPU support, install NVIDIA drivers in Windows"
    echo "  and use WSL2 with CUDA support"
fi

echo ""
echo "[3/3] Verifying installation..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "To run Abadd0n:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "To train:"
echo "  python train.py --epochs 1000 --data data.txt"
echo ""
