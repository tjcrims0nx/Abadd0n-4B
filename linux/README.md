# Abadd0n - Linux/WSL Setup

## Quick Start

```bash
# Run the setup script
chmod +x linux/setup.sh
./linux/setup.sh

# Activate virtual environment
source venv/bin/activate

# Run Abadd0n
python main.py

# Or train from scratch
python train.py --epochs 1000 --data data.txt
```

## Requirements

- Python 3.9+
- 4GB+ RAM (8GB recommended for training)
- For GPU training: NVIDIA GPU with CUDA

## Virtual Environment

The setup script creates a `venv/` directory with all dependencies.

To activate manually:
```bash
source venv/bin/activate
```

To deactivate:
```bash
deactivate
```

## GPU Support

### Native Linux with NVIDIA GPU
The setup script will detect your GPU and offer to install CUDA-enabled PyTorch.

### WSL2 with NVIDIA GPU
1. Install NVIDIA drivers on Windows
2. Install WSL2 with Ubuntu
3. Install CUDA for WSL:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-pending
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
   sudo apt-get update
   sudo apt-get install cuda
   ```

Then run setup.sh and choose CUDA support.

## File Descriptions

- `requirements.txt` - Core dependencies only
- `requirements_full.txt` - Full dependencies for QLoRA/DPO training
- `setup.sh` - Automated setup script
