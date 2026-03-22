#!/usr/bin/env bash
# Abadd0n — create/update venv_wsl and install the Linux training stack.
# Run from WSL:  bash linux/setup_wsl.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d "venv_wsl/bin" ]; then
  echo "[setup_wsl] Creating venv_wsl …"
  python3 -m venv venv_wsl
fi

# shellcheck source=/dev/null
source venv_wsl/bin/activate

# Keep Triton / torch.compile caches on the Linux filesystem (not /mnt/c) — much faster.
export TRITON_CACHE_DIR="${HOME}/.cache/triton-abadd0n"
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch-inductor-abadd0n"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

python -m pip install -U pip wheel

if ! python -c "import torch" 2>/dev/null; then
  echo ""
  echo "[setup_wsl] PyTorch is not installed in this venv."
  echo "Install CUDA wheels first (pick the index that matches your GPU driver), e.g.:"
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
  echo "Then re-run:  bash linux/setup_wsl.sh"
  exit 1
fi

echo "[setup_wsl] Installing Unsloth + training deps …"
pip install -r linux/requirements_wsl.txt

echo "[setup_wsl] Verifying pre_unsloth + Unsloth import (first run can take 5–20+ min on /mnt/c — be patient) …"
PYTHONUNBUFFERED=1 python -u linux/wsl_check.py

echo ""
echo "[setup_wsl] Done. Activate with:  source venv_wsl/bin/activate"
echo "  Chat:     python main.py  |  python cli.py"
echo "  Slash:    /math <expr>  |  /search <query>"
echo "  Doctor:   python cli.py doctor"
echo "  Tools:    python -m tests.test_tools --skip-network"
echo "  Dataset:  python dataset_builder.py --generate --validate"
echo "  QLoRA:    python unsloth_lora_train.py  |  Export:  python export_ollama.py"
