#!/usr/bin/env bash
# Abadd0n — Linux / WSL entrypoint for environment setup.
# WSL2 + GPU: delegates to setup_wsl.sh (venv_wsl + training stack).
# Native Linux: venv/ + CPU PyTorch + requirements_wsl.txt (install CUDA torch first if you have a GPU).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  Abadd0n - Linux setup"
echo "========================================"

if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  echo "WSL detected — using linux/setup_wsl.sh (venv_wsl, GPU stack)."
  exec bash linux/setup_wsl.sh
fi

echo "Native Linux — creating venv/ under repo root ..."
python3 -m venv venv
# shellcheck source=/dev/null
source venv/bin/activate
python -m pip install -q -U pip wheel

echo "Installing CPU PyTorch wheels (for CUDA, install from pytorch.org first, then skip this step)."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing training stack (linux/requirements_wsl.txt) ..."
pip install -r linux/requirements_wsl.txt

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo "Verifying pre_unsloth inductor compat (no full Unsloth import) ..."
python -c "import pre_unsloth; pre_unsloth.before_import(); import torch._inductor.config as c; assert 'triton.enable_persistent_tma_matmul' in c._allowed_keys; print('pre_unsloth inductor compat OK')"

echo ""
echo "Done.  source venv/bin/activate"
echo "       python main.py   or   python cli.py"
echo "       python cli.py doctor   # diagnostics (no model load)"
echo "       python -m tests.test_tools --skip-network   # tools test"
echo "       python dataset_builder.py --generate --validate   # prepare dataset"
echo "       python unsloth_lora_train.py   # QLoRA  |  python export_ollama.py"
