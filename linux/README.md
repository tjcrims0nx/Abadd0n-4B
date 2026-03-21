# Abadd0n — Linux / WSL

## WSL2 (recommended for GPU)

From the repo root in WSL:

```bash
chmod +x linux/setup_wsl.sh
bash linux/setup_wsl.sh
source venv_wsl/bin/activate
python main.py
```

Uses `linux/requirements_wsl.txt` after PyTorch is present in the venv. See the main [README](../README.md) for CUDA notes.

## Native Linux

```bash
chmod +x linux/setup.sh
./linux/setup.sh
source venv/bin/activate
```

`setup.sh` installs **CPU** PyTorch by default. If you have an NVIDIA GPU, install the matching CUDA PyTorch build from [pytorch.org](https://pytorch.org) **before** `pip install -r linux/requirements_wsl.txt`, or install torch in the venv first and re-run the requirements line.

## Files

| File | Purpose |
|------|---------|
| `setup_wsl.sh` | WSL2: `venv_wsl`, caches on `~/.cache`, full training deps |
| `setup.sh` | Native Linux: `venv`, CPU torch + `requirements_wsl.txt` |
| `requirements_wsl.txt` | Unsloth + TRL + pinned HF stack (no torch — install torch separately) |
| `wsl_check.py` | Smoke test: torch CUDA + Unsloth import |
