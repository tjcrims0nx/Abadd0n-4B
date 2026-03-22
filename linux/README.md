# Abadd0n — Linux / WSL

> **⚠️ Untested:** Linux and WSL install/environment have not been verified. Use at your own risk.

## What gets installed

`linux/requirements_wsl.txt` installs the full stack:

- **Unsloth + TRL + HF** — QLoRA/SFT training, pinned transformers 4.57.6
- **CLI** — `rich`, `readchar` for slash menu, Tab actions, panels
- **Web search** — `googlesearch-python`, `beautifulsoup4`, `requests` for `/search` and interactive search (no API key)
- **Tools** — `/math` (stdlib only), file tools, web fetch, ClawHub skills

**You must install PyTorch separately.** Use [pytorch.org](https://pytorch.org) for CUDA builds; WSL typically uses cu128.

---

## WSL2 (recommended for GPU)

From the repo root in WSL:

```bash
chmod +x linux/setup_wsl.sh
bash linux/setup_wsl.sh
source venv_wsl/bin/activate
python main.py
```

**First-time PyTorch (if not installed):**

```bash
source venv_wsl/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
bash linux/setup_wsl.sh
```

Uses `venv_wsl`, Triton/Inductor caches in `~/.cache`. Use `python cli.py doctor` for diagnostics (no model load).

---

## Native Linux

```bash
chmod +x linux/setup.sh
./linux/setup.sh
source venv/bin/activate
```

`setup.sh` uses `venv/` and installs **CPU** PyTorch by default. For NVIDIA GPU, install CUDA PyTorch from [pytorch.org](https://pytorch.org) **before** or **after** running `setup.sh`, then `pip install -r linux/requirements_wsl.txt` if needed.

---

## After install

| Task | Command |
|------|---------|
| Chat | `python main.py` or `python cli.py` |
| Slash: math | `/math 2 + 3 * 4` |
| Slash: search | `/search python asyncio tutorial` |
| Doctor | `python cli.py doctor` |
| Tools test | `python -m tests.test_tools --skip-network` |
| Dataset | `python dataset_builder.py --generate --validate` |
| QLoRA | `python unsloth_lora_train.py` |
| Export | `python export_ollama.py` |

**Always activate the venv first** (`venv_wsl` on WSL, `venv` on native Linux).

---

## Files

| File | Purpose |
|------|---------|
| `setup_wsl.sh` | WSL2: `venv_wsl`, `~/.cache` caches, full deps |
| `setup.sh` | Native Linux: `venv`, CPU torch + `requirements_wsl.txt` |
| `requirements_wsl.txt` | Unsloth + TRL + pinned HF + rich/readchar + web search |
| `wsl_check.py` | Smoke test: torch CUDA + Unsloth import |
