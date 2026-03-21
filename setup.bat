@echo off
setlocal EnableExtensions
REM Abadd0n — Windows setup (CUDA 12.1 stack from repo root requirements.txt)
cd /d "%~dp0"

echo ========================================
echo   Abadd0n - Windows setup
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not on PATH. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

python --version
echo.

set "VENV=%~dp0venv_win"
if not exist "%VENV%\Scripts\python.exe" (
    echo [1/5] Creating venv_win ...
    python -m venv "%VENV%"
) else (
    echo [1/5] venv_win already exists — skipping create
)

call "%VENV%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate venv_win
    pause
    exit /b 1
)

echo [2/5] Upgrading pip ...
python -m pip install -q --upgrade pip wheel

echo [3/5] Installing dependencies ^(root requirements.txt, CUDA 12.1 index^) ...
pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

echo [4/5] Verifying PyTorch ...
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ERROR: PyTorch verification failed
    pause
    exit /b 1
)

echo [5/5] Verifying pre_unsloth inductor compat ^(no full Unsloth import^) ...
python -c "import pre_unsloth; pre_unsloth.before_import(); import torch._inductor.config as _ic; assert 'triton.enable_persistent_tma_matmul' in _ic._allowed_keys; print('  pre_unsloth: inductor compat OK')"
if errorlevel 1 (
    echo ERROR: pre_unsloth inductor registration failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup complete
echo ========================================
echo   Activate:  venv_win\Scripts\activate
echo   Chat:     python main.py
echo   QLoRA:    python unsloth_lora_train.py
echo   Char LM:  python train.py --help
echo ========================================
pause
endlocal
