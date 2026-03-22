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
    echo [1/6] Creating venv_win ...
    python -m venv "%VENV%"
) else (
    echo [1/6] venv_win already exists — skipping create
)

call "%VENV%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate venv_win
    pause
    exit /b 1
)

echo [2/6] Upgrading pip ...
python -m pip install -q --upgrade pip wheel

echo [3/6] Installing dependencies ^(root requirements.txt, CUDA 12.1 index^) ...
echo   NOTE: Includes PyTorch CUDA, Unsloth, etc. — may take several minutes.
echo.
if not exist "%~dp0requirements.txt" (
    echo ERROR: requirements.txt not found. Run setup.bat from repo root.
    pause
    exit /b 1
)
pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)
pip check
if errorlevel 1 (
    echo WARNING: pip check reported dependency conflicts — see above
)

echo [4/6] Verifying PyTorch ...
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ERROR: PyTorch verification failed
    pause
    exit /b 1
)

echo [5/6] Verifying pre_unsloth inductor compat ...
python -c "import pre_unsloth; pre_unsloth.before_import(); import torch._inductor.config as _ic; assert 'triton.enable_persistent_tma_matmul' in _ic._allowed_keys; print('  pre_unsloth: inductor compat OK')"
if errorlevel 1 (
    echo ERROR: pre_unsloth inductor registration failed
    pause
    exit /b 1
)

echo.
echo [6/6] Creating dataset.jsonl ^(for QLoRA training^) ...
if not exist "%~dp0dataset.jsonl" (
    python dataset_builder.py --generate --validate
) else (
    echo   dataset.jsonl exists — skipping
)

echo.
echo ========================================
echo   Setup complete
echo ========================================
echo   ALWAYS activate:  venv_win\Scripts\activate
echo   Chat:     python main.py   or  python cli.py
echo   Slash:    /math ^<expr^>   /search ^<query^>   Right arrow = accept suggestion
echo   Doctor:   python cli.py doctor
echo   Tools:    python -m tests.test_tools --skip-network
echo   Dataset:  python dataset_builder.py --generate --validate
echo   QLoRA:    python unsloth_lora_train.py
echo   HF:       python export_hf.py USERNAME/repo --lora-only
echo ========================================
pause
endlocal
