@echo off
REM Abadd0n — Remove cached/build artifacts (keeps lora_model, outputs, dataset)
cd /d "%~dp0"

echo Cleaning cache artifacts ...
if exist "__pycache__" rd /s /q "__pycache__"
if exist "core\__pycache__" rd /s /q "core\__pycache__"
if exist "tests\__pycache__" rd /s /q "tests\__pycache__"
if exist "unsloth_compiled_cache" rd /s /q "unsloth_compiled_cache"
if exist "abadd0n_merged" rd /s /q "abadd0n_merged"
if exist "abadd0n_gguf" rd /s /q "abadd0n_gguf"
if exist "abadd0n_gguf_gguf" rd /s /q "abadd0n_gguf_gguf"
if exist "tests\_test_tools_run" rd /s /q "tests\_test_tools_run"
if exist "tests\_slash_run" rd /s /q "tests\_slash_run"
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

echo Done. lora_model, outputs, dataset.jsonl kept.
