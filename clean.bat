@echo off
REM Abadd0n — Remove cached/build artifacts (keeps lora_model, outputs, dataset)
cd /d "%~dp0"

echo Cleaning cache artifacts ...
if exist "__pycache__" rd /s /q "__pycache__"
if exist "core\__pycache__" rd /s /q "core\__pycache__"
if exist "tests\__pycache__" rd /s /q "tests\__pycache__"
if exist "unsloth_compiled_cache" rd /s /q "unsloth_compiled_cache"
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

echo Done. lora_model, outputs, dataset.jsonl kept.
