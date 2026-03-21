@echo off
REM Kills ALL python.exe processes on this machine — use only if you know no other Python work is running.
echo [ABADD0N] WARNING: This terminates every Python process (not only main.py).
set /p OK=Continue? (y/N): 
if /i not "%OK%"=="y" exit /b 0
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
echo Done.
pause
