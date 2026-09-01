@echo off
title MindFlow Audio Demo Launcher
cd /d "%~dp0"

echo ===================================================
echo       Launching MindFlow Audio Demo Suite
echo ===================================================

if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" launch_demo.py
) else (
    python launch_demo.py
)

pause
