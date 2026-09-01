"""
MindFlow — Master One-Click Demo Launcher
========================================
Run this script on ANY PC or Laptop to launch your MindFlow Audio Demo.
Automatically detects CUDA GPU or CPU, loads checkpoints, and provides
instant access to the Web Dashboard or Live Stream.

Usage:
    python launch_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

PIPELINE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def check_environment():
    print("=" * 65)
    print("             MINDFLOW AUDIO DEMO LAUNCHER             ")
    print("=" * 65)
    
    device = "CUDA GPU" if torch.cuda.is_available() else "CPU (Laptop Mode)"
    print(f"  • Compute Device:    {device}")
    
    s1_ckpt = PIPELINE_DIR / "checkpoints" / "stage1_best.pt"
    s2_ckpt = PIPELINE_DIR / "checkpoints" / "stage2_stress_best.pt"
    
    s1_status = "FOUND" if s1_ckpt.exists() else "MISSING"
    s2_status = "FOUND" if s2_ckpt.exists() else "MISSING"
    
    print(f"  • Stage 1 Weights:   {s1_status}")
    print(f"  • Stage 2 Weights:   {s2_status}")
    print("=" * 65)


def main():
    check_environment()

    while True:
        print("\nChoose how you want to run the MindFlow Demo:")
        print("  1. 🌐 Launch Interactive Web Dashboard (Browser UI with Charts & Gauge)")
        print("  2. ⚡ Launch Automated Real-Time Microphone Stream (Terminal Live Stream)")
        print("  3. 🎙️ Launch Interactive Menu (Single Clips, WAV Test, Calibration)")
        print("  4. 📁 Test a Single WAV Audio File")
        print("  5. ❌ Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            print("\n>>> Launching Web Dashboard... Open http://127.0.0.1:7860 in your browser.")
            cmd = [PYTHON_EXE, str(PIPELINE_DIR / "app_web_dashboard.py")]
            subprocess.run(cmd)

        elif choice == "2":
            print("\n>>> Launching Automated Real-Time Stream (Auto-Calibrates then Streams Live)...")
            cmd = [PYTHON_EXE, str(PIPELINE_DIR / "run_realtime_stream.py")]
            subprocess.run(cmd)

        elif choice == "3":
            print("\n>>> Launching Interactive Microphone Suite...")
            cmd = [PYTHON_EXE, str(PIPELINE_DIR / "demo_live_mic.py")]
            subprocess.run(cmd)

        elif choice == "4":
            default_sample = PIPELINE_DIR / "demo_samples" / "sample_happy.wav"
            wav_path = input(f"Enter WAV file path (default: {default_sample.name}): ").strip().strip('"').strip("'")
            if not wav_path:
                wav_path = str(default_sample)
            cmd = [PYTHON_EXE, str(PIPELINE_DIR / "demo_quick_test.py"), wav_path]
            subprocess.run(cmd)

        elif choice == "5":
            print("\nExiting launcher. Good luck with your review!")
            break
        else:
            print("Invalid choice, please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()
