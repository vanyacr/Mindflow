"""
MindFlow — Continuous Real-Time Streaming (Sliding Window)
==========================================================
Continuously streams live microphone audio using a 6-second sliding window
with a 1-second step/hop rate.

Features:
- Rolling circular buffer of 6.0 seconds @ 16kHz
- Evaluates emotion and continuous PHQ-8 stress every 1 second in real time
- In-place terminal refresh for a live monitoring dashboard experience
- Supports optional User Profile baseline calibration

Usage:
    python demo_streaming_mic.py [profiles/demo_user_profile.json]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import deque
import threading

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import soundfile as sf
import torch

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import UNIFIED_EMOTIONS
from inference.audio_interface import AudioInference
from inference.user_calibration import UserProfile, UserProfileCalibrator

SAMPLE_RATE = 16000
WINDOW_SECONDS = 6.0            # Model receptive field
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SECONDS = 1.0               # Updates every 1.0 second
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)


def render_bar(val: float, max_val: float = 1.0, length: int = 20) -> str:
    filled = int(round((val / max_val) * length))
    filled = min(max(filled, 0), length)
    bar = "#" * filled + "-" * (length - filled)
    return f"[{bar}] {val * 100:4.1f}%"


def render_stress(val: float) -> str:
    if val < 0.25:
        tag = "Normal / Low"
    elif val < 0.50:
        tag = "Mild"
    elif val < 0.75:
        tag = "Moderate"
    else:
        tag = "High / Severe"
    bar = "=" * int(val * 15) + "-" * (15 - int(val * 15))
    return f"[{bar}] {val:0.2f} ({tag})"


def stream_audio(profile_path: str | None = None):
    if not HAS_SOUNDDEVICE:
        print("[Error] sounddevice module is required for continuous mic streaming.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MindFlow Audio Model on {device.upper()}...")
    model = AudioInference(device=device)

    profile = None
    if profile_path and Path(profile_path).exists():
        profile = UserProfile.load(profile_path)
        print(f"[OK] Loaded User Baseline: {profile.user_id}")

    # Circular buffer
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    buffer_lock = threading.Lock()
    is_running = True

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            pass
        data = indata[:, 0]
        with buffer_lock:
            buffer = np.roll(buffer, -len(data))
            buffer[-len(data):] = data

    print("\n" + "=" * 70)
    print("      🧠 MINDFLOW REAL-TIME CONTINUOUS SLIDING WINDOW STREAMING      ")
    print("      Window: 6.0s Rolling Buffer | Step/Hop: 1.0s Refresh Rate      ")
    print("      Press Ctrl+C at any time to stop streaming                      ")
    print("=" * 70)
    print("Starting audio stream in 2 seconds...")
    time.sleep(2)

    temp_chunk_path = Path(__file__).resolve().parent / "demo_temp" / "stream_chunk.wav"
    temp_chunk_path.parent.mkdir(exist_ok=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=HOP_SAMPLES, callback=audio_callback):
        step = 0
        try:
            while is_running:
                time.sleep(HOP_SECONDS)
                step += 1

                with buffer_lock:
                    current_window = buffer.copy()

                # Check if buffer has audible sound
                rms_energy = np.sqrt(np.mean(current_window ** 2))
                if rms_energy < 0.003:
                    sys.stdout.write(f"\r[Time: {step*HOP_SECONDS:4.0f}s]  Listening for speech... (Silence detected)                      ")
                    sys.stdout.flush()
                    continue

                # Save temporary rolling chunk and predict
                sf.write(str(temp_chunk_path), current_window, SAMPLE_RATE)
                res = model.predict(temp_chunk_path, user_profile=profile)

                emo = res.get("calibrated_emotion", res["emotion"]).upper()
                stress = res.get("calibrated_stress", res["stress"])
                top_p = max(res.get("calibrated_emotion_probs", res["emotion_probs"]).values()) * 100

                # Formatted continuous real-time readout line
                meta_str = ""
                if "calibration_metadata" in res:
                    p_shift = res["calibration_metadata"]["pitch_delta_pct"]
                    meta_str = f" | Pitch Δ: {p_shift:+.0f}%"

                line = f"\r⏱️ {step*HOP_SECONDS:4.0f}s | Emotion: {emo:8s} ({top_p:4.1f}%) | Stress: {render_stress(stress)}{meta_str}"
                sys.stdout.write(line.ljust(85))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n[Stopped] Live audio streaming session finished.")


if __name__ == "__main__":
    prof = sys.argv[1] if len(sys.argv) > 1 else None
    stream_audio(prof)
