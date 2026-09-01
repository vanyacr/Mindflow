"""
MindFlow — Seamless Real-Time Live Streaming with Auto-Calibration
==================================================================
One-Click Workflow:
1. Automatically records 6 seconds of neutral speech to calibrate your personal voice.
2. Immediately transitions into continuous live real-time sliding window streaming.
3. Continuously predicts Emotion, Stress (PHQ-8), and Acoustic Deltas live as you speak.

Usage:
    python run_realtime_stream.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
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
CALIBRATION_SECONDS = 6.0       # 6s neutral baseline speech
WINDOW_SECONDS = 6.0            # 6.0s rolling buffer for model
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SECONDS = 1.0               # Updates every 1.0 second
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

TEMP_DIR = Path(__file__).resolve().parent / "demo_temp"
TEMP_DIR.mkdir(exist_ok=True)


def render_stress_tag(val: float) -> str:
    if val < 0.25:
        tag = "Normal"
    elif val < 0.50:
        tag = "Mild Stress"
    elif val < 0.75:
        tag = "Moderate"
    else:
        tag = "High / Severe"
    bar = "=" * int(val * 12) + "-" * (12 - int(val * 12))
    return f"[{bar}] {val:0.2f} ({tag})"


def main():
    if not HAS_SOUNDDEVICE:
        print("[Error] 'sounddevice' module is required. Please ensure microphone access is enabled.")
        return

    print("=" * 70)
    print("      🧠 MINDFLOW REAL-TIME AUDIO STREAMING & MONITORING ENGINE      ")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Loading MindFlow Audio Model on {device.upper()}...")
    model = AudioInference(device=device)
    print("[1/3] Model ready!\n")

    # =========================================================================
    # STEP 1: AUTOMATIC ONE-TIME BASELINE CALIBRATION
    # =========================================================================
    print("=" * 70)
    print(" STEP 1: PERSONAL VOICE CALIBRATION (6 Seconds)                      ")
    print(" Please speak naturally and calmly in your normal, neutral tone.     ")
    print(" (e.g. read: 'Today is a normal day. I am testing MindFlow voice.')  ")
    print("=" * 70)

    for i in range(3, 0, -1):
        print(f"Starting calibration in {i}...", end="\r")
        time.sleep(1)
    print(">>> 🔴 CALIBRATING NOW: Speak in your normal neutral voice...        ")

    calib_audio = sd.rec(int(CALIBRATION_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print(">>> [OK] Calibration recording complete! Processing baseline...\n")

    calib_path = TEMP_DIR / "user_baseline.wav"
    sf.write(str(calib_path), calib_audio.flatten(), SAMPLE_RATE)
    profile = model.register_user_baseline(calib_path, user_id="live_user")

    print(f"✅ Baseline Calibrated:")
    print(f"   • Base Pitch (F0): {profile.base_pitch_mean:.1f} Hz")
    print(f"   • Base Energy:     {profile.base_energy_rms:.4f}")
    print(f"   • Base Pause Rate: {profile.base_pause_ratio * 100:.1f}%\n")

    # =========================================================================
    # STEP 2: CONTINUOUS REAL-TIME SLIDING WINDOW STREAMING
    # =========================================================================
    print("=" * 70)
    print(" STEP 2: CONTINUOUS REAL-TIME MONITORING ACTIVATED (SLIDING WINDOW)  ")
    print(f" Window: {WINDOW_SECONDS}s Rolling Buffer | Update Rate: Every {HOP_SECONDS}s               ")
    print(" Speak into the mic (Happy, Angry, Sad, Stressed, etc.)              ")
    print(" Press Ctrl+C to stop.                                               ")
    print("=" * 70)

    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    buffer_lock = threading.Lock()
    temp_chunk_path = TEMP_DIR / "live_stream_chunk.wav"

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        data = indata[:, 0]
        with buffer_lock:
            buffer = np.roll(buffer, -len(data))
            buffer[-len(data):] = data

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=HOP_SAMPLES, callback=audio_callback):
        step = 0
        try:
            while True:
                time.sleep(HOP_SECONDS)
                step += 1

                with buffer_lock:
                    current_window = buffer.copy()

                rms_energy = np.sqrt(np.mean(current_window ** 2))
                if rms_energy < 0.003:
                    sys.stdout.write(f"\r[⏱️ {step*HOP_SECONDS:4.0f}s]  Listening for speech... (Silence)                                  ")
                    sys.stdout.flush()
                    continue

                sf.write(str(temp_chunk_path), current_window, SAMPLE_RATE)
                res = model.predict(temp_chunk_path, user_profile=profile)

                emo = res.get("calibrated_emotion", res["emotion"]).upper()
                stress = res.get("calibrated_stress", res["stress"])
                top_p = max(res.get("calibrated_emotion_probs", res["emotion_probs"]).values()) * 100

                meta_str = ""
                if "calibration_metadata" in res:
                    p_shift = res["calibration_metadata"]["pitch_delta_pct"]
                    e_shift = res["calibration_metadata"]["energy_delta_db"]
                    meta_str = f" | Pitch Δ: {p_shift:+3.0f}% | Energy Δ: {e_shift:+3.1f}dB"

                line = f"\r[⏱️ {step*HOP_SECONDS:4.0f}s] Emotion: {emo:8s} ({top_p:4.1f}%) | Stress: {render_stress_tag(stress)}{meta_str}"
                sys.stdout.write(line.ljust(95))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n[Finished] Live streaming session ended.")


if __name__ == "__main__":
    main()
