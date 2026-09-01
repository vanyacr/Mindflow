"""
MindFlow — Live Audio & Microphone Real-Time Demo
=================================================
Interactive demo tool to test:
1. Live microphone speech recording (or WAV file testing).
2. One-time user baseline calibration (neutral voice profile).
3. Real-time emotion classification (7 classes), continuous PHQ-8 stress score,
   and personalized acoustic delta biomarker feedback.

Supports both GPU (CUDA) and CPU (Laptop without dedicated GPU).

Usage:
    python demo_live_mic.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for UTF-8
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

TEMP_DIR = Path(__file__).resolve().parent / "demo_temp"
TEMP_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000


def render_bar(val: float, max_val: float = 1.0, length: int = 25) -> str:
    """Renders a visual ASCII progress bar."""
    filled = int(round((val / max_val) * length))
    filled = min(max(filled, 0), length)
    bar = "#" * filled + "-" * (length - filled)
    return f"[{bar}] {val * 100:5.1f}%"


def render_stress_bar(val: float, length: int = 25) -> str:
    filled = int(round(val * length))
    filled = min(max(filled, 0), length)
    if val < 0.25:
        tag = "Minimal / Normal"
    elif val < 0.50:
        tag = "Mild Stress"
    elif val < 0.75:
        tag = "Moderate (Elevated)"
    else:
        tag = "Severe (High Stress)"
    bar = "=" * filled + "-" * (length - filled)
    return f"[{bar}] {val:0.3f} ({tag})"


def record_microphone(duration_seconds: int = 5, prompt_msg: str = "Recording...") -> Path:
    """Records audio from default system microphone and saves as 16kHz mono WAV."""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice is not available. Please ensure microphone access is enabled.")

    print(f"\n[MIC] {prompt_msg}")
    print(f"    Get ready... Recording {duration_seconds} seconds of speech:")
    for i in range(3, 0, -1):
        print(f"    Starting in {i}...", end="\r")
        time.sleep(1)
    print("    >>> RECORDING NOW! Speak clearly into the microphone...          ")

    audio = sd.rec(int(duration_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("    [OK] Recording complete! Processing audio...\n")

    audio_flat = audio.flatten()
    out_path = TEMP_DIR / f"rec_{int(time.time())}.wav"
    sf.write(str(out_path), audio_flat, SAMPLE_RATE)
    return out_path


def display_results(result: dict, user_id: str | None = None):
    print("=" * 65)
    print("                 MINDFLOW AUDIO PREDICTION REPORT                 ")
    print("=" * 65)

    emotion = result.get("calibrated_emotion", result["emotion"]).upper()
    print(f"  Primary Emotion:   [TARGET] {emotion}")
    
    stress_val = result.get("calibrated_stress", result["stress"])
    print(f"  Stress (PHQ-8):    {render_stress_bar(stress_val)}")

    print("\n  Emotion Distribution (7 Classes):")
    probs = result.get("calibrated_emotion_probs", result["emotion_probs"])
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        marker = " <-- (Top)" if emo == result.get("calibrated_emotion", result["emotion"]) else ""
        print(f"    {emo.capitalize():9s} : {render_bar(p)}{marker}")

    if "calibration_metadata" in result:
        meta = result["calibration_metadata"]
        print("\n  Personalized Acoustic Biomarkers (vs. Neutral Baseline):")
        print(f"    - Voice Pitch Shift  : {meta['pitch_delta_pct']:+.1f}%")
        print(f"    - Speech Energy Shift: {meta['energy_delta_db']:+.1f} dB")
        print(f"    - Pause Ratio Shift  : {meta['pause_ratio_delta']:+.2f}")
        print(f"    - Embedding Cosine Sim: {meta['cosine_similarity_to_base']:.3f}")
        if meta["clinical_markers"]:
            print(f"    - Clinical Biomarkers: [ALERT] {', '.join(meta['clinical_markers'])}")
        else:
            print("    - Clinical Biomarkers: None (Within normal variation)")

    emb = result["embedding"]
    print(f"\n  Multimodal Fusion Vector: 768-dim WavLM embedding generated (Norm={np.linalg.norm(emb):.2f})")
    print("=" * 65)


def run_continuous_sliding_window(model: AudioInference, profile: UserProfile | None, window_sec: float = 6.0, hop_sec: float = 1.0):
    """Continuously runs a 6.0-second rolling sliding window updated every 1.0 second."""
    if not HAS_SOUNDDEVICE:
        print("[!] sounddevice is not available.")
        return

    import threading
    window_samples = int(window_sec * SAMPLE_RATE)
    hop_samples = int(hop_sec * SAMPLE_RATE)
    buffer = np.zeros(window_samples, dtype=np.float32)
    buffer_lock = threading.Lock()
    temp_chunk_path = TEMP_DIR / "stream_chunk.wav"

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        data = indata[:, 0]
        with buffer_lock:
            buffer = np.roll(buffer, -len(data))
            buffer[-len(data):] = data

    print("\n" + "=" * 70)
    print("      >>> CONTINUOUS SLIDING WINDOW STREAMING ACTIVATED <<<         ")
    print(f"      Rolling Window: {window_sec}s | Hop/Step Rate: {hop_sec}s                   ")
    print("      Press Ctrl+C at any time to return to the main menu           ")
    print("=" * 70)
    print("Starting stream in 2 seconds...")
    time.sleep(2)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=hop_samples, callback=callback):
        step = 0
        try:
            while True:
                time.sleep(hop_sec)
                step += 1
                with buffer_lock:
                    current_win = buffer.copy()

                rms_energy = np.sqrt(np.mean(current_win ** 2))
                if rms_energy < 0.003:
                    sys.stdout.write(f"\r[Time: {step*hop_sec:4.0f}s]  Listening for speech... (Silence detected)                      ")
                    sys.stdout.flush()
                    continue

                sf.write(str(temp_chunk_path), current_win, SAMPLE_RATE)
                res = model.predict(temp_chunk_path, user_profile=profile)

                emo = res.get("calibrated_emotion", res["emotion"]).upper()
                stress = res.get("calibrated_stress", res["stress"])
                top_p = max(res.get("calibrated_emotion_probs", res["emotion_probs"]).values()) * 100

                meta_str = ""
                if "calibration_metadata" in res:
                    p_shift = res["calibration_metadata"]["pitch_delta_pct"]
                    meta_str = f" | Pitch Δ: {p_shift:+.0f}%"

                line = f"\r[⏱️ {step*hop_sec:4.0f}s] Emotion: {emo:8s} ({top_p:4.1f}%) | Stress: {render_stress_bar(stress)}{meta_str}"
                sys.stdout.write(line.ljust(85))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n[OK] Stopped streaming.")


def main():
    print("=" * 65)
    print("       MindFlow Audio Branch -- Live Real-Time Testing Suite       ")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [System] Device detected: {device.upper()}")
    if device == "cpu":
        print("  [Note] Running in optimized CPU mode suitable for laptops.")
    
    print("  [System] Loading WavLM Large Stage 1 + Stage 2 checkpoints...")
    try:
        model = AudioInference(device=device)
        print("  [System] Model loaded and ready!")
    except Exception as e:
        print(f"  [Error] Failed to load model checkpoints: {e}")
        return

    active_profile: UserProfile | None = None
    user_name = "demo_user"

    while True:
        print("\nChoose an option:")
        print("  1. Record Single Speech Clip (5 seconds)")
        print("  2. Continuous Live Sliding Window Streaming (Real-Time Monitor)")
        print("  3. Calibrate My Personal Voice Baseline (8s neutral recording)")
        print("  4. Test with an Existing WAV File")
        print("  5. Load an Existing User Profile (.json)")
        print("  6. Exit")
        
        status_str = f"Active User: {active_profile.user_id} (Calibrated OK)" if active_profile else "No User Profile Active (Raw Model Mode)"
        print(f"\n  [Status] {status_str}")
        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            if not HAS_SOUNDDEVICE:
                print("[!] sounddevice is not detected. Please use option 4 to test with a WAV file.")
                continue
            try:
                clip_path = record_microphone(duration_seconds=5, prompt_msg="Testing Emotion & Stress")
                res = model.predict(clip_path, user_profile=active_profile)
                display_results(res, user_name)
            except Exception as e:
                print(f"[!] Recording error: {e}")

        elif choice == "2":
            run_continuous_sliding_window(model, active_profile, window_sec=6.0, hop_sec=1.0)

        elif choice == "3":
            if not HAS_SOUNDDEVICE:
                print("[!] sounddevice is not detected.")
                continue
            user_name = input("Enter your name / ID (e.g. vaish): ").strip() or "demo_user"
            print(f"\nSetting up voice calibration for '{user_name}'...")
            print("   Please speak naturally in a neutral, calm voice when recording starts.")
            print("   (e.g., read: 'Today is a normal day. I am testing the speech recognition system.')")
            try:
                base_clip = record_microphone(duration_seconds=8, prompt_msg="Recording Neutral Baseline Voice")
                active_profile = model.register_user_baseline(base_clip, user_id=user_name)
                print(f"[OK] Baseline Profile successfully saved to profiles/{user_name}_profile.json!")
                print(f"   Base Pitch: {active_profile.base_pitch_mean:.1f} Hz | Base Energy: {active_profile.base_energy_rms:.4f}")
            except Exception as e:
                print(f"[!] Calibration error: {e}")

        elif choice == "4":
            wav_input = input("Enter full path to WAV audio file: ").strip().strip('\"').strip(\"'\")
            wav_path = Path(wav_input)
            if not wav_path.exists():
                print(f"[!] File not found: {wav_path}")
                continue
            try:
                res = model.predict(wav_path, user_profile=active_profile)
                display_results(res, user_name)
            except Exception as e:
                print(f"[!] Prediction error: {e}")

        elif choice == "5":
            prof_input = input("Enter path to profile JSON (default: profiles/demo_user_profile.json): ").strip()
            prof_path = Path(prof_input) if prof_input else Path("profiles/demo_user_profile.json")
            if not prof_path.exists():
                print(f"[!] Profile file not found: {prof_path}")
                continue
            active_profile = UserProfile.load(prof_path)
            user_name = active_profile.user_id
            print(f"[OK] Loaded Profile for user '{user_name}'!")

        elif choice == "6":
            print("\nExiting demo. Good luck with your presentation!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
