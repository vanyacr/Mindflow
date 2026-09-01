"""
MindFlow — Quick Audio Verification & Demo CLI
==============================================
Quickly test any audio WAV file and see full emotion distribution, PHQ-8 stress score,
and 768-dim multimodal fusion vector.

Usage:
    python demo_quick_test.py path/to/sample.wav [path/to/profile.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix Windows console encoding for UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference.audio_interface import AudioInference
from inference.user_calibration import UserProfile


def main():
    if len(sys.argv) < 2:
        sample = Path(__file__).resolve().parent / "demo_samples" / "sample_happy.wav"
        if not sample.exists():
            print("Usage: python demo_quick_test.py path/to/sample.wav [profile.json]")
            sys.exit(1)
        audio_path = sample
    else:
        audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print(f"Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    profile = None
    if len(sys.argv) >= 3:
        prof_path = Path(sys.argv[2])
        if prof_path.exists():
            profile = UserProfile.load(prof_path)
            print(f"Loaded User Baseline Profile: {profile.user_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")
    model = AudioInference(device=device)

    print(f"\nEvaluating: {audio_path.name}")
    res = model.predict(audio_path, user_profile=profile)

    print("\n" + "=" * 60)
    print("                [MINDFLOW PREDICTION REPORT]                ")
    print("=" * 60)
    print(f"  Emotion:       [TARGET] {res.get('calibrated_emotion', res['emotion']).upper()}")
    print(f"  Stress Score:  {res.get('calibrated_stress', res['stress']):.3f} (PHQ-8 Continuous [0-1])")
    
    print("\n  Emotion Class Probabilities:")
    probs = res.get("calibrated_emotion_probs", res["emotion_probs"])
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 25) + "-" * (25 - int(p * 25))
        print(f"    {emo.capitalize():9s} [{bar}] {p*100:5.1f}%")

    if "calibration_metadata" in res:
        meta = res["calibration_metadata"]
        print("\n  Personalized Acoustic Deltas:")
        print(f"    - Pitch Shift:       {meta['pitch_delta_pct']:+.1f}%")
        print(f"    - Energy Delta:      {meta['energy_delta_db']:+.1f} dB")
        print(f"    - Pause Ratio Delta: {meta['pause_ratio_delta']:+.2f}")
        print(f"    - Cosine Similarity: {meta['cosine_similarity_to_base']:.3f}")

    emb = res["embedding"]
    print(f"\n  Multimodal Fusion Vector (768-D): length {len(emb)}, L2 Norm: {np.linalg.norm(emb):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
