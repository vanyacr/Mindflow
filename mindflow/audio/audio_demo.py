"""
MindFlow — Audio Demo & Evaluation Suite for Project Reviews.

Usage Examples:
    1. Single file evaluation:
       python audio_demo.py --file path/to/audio.wav

    2. Single file with visual plot:
       python audio_demo.py --file path/to/audio.wav --plot

    3. Entire folder / batch of new audio files:
       python audio_demo.py --dir path/to/my_new_audios/

    4. Live Microphone Recording:
       python audio_demo.py --mic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa

# Ensure stdout handles unicode/utf-8 safely on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add mindflow_pipeline to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.audio_interface import AudioInference
from mic_capture import record_mic


def render_ascii_bar(val: float, max_bars: int = 25) -> str:
    filled = int(round(val * max_bars))
    return "#" * filled + "-" * (max_bars - filled)


def print_prediction_card(audio_path: str, result: dict) -> None:
    emotion = result["emotion"].upper()
    probs = result["emotion_probs"]
    stress = result["stress"]
    
    print("\n" + "=" * 65)
    print(f"[AUDIO FILE] {Path(audio_path).name}")
    print("=" * 65)
    print(f"  PREDICTED EMOTION : {emotion} ({probs[result['emotion']]*100:.1f}%)")
    print(f"  ESTIMATED STRESS   : {stress*100:.1f}% ({'HIGH / ELEVATED' if stress >= 0.417 else 'NORMAL / LOW'})")
    print("-" * 65)
    print("  Emotion Probability Distribution:")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = render_ascii_bar(p, max_bars=20)
        print(f"   {emo:10s} : [{bar}] {p*100:5.1f}%")
    print("=" * 65 + "\n")


def plot_prediction(audio_path: str, result: dict, save_path: str | None = None) -> None:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    probs = result["emotion_probs"]
    emotions = list(probs.keys())
    values = [probs[e] * 100 for e in emotions]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Waveform
    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(time_axis, y, color="#2b5c8f", lw=0.8)
    ax1.set_title(f"Audio Waveform: {Path(audio_path).name}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Probability Bar Chart
    colors = ["#4C72B0" if e != result["emotion"] else "#D9534F" for e in emotions]
    bars = ax2.bar(emotions, values, color=colors)
    ax2.set_title(f"Predicted Emotion: {result['emotion'].upper()} | Stress: {result['stress']*100:.1f}%", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    out = save_path or str(Path(audio_path).with_suffix(".png"))
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT SAVED] Visual analysis chart saved to: {out}")


def process_directory(infer: AudioInference, dir_path: str) -> None:
    folder = Path(dir_path)
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = [f for f in folder.glob("*") if f.suffix.lower() in audio_extensions]
    
    if not files:
        print(f"[ERROR] No audio files found in: {dir_path}")
        return
        
    print(f"\n[BATCH EVALUATION] Processing {len(files)} audio files from: {dir_path}")
    results = []
    
    for i, file_path in enumerate(files, 1):
        try:
            res = infer.predict(str(file_path))
            results.append({
                "Filename": file_path.name,
                "Predicted Emotion": res["emotion"],
                "Confidence (%)": round(res["emotion_probs"][res["emotion"]] * 100, 1),
                "Stress (%)": round(res["stress"] * 100, 1),
            })
            print(f"[{i}/{len(files)}] {file_path.name[:25]:25s} -> {res['emotion']:10s} (Stress: {res['stress']*100:4.1f}%)")
        except Exception as e:
            print(f"[{i}/{len(files)}] Error processing {file_path.name}: {e}")
            
    df = pd.DataFrame(results)
    out_csv = folder / "demo_results_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 65)
    print(f"[SUMMARY] Batch Processing Complete! Results Table:")
    print(df.to_string(index=False))
    print(f"\n[SAVED] Detailed results saved to: {out_csv}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MindFlow Audio Inference & Review Demo")
    parser.add_argument("--file", type=str, help="Path to single audio file for evaluation")
    parser.add_argument("--dir", type=str, help="Path to a directory containing audio files to process in batch")
    parser.add_argument("--mic", action="store_true", help="Record live speech from microphone and predict")
    parser.add_argument("--duration", type=float, default=5.0, help="Mic recording duration in seconds (default: 5.0)")
    parser.add_argument("--plot", action="store_true", help="Generate and save visual prediction graph")
    parser.add_argument("--stage1", type=str, default="checkpoints/stage1_best.pt", help="Stage 1 checkpoint path")
    parser.add_argument("--stage2", type=str, default="checkpoints/stage2_stress_best.pt", help="Stage 2 checkpoint path")
    args = parser.parse_args()

    # Initialize model
    print("\n[INIT] Loading MindFlow Audio Neural Network...")
    infer = AudioInference(stage1_checkpoint=args.stage1, stage2_checkpoint=args.stage2)
    print("[SUCCESS] Model loaded successfully!\n")

    if args.mic:
        temp_audio = record_mic(duration_seconds=args.duration, output_path="temp_live_mic.wav")
        res = infer.predict(temp_audio)
        print_prediction_card(temp_audio, res)
        if args.plot:
            plot_prediction(temp_audio, res, save_path="live_mic_prediction.png")
            
    elif args.file:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        res = infer.predict(args.file)
        print_prediction_card(args.file, res)
        if args.plot:
            plot_prediction(args.file, res)

    elif args.dir:
        process_directory(infer, args.dir)

    else:
        # Default interactive menu
        print("=" * 65)
        print("  MindFlow Audio Module -- Review Demo Mode")
        print("=" * 65)
        print("Options:")
        print("  1. Record Live Audio from Microphone")
        print("  2. Test a specific audio file")
        print("  3. Process a folder of audio files")
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            temp_audio = record_mic(duration_seconds=5.0, output_path="temp_live_mic.wav")
            res = infer.predict(temp_audio)
            print_prediction_card(temp_audio, res)
        elif choice == "2":
            p = input("Enter path to audio file: ").strip().strip('"').strip("'")
            if Path(p).exists():
                res = infer.predict(p)
                print_prediction_card(p, res)
            else:
                print(f"[ERROR] File not found: {p}")
        elif choice == "3":
            d = input("Enter folder path: ").strip().strip('"').strip("'")
            if Path(d).exists():
                process_directory(infer, d)
            else:
                print(f"[ERROR] Directory not found: {d}")


if __name__ == "__main__":
    main()
