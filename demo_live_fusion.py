"""Live demo runner for text + keystroke stress outputs.

Usage:
    python demo_live_fusion.py

Type normally while this runs. Every window, it prints keystroke score.
Optionally provide a text line to compute text score and fused score.
"""

from __future__ import annotations

import argparse
import threading
import time

from inference import run_text_and_keystroke_inference
from keystroke import KeystrokeTracker


def _input_worker(stop_event: threading.Event, text_box: dict) -> None:
    while not stop_event.is_set():
        try:
            value = input("Text (optional, press Enter to skip, q to quit): ").strip()
        except EOFError:
            stop_event.set()
            return

        if value.lower() in {"q", "quit", "exit"}:
            stop_event.set()
            return

        text_box["value"] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Live text+keystroke fusion demo")
    parser.add_argument("--window", type=int, default=15, help="Window length in seconds (10-30 recommended)")
    parser.add_argument("--model", type=str, default="", help="Path to trained keystroke_model.pkl")
    parser.add_argument("--baseline", type=str, default="", help="Path to baseline JSON from enrollment")
    args = parser.parse_args()

    tracker = KeystrokeTracker(poll_interval_s=0.1)
    if args.baseline:
        try:
            tracker.load_baseline(args.baseline)
            print(f"Loaded baseline from: {args.baseline}")
        except Exception as exc:
            print(f"Could not load baseline ({args.baseline}): {exc}")

    print("Starting keystroke listener...")

    try:
        tracker.start_listener()
    except RuntimeError as exc:
        print(f"Listener unavailable: {exc}")
        print("Install dependency with: pip install pynput")
        return

    stop_event = threading.Event()
    text_box = {"value": ""}
    thread = threading.Thread(target=_input_worker, args=(stop_event, text_box), daemon=True)
    thread.start()

    print("Live demo started. Type on keyboard to generate keystroke data.")
    print("Press q in text prompt to exit.\n")

    try:
        while not stop_event.is_set():
            time.sleep(max(1, args.window))

            text = text_box.get("value") or None
            text_box["value"] = ""

            result = run_text_and_keystroke_inference(
                tracker=tracker,
                text=text,
                model_path=(args.model or None),
                keystroke_window_seconds=args.window,
            )

            k = result["keystroke"]
            t = result["text"]
            f = result["fusion"]

            print("-" * 68)
            print(f"Keystroke score: {k['keystroke_score']:.3f} | confidence: {k['confidence']:.3f} | model: {k['model_used']}")
            print(f"Keystroke z-features: {k['zscore_features']}")
            if float(k["features"].get("sample_count", 0.0)) < 8:
                print("Note: low keystroke activity in current window; score may be unstable.")

            if t is not None:
                print(f"Text score: {t['text_score']:.3f} | sentiment: {t['sentiment']:.3f} | keywords: {t['emotion_keywords']}")
            else:
                print("Text score: skipped (no new text)")

            print(f"Fused score: {f['final_score']:.3f} | used: {f['used_modalities']} | mask: {f['mask']}")

            tracker.clear_old_records(keep_seconds=600)

    finally:
        tracker.stop_listener()
        stop_event.set()
        print("\nDemo stopped.")


if __name__ == "__main__":
    main()
