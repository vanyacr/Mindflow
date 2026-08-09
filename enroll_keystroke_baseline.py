"""Enroll a user baseline for keystroke z-score normalization.

Usage:
    python enroll_keystroke_baseline.py --minutes 3 --window 15 --out data/keystroke/baseline_user1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from keystroke import KeystrokeTracker


RAW_KEYS = [
    "velocity",
    "dwell_mean_ms",
    "dwell_std_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "pause_freq",
    "error_count",
    "backspace_rate",
    "burst_ratio",
    "key_variation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline keystroke stats")
    parser.add_argument("--minutes", type=float, default=3.0, help="Enrollment duration in minutes")
    parser.add_argument("--window", type=int, default=15, help="Feature window in seconds")
    parser.add_argument("--out", type=str, default="data/keystroke/baseline.json", help="Baseline JSON output path")
    args = parser.parse_args()

    tracker = KeystrokeTracker(poll_interval_s=0.1)
    print("Starting baseline enrollment listener.")
    print("Type normally and calmly during enrollment.")

    try:
        tracker.start_listener()
    except RuntimeError as exc:
        raise RuntimeError(f"Could not start listener: {exc}") from exc

    rows = []
    end = time.time() + (args.minutes * 60.0)

    try:
        while time.time() < end:
            time.sleep(max(1, args.window))
            feats = tracker.compute_features(window_seconds=args.window)
            if feats.get("sample_count", 0.0) >= 8:
                rows.append(feats)
                print(f"Captured baseline window {len(rows)}: {feats}")
            else:
                print("Skipping sparse window (<8 keystrokes).")
            tracker.clear_old_records(keep_seconds=600)
    finally:
        tracker.stop_listener()

    if not rows:
        raise RuntimeError("No valid baseline windows captured. Please type continuously and retry.")

    means = {k: float(np.mean([r[k] for r in rows])) for k in RAW_KEYS}
    stds = {k: float(max(1e-6, np.std([r[k] for r in rows]))) for k in RAW_KEYS}

    tracker.set_baseline_stats(means=means, stds=stds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracker.save_baseline(out_path)

    print(f"Saved baseline to: {out_path}")
    print("Baseline means:", means)
    print("Baseline stds:", stds)


if __name__ == "__main__":
    main()
