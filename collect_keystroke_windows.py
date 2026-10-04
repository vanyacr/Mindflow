"""Collect labeled keystroke windows into CSV for model training.

This script records keystroke windows and writes z-score feature rows.
You run it twice (or more):
- label=0 for normal typing
- label=1 for stressed typing

Usage:
    python collect_keystroke_windows.py --label 0 --minutes 2 --out data/keystroke/windows.csv
    python collect_keystroke_windows.py --label 1 --minutes 2 --out data/keystroke/windows.csv --append
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from keystroke import KeystrokeTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled keystroke window features")
    parser.add_argument("--label", type=int, choices=[0, 1], required=True, help="0=normal, 1=stressed")
    parser.add_argument("--minutes", type=float, default=2.0, help="Collection duration in minutes")
    parser.add_argument("--window", type=int, default=15, help="Feature window in seconds")
    parser.add_argument("--out", type=str, default="data/keystroke/windows.csv", help="CSV output")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV")
    parser.add_argument("--baseline", type=str, default="", help="Optional baseline JSON for z-score normalization")
    args = parser.parse_args()

    tracker = KeystrokeTracker(poll_interval_s=0.1)
    if args.baseline:
        try:
            tracker.load_baseline(args.baseline)
            print(f"Loaded baseline from: {args.baseline}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load baseline file '{args.baseline}': {exc}") from exc

    print("Starting listener. Type naturally in any app/window.")
    print(f"Collecting label={args.label} for {args.minutes} minutes...")

    try:
        tracker.start_listener()
    except RuntimeError as exc:
        raise RuntimeError(f"Could not start listener: {exc}") from exc
    rows = []

    start = time.time()
    end = start + (args.minutes * 60.0)

    try:
        while time.time() < end:
            time.sleep(max(1, args.window))
            out = tracker.keystroke_stress_score(window_seconds=args.window, model_path=None)
            z = out["zscore_features"]
            z["label"] = args.label
            z["sample_count"] = out["features"]["sample_count"]
            rows.append(z)
            print(f"Captured window {len(rows)}: {z}")
            tracker.clear_old_records(keep_seconds=600)
    finally:
        tracker.stop_listener()

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out_path.exists():
        old = pd.read_csv(out_path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(rows)} new windows to {out_path}")


if __name__ == "__main__":
    main()
