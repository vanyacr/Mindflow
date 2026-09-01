"""
Phase 1, step 5 — Data Analysis.

Reads processed/metadata/metadata.csv and produces:
  - emotion_distribution.png
  - gender_distribution.png
  - dataset_distribution.png
  - duration_histogram.png

Requires: pandas, matplotlib
    pip install pandas matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from config.settings import METADATA_DIR

OUT_DIR = METADATA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    path = METADATA_DIR / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_phase1.py first."
        )
    df = pd.read_csv(path)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return df


def plot_bar(series: pd.Series, title: str, xlabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    series.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / filename}")


def plot_duration_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df["duration"].dropna().plot(kind="hist", bins=40, ax=ax, color="#55A868")
    ax.set_title("Audio Clip Duration Distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "duration_histogram.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'duration_histogram.png'}")


def print_imbalance_summary(df: pd.DataFrame) -> None:
    print("\n=== Class balance summary ===")
    counts = df["emotion"].value_counts()
    print(counts)
    ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    print(f"\nMax/min class ratio: {ratio:.2f}x")
    if ratio > 3:
        print("-> Significant imbalance. Consider class-weighted loss, "
              "oversampling minority classes, or augmentation (Phase 2) "
              "targeted at the smallest classes.")


def main():
    df = load_metadata()

    plot_bar(df["emotion"].value_counts(), "Emotion Distribution", "Emotion", "emotion_distribution.png")
    plot_bar(df["gender"].fillna("unknown").replace("", "unknown").value_counts(),
              "Gender Distribution", "Gender", "gender_distribution.png")
    plot_bar(df["dataset"].value_counts(), "Dataset Distribution", "Dataset", "dataset_distribution.png")
    plot_duration_histogram(df)
    print_imbalance_summary(df)


if __name__ == "__main__":
    main()
