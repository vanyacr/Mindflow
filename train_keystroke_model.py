"""Train and evaluate keystroke stress model from CSV.

Expected CSV columns:
- velocity_zscore
- dwell_mean_zscore
- dwell_std_zscore
- latency_mean_zscore
- pause_freq_zscore
- error_count_zscore
- label  (0=normal, 1=stressed)

Usage:
    python train_keystroke_model.py --csv data/keystroke/windows.csv --out keystroke_model.pkl
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import joblib


FEATURES = [
    "velocity_zscore",
    "dwell_mean_zscore",
    "dwell_std_zscore",
    "latency_mean_zscore",
    "latency_std_zscore",
    "pause_freq_zscore",
    "error_count_zscore",
    "backspace_rate_zscore",
    "burst_ratio_zscore",
    "key_variation_zscore",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train keystroke stress model")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled window feature CSV")
    parser.add_argument("--out", type=str, default="keystroke_model.pkl", help="Model output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].astype(float)
    y = df["label"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Need both classes in label column (0 and 1).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=args.seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("F1 score:", round(float(f1), 4))
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, pred, digits=4))
    print("Example probabilities (first 5):", [round(float(v), 4) for v in proba[:5]])

    joblib.dump(model, args.out)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
