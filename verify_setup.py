"""
verify_setup.py — Run this AFTER dropping in the new files, BEFORE python train.py.

Loads train + val datasets exactly like train.py would, but does no training.
Just reports what got loaded from each source, and the final per-class balance,
so you can catch a bad path or empty loader before burning GPU hours on it.

Run:
    python verify_setup.py
"""

from collections import Counter
import numpy as np

import config
from datasets import build_dataset

print("=" * 70)
print("VERIFY SETUP — dataset loading only, no training")
print("=" * 70)

print("\n--- TRAIN split ---")
train_ds = build_dataset("train")

print("\n--- VAL split ---")
val_ds = build_dataset("val")

print("\n" + "=" * 70)
print("Per-source sample counts")
print("=" * 70)
for split_name, ds in [("train", train_ds), ("val", val_ds)]:
    print(f"\n[{split_name}]")
    for part in ds.datasets:
        print(f"  {part.dataset_name:<10} {len(part):>7} samples")

print("\n" + "=" * 70)
print("Per-class distribution (train)")
print("=" * 70)
labels = []
for part in train_ds.datasets:
    for _, soft in part.samples:
        labels.append(config.IDX_TO_EMOTION[int(np.argmax(soft))])
counts = Counter(labels)
total = sum(counts.values())
for emo in config.EMOTIONS:
    c = counts.get(emo, 0)
    pct = 100 * c / max(total, 1)
    bar = "#" * int(pct / 2)
    print(f"  {emo:<10} {c:>7}  ({pct:5.1f}%)  {bar}")

print(f"\n  TOTAL train samples: {total}")

expected_sources = {"affectnet", "ckplus", "fer_ck48", "fer_k7", "fer_stock",
                     "rafdb", "dfew", "ferv39k"}
loaded_sources = {p.dataset_name for p in train_ds.datasets}
missing = expected_sources - loaded_sources
if missing:
    print(f"\n  ⚠ WARNING: these sources loaded ZERO samples and were dropped: {missing}")
    print("  Check the corresponding path in config.py before running train.py")
else:
    print("\n  ✓ All 8 sources loaded successfully (AffectNet, CK+, FER+ x3, RAF-DB, DFEW, FERV39K)")

print("=" * 70)
