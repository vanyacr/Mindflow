"""
MindFlow Phase 2, Stage 2 (v3-CV) — Partial backbone fine-tuning, PROPERLY
validated with 5-fold cross-validation and a FIXED epoch count.

Why this version exists:
    The single-split v3 runs showed real improvement over baseline in all
    4 tested seeds (avg baseline ~0.21 -> avg best-epoch ~0.49) -- but each
    run picked whichever epoch scored highest on its own validation set,
    which optimistically inflates the number. That's not a fair result to
    ship or report as "the accuracy."

    This script fixes that by:
    1. Using 5-fold CV (like v2's original validation) instead of one split
    2. Training each fold for a FIXED number of epochs (chosen in advance,
       not selected per-fold based on which looked best) -- this removes
       the cherry-picking bias entirely
    3. Reporting the average final-epoch correlation across all 5 folds,
       directly comparable to v2's reported 0.185 +/- 0.084

    FIXED_EPOCHS=4 was chosen from observing the single-split runs: corr
    was elevated in most runs by epoch 2-4 before later epochs became
    unstable/degraded. This is a reasonable, defensible choice -- not
    tuned to maximize any one fold's result.

Usage:
    python train_stage2_stress_v3_cv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import METADATA_DIR, UNIFIED_EMOTIONS
from audio_model.audio_model import AudioModel

SAMPLE_RATE = 16000
CROP_SECONDS = 15
MAX_PHQ8_SCORE = 24.0
PHQ8_BINARY_STRESS_CUTOFF = 10.0 / MAX_PHQ8_SCORE
NUM_UNFREEZE_LAYERS = 2
N_FOLDS = 5
FIXED_EPOCHS = 4   # decided IN ADVANCE, not per-fold -- see docstring
V2_BASELINE_CORR = 0.185
V2_BASELINE_STD = 0.084


def load_daic_metadata() -> pd.DataFrame:
    path = METADATA_DIR / "metadata_daic_woz.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["audio_path", "phq8_score"])
    df["stress_target"] = (df["phq8_score"] / MAX_PHQ8_SCORE).clip(0, 1)
    return df.reset_index(drop=True)


class StressAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_len = CROP_SECONDS * SAMPLE_RATE
        try:
            info = sf.info(row["audio_path"])
            total_samples = int(info.frames * SAMPLE_RATE / info.samplerate)
        except Exception:
            total_samples = crop_len

        if total_samples <= crop_len:
            y, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
            y = np.pad(y, (0, max(0, crop_len - len(y))))[:crop_len]
        else:
            max_offset = total_samples - crop_len
            offset_samples = np.random.randint(0, max_offset + 1)
            y, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True,
                                 offset=offset_samples / SAMPLE_RATE, duration=CROP_SECONDS)
            if len(y) < crop_len:
                y = np.pad(y, (0, crop_len - len(y)))
            y = y[:crop_len]

        return (
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(row["stress_target"], dtype=torch.float32),
            torch.tensor(row["phq8_binary"], dtype=torch.float32),
        )


def collate_fn(batch):
    waveforms, targets, binaries = zip(*batch)
    return torch.stack(waveforms), torch.stack(targets), torch.stack(binaries)


def unfreeze_top_layers(model: AudioModel, num_layers: int):
    for param in model.parameters():
        param.requires_grad = False
    total_layers = len(model.backbone.encoder.layers)
    for i, layer in enumerate(model.backbone.encoder.layers):
        if i >= total_layers - num_layers:
            for param in layer.parameters():
                param.requires_grad = True
    for param in model.stress_head.parameters():
        param.requires_grad = True


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_binaries = [], [], []
    with torch.no_grad():
        for waveforms, targets, binaries in loader:
            waveforms = waveforms.to(device)
            output = model(waveforms)
            all_preds.extend(output["stress"].cpu().numpy())
            all_targets.extend(targets.numpy())
            all_binaries.extend(binaries.numpy())
    mae = mean_absolute_error(all_targets, all_preds)
    corr, _ = pearsonr(all_targets, all_preds) if len(set(all_targets)) > 1 else (0.0, 1.0)
    pred_binary = [1 if p >= PHQ8_BINARY_STRESS_CUTOFF else 0 for p in all_preds]
    bin_acc = accuracy_score(all_binaries, pred_binary)
    bin_f1 = f1_score(all_binaries, pred_binary, zero_division=0)
    return mae, corr, bin_acc, bin_f1


def run_one_fold(fold_idx, train_df, val_df, args, device):
    model = AudioModel(num_emotions=len(UNIFIED_EMOTIONS)).to(device)
    checkpoint = torch.load(args.v2_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    unfreeze_top_layers(model, NUM_UNFREEZE_LAYERS)

    train_loader = DataLoader(StressAudioDataset(train_df), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(StressAudioDataset(val_df), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.stress_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)

    pos_rate = train_df["phq8_binary"].mean()
    pos_weight_val = (1 - pos_rate) / max(pos_rate, 1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        for waveforms, targets, binaries in train_loader:
            waveforms, targets, binaries = waveforms.to(device), targets.to(device), binaries.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                output = model(waveforms)
                preds = output["stress"]
                per_sample_loss = nn.functional.mse_loss(preds, targets, reduction="none")
                weights = torch.where(binaries > 0.5,
                                       torch.tensor(pos_weight_val, device=device),
                                       torch.tensor(1.0, device=device))
                loss = (per_sample_loss * weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Evaluate ONLY after the fixed epoch count -- no per-epoch best-checkpoint selection
    mae, corr, acc, f1 = evaluate(model, val_loader, device)
    print(f"Fold {fold_idx}/{N_FOLDS} | val_mae={mae:.3f} val_corr={corr:.3f} "
          f"val_bin_acc={acc:.3f} val_bin_f1={f1:.3f}  (after {FIXED_EPOCHS} fixed epochs)")
    return mae, corr, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--v2-checkpoint", type=str, default="checkpoints/stage2_stress_best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Fixed epoch count per fold: {FIXED_EPOCHS} (decided in advance -- no cherry-picking)\n")

    df = load_daic_metadata()
    print(f"Loaded {len(df)} DAIC-WOZ sessions, binary positive rate: {df['phq8_binary'].mean():.2f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_maes, fold_corrs, fold_accs, fold_f1s = [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["phq8_binary"]), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        mae, corr, acc, f1 = run_one_fold(fold_idx, train_df, val_df, args, device)
        fold_maes.append(mae)
        fold_corrs.append(corr)
        fold_accs.append(acc)
        fold_f1s.append(f1)

    print("\n=== v3-CV Fixed-Epoch 5-Fold Summary (directly comparable to v2) ===")
    print(f"MAE:     {np.mean(fold_maes):.3f} +/- {np.std(fold_maes):.3f}")
    print(f"Corr:    {np.mean(fold_corrs):.3f} +/- {np.std(fold_corrs):.3f}")
    print(f"Bin Acc: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"Bin F1:  {np.mean(fold_f1s):.3f} +/- {np.std(fold_f1s):.3f}")
    print(f"\nv2 baseline (frozen backbone): corr={V2_BASELINE_CORR:.3f} +/- {V2_BASELINE_STD:.3f}")
    print(f"v3 (partial fine-tune, fixed {FIXED_EPOCHS} epochs): corr={np.mean(fold_corrs):.3f} +/- {np.std(fold_corrs):.3f}")

    if np.mean(fold_corrs) > V2_BASELINE_CORR:
        print("\n-> v3 outperforms v2 on this fair, apples-to-apples comparison.")
        print("   This IS safe to report/ship as a genuine improvement.")
    else:
        print("\n-> v3 does NOT clearly outperform v2 once cherry-picking is removed.")
        print("   Ship v2 as your production model; report v3 as an explored-but-inconclusive direction.")


if __name__ == "__main__":
    main()
