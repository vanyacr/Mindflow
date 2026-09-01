"""
MindFlow Phase 2, Stage 1 — Train the audio model's EMOTION head only.

Why emotion-only for Stage 1:
    CREMA-D / RAVDESS / SAVEE / TESS are acted-emotion datasets with no
    stress or confidence labels. Training those heads now would just teach
    them noise. Stress gets a real training signal later, in Stage 2, from
    DAIC-WOZ's PHQ-8 scores. Confidence has no labeled dataset yet at all —
    revisit once you have one (self-reported confidence, or a proxy).

Data:
    Reads metadata_crema_d.csv, metadata_ravdess.csv, metadata_savee.csv,
    metadata_tess.csv directly (NOT the combined metadata.csv, so this
    never has to wait on MELD/DAIC-WOZ to finish processing).

Split:
    Speaker-disjoint 85/15 train/val split — critical for acted datasets,
    since random splitting would let the same actor's voice appear in both
    train and val, inflating validation accuracy artificially. TESS is
    forced into train-only (per config.settings.SPEAKER_DISJOINT_INCOMPATIBLE)
    since it only has 2 actresses total — not enough speakers to hold any
    out for validation.

Usage:
    python train_stage1.py --epochs 15 --batch-size 32
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import METADATA_DIR, UNIFIED_EMOTIONS, SPEAKER_DISJOINT_INCOMPATIBLE
from audio_model.audio_model import AudioModel

STAGE1_DATASETS = ["crema_d", "ravdess", "savee", "tess", "meld", "iemocap"]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(UNIFIED_EMOTIONS)}
MAX_AUDIO_SECONDS = 6  # clips longer than this get truncated; shorter ones padded
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Data loading + speaker-disjoint split
# ---------------------------------------------------------------------------

def load_stage1_metadata() -> pd.DataFrame:
    frames = []
    for name in STAGE1_DATASETS:
        path = METADATA_DIR / f"metadata_{name}.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No Stage 1 metadata files found. Run run_phase1.py first.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["emotion", "audio_path"])
    combined["emotion_idx"] = combined["emotion"].map(EMOTION_TO_IDX)
    return combined


def speaker_disjoint_split(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42):
    """
    Splits by speaker, not by row, so no actor appears in both sets.
    TESS is forced entirely into train (only 2 speakers total; can't hold one out).
    """
    forced_train = df[df["dataset"].str.upper().isin([d.upper() for d in SPEAKER_DISJOINT_INCOMPATIBLE])]
    splittable = df[~df["dataset"].str.upper().isin([d.upper() for d in SPEAKER_DISJOINT_INCOMPATIBLE])]

    if len(splittable) == 0:
        return df, pd.DataFrame(columns=df.columns)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    # group by (dataset, speaker) since speaker IDs can collide across datasets (e.g. "1")
    groups = splittable["dataset"] + "_" + splittable["speaker"].astype(str)
    train_idx, val_idx = next(gss.split(splittable, groups=groups))

    train_df = pd.concat([splittable.iloc[train_idx], forced_train], ignore_index=True)
    val_df = splittable.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmotionAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            y, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        except Exception:
            # Fallback for any corrupt sample
            y = np.zeros(MAX_AUDIO_SECONDS * SAMPLE_RATE, dtype=np.float32)

        # 1. Trim silence if longer than 0.5s
        trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(trimmed) > int(0.5 * SAMPLE_RATE):
            y = trimmed

        # 2. Peak normalization
        peak = np.max(np.abs(y))
        if peak > 1e-6:
            y = y / peak * 0.95

        # 3. Data Augmentation during training (Gain perturbation)
        if self.augment:
            # Random gain perturbation between -6dB and +6dB
            gain_db = np.random.uniform(-6.0, 6.0)
            gain_factor = 10.0 ** (gain_db / 20.0)
            y = np.clip(y * gain_factor, -1.0, 1.0)

        max_len = MAX_AUDIO_SECONDS * SAMPLE_RATE
        if len(y) > max_len:
            # Random crop during training, center crop during validation
            if self.augment:
                offset = np.random.randint(0, len(y) - max_len + 1)
                y = y[offset : offset + max_len]
            else:
                start = (len(y) - max_len) // 2
                y = y[start : start + max_len]
        elif len(y) < max_len:
            # Tile active speech rather than padding dead zeros
            n_repeats = int(np.ceil(max_len / max(len(y), 1)))
            y = np.tile(y, n_repeats)[:max_len]

        return torch.tensor(y, dtype=torch.float32), int(row["emotion_idx"])


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def make_optimizer(model: AudioModel, backbone_lr: float, head_lr: float):
    """Differential LR: WavLM's unfrozen layers get a small LR (already pretrained,
    should move slowly); the new pooling/projection/heads get a normal LR (random
    init, need to learn faster)."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.pooling.parameters())
        + list(model.projection.parameters())
        + list(model.emotion_head.parameters())
    )
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ])


from tqdm import tqdm


def run_epoch(model, loader, optimizer, criterion, device, scaler, train: bool, desc: str = ""):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    with torch.set_grad_enabled(train):
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                output = model(waveforms)
                loss = criterion(output["emotion_logits"], labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss * waveforms.size(0)
            preds = output["emotion_logits"].argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to a checkpoint to resume/extend training from (e.g. checkpoints/stage1_best.pt)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_stage1_metadata()
    print(f"Loaded {len(df)} labeled rows across {sorted(df['dataset'].unique())}")
    print(df["emotion"].value_counts())

    train_df, val_df = speaker_disjoint_split(df)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    train_loader = DataLoader(
        EmotionAudioDataset(train_df, augment=True), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        EmotionAudioDataset(val_df, augment=False), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )

    model = AudioModel(num_emotions=len(UNIFIED_EMOTIONS)).to(device)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise RuntimeError(f"--resume-from checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        # stress_head architecture changed (single layer -> MLP) after this
        # checkpoint was saved. Stage 1 never trains stress_head anyway, so
        # skip those keys and load everything else (backbone + emotion head).
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("stress_head")}
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"Resumed weights from {resume_path}")
        print(f"  Missing keys (expected -- stress_head is separate/untouched by Stage 1): {missing}")
        if unexpected:
            print(f"  WARNING -- unexpected keys: {unexpected}")

    trainable, total = model.trainable_parameter_count()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr)

    # Class-weighted loss: rarer classes (e.g. 'surprise') get weighted more
    # heavily so the model can't just under-attend to them.
    class_counts = train_df["emotion_idx"].value_counts().sort_index()
    class_counts = class_counts.reindex(range(len(UNIFIED_EMOTIONS)), fill_value=1)
    class_weights = (1.0 / class_counts.values)
    class_weights = class_weights / class_weights.sum() * len(UNIFIED_EMOTIONS)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {dict(zip(UNIFIED_EMOTIONS, class_weights.round(2)))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, train=True,
            desc=f"Epoch {epoch:02d}/{args.epochs} [Train]"
        )
        val_loss, val_acc, val_f1, val_labels, val_preds = run_epoch(
            model, val_loader, optimizer, criterion, device, scaler, train=False,
            desc=f"Epoch {epoch:02d}/{args.epochs} [Val]"
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_dir / "stage1_best.pt")
            print(f"  -> New best val_f1={val_f1:.3f}, checkpoint saved.")

    # Final confusion matrix on the best epoch's validation predictions
    print("\nFinal validation confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(val_labels, val_preds, labels=list(range(len(UNIFIED_EMOTIONS))))
    print("Labels:", UNIFIED_EMOTIONS)
    print(cm)


if __name__ == "__main__":
    main()