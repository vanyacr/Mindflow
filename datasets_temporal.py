"""
datasets_temporal.py — Track B sequence-level dataset loaders.

Confirmed real folder structures (via inspect_dfew*.py / inspect_ferv*.py):

  DFEW:
    Clip/clip_224x16f/{video_id:05d}/{1..16}.jpg   (non-padded ints, fixed 16 frames)
    EmoLabel_DataSplit/{train,test}/set_1.csv       (columns: video_name, label 1-7)
    No val split — test doubles as val (documented lesson).

  FERV39K:
    extracted_faces/2_ClipsforFaceCrop/{Scene}/{Emotion}/{clip_id:04d}/{00..NN}.jpg
      (zero-padded, VARIABLE frame count per clip)
    drive-download.../4_setups/22_scenes/{train,test}_{Scene}.csv
      space-delimited: "Scene/Emotion/ClipID Emotion"  (label redundant with path)
    No official val split either — same test-doubles-as-val approach for consistency.

Both loaders return (T, C, H, W) float tensors + one-hot label, ready to
feed into a per-frame frozen backbone -> BiGRU.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from collections import Counter

import config_temporal as cfg


def one_hot(label_str: str) -> np.ndarray:
    v = np.zeros(cfg.NUM_CLASSES, dtype=np.float32)
    idx = cfg.EMOTION_TO_IDX.get(label_str)
    if idx is not None:
        v[idx] = 1.0
    return v


def _uniform_sample_indices(n_available: int, n_target: int) -> list:
    """
    Pick n_target frame indices from n_available frames, evenly spaced.
    Handles both downsampling (FERV39K clips with >16 frames) and
    upsampling with repetition (rare clips with <16 frames).
    """
    if n_available <= 0:
        return []
    if n_available == n_target:
        return list(range(n_available))
    idx = np.linspace(0, n_available - 1, n_target)
    return [int(round(i)) for i in idx]


class SequenceDataset(Dataset):
    """
    Base class. self.samples = list of (frame_path_list, one_hot_label, source_name)
    Each __getitem__ loads SEQ_LEN frames, applies the same transform to
    every frame in the sequence (so augmentation is temporally consistent),
    and returns a (T, C, H, W) tensor.
    """
    def __init__(self, transform=None, dataset_name: str = ""):
        self.samples = []
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, soft_label = self.samples[idx]
        frames = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            if self.transform:
                img = self.transform(image=img)["image"]
            frames.append(img)
        seq = torch.stack(frames, dim=0)   # (T, C, H, W)
        return seq, torch.tensor(soft_label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
#  DFEW
# ─────────────────────────────────────────────────────────────

class DFEWSequenceDataset(SequenceDataset):
    def __init__(self, split: str = "train", transform=None, fold: str = None):
        super().__init__(transform, "dfew")
        fold = fold or cfg.DFEW_FOLD

        # No val split exists — test doubles as val (documented lesson)
        split_folder = "train" if split == "train" else "test"
        csv_path = cfg.DFEW_SPLIT_DIR / split_folder / f"{fold}.csv"

        if not csv_path.exists():
            print(f"  [DFEW]    CSV not found at {csv_path} — skipping")
            return

        df = pd.read_csv(csv_path)
        loaded, missing = 0, 0

        for _, row in df.iterrows():
            video_id = int(row["video_name"])
            label_int = int(row["label"])
            label_str = cfg.DFEW_LABEL_MAP.get(label_int)
            if label_str is None:
                continue

            clip_dir = cfg.DFEW_CLIP_DIR / f"{video_id:05d}"
            if not clip_dir.exists():
                missing += 1
                continue

            # frames are non-zero-padded ints: 1.jpg .. 16.jpg — sort numerically
            frame_files = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if not frame_files:
                missing += 1
                continue

            idxs = _uniform_sample_indices(len(frame_files), cfg.SEQ_LEN)
            chosen = [frame_files[i] for i in idxs]

            self.samples.append((chosen, one_hot(label_str)))
            loaded += 1

        print(f"  [DFEW]    {split} ({fold}): {loaded} clips loaded"
              + (f", {missing} missing on disk" if missing else ""))


# ─────────────────────────────────────────────────────────────
#  FERV39K
# ─────────────────────────────────────────────────────────────

class FERV39KSequenceDataset(SequenceDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "ferv39k")

        loaded, missing = 0, 0

        # FERV39K only ships train_*.csv / test_*.csv — no val_*.csv exists.
        # Same no-val-split situation as DFEW: test doubles as val.
        split_prefix = "train" if split == "train" else "test"

        for scene in cfg.FERV_SCENES:
            csv_path = cfg.FERV_SPLIT_DIR / f"{split_prefix}_{scene}.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # format: "Scene/Emotion/ClipID Emotion"
                    parts = line.rsplit(" ", 1)
                    if len(parts) != 2:
                        continue
                    rel_path, label_folder = parts

                    label_str = cfg.FERV_FOLDER_MAP.get(label_folder)
                    if label_str is None:
                        continue

                    clip_dir = cfg.FERV_CLIP_DIR / rel_path
                    if not clip_dir.exists():
                        missing += 1
                        continue

                    # frames are zero-padded: 00.jpg .. NN.jpg, variable count
                    frame_files = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
                    if not frame_files:
                        missing += 1
                        continue

                    idxs = _uniform_sample_indices(len(frame_files), cfg.SEQ_LEN)
                    chosen = [frame_files[i] for i in idxs]

                    self.samples.append((chosen, one_hot(label_str)))
                    loaded += 1

        print(f"  [FERV39K] {split}: {loaded} clips loaded"
              + (f", {missing} missing on disk" if missing else ""))


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION — lighter than Track A since it's applied per-frame
#  across a whole sequence; heavy per-frame aug would be temporally
#  inconsistent unless we fix the random state per-clip, so we keep
#  train-time augmentation mild and geometric-only.
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str):
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────
#  BUILD COMBINED DATASET + CAPPED WEIGHTED SAMPLER
# ─────────────────────────────────────────────────────────────

def build_temporal_dataset(split: str = "train") -> ConcatDataset:
    tf = get_transforms(split)

    parts = [
        DFEWSequenceDataset(split, tf),
        FERV39KSequenceDataset(split, tf),
    ]
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise RuntimeError("No temporal datasets loaded — check config_temporal.py paths")

    combined = ConcatDataset(parts)
    print(f"\n  Total temporal {split}: {len(combined)} clips from {len(parts)} datasets\n")
    return combined


def build_temporal_weighted_sampler(dataset: ConcatDataset) -> WeightedRandomSampler:
    """
    Same inverse-frequency weighting as Track A's build_weighted_sampler,
    but with CLASS_WEIGHT_CAP applied — this is the direct fix for the
    disgust-11.5x overfit that paused Track B last time.
    """
    label_indices = []
    for ds in dataset.datasets:
        for _, soft in ds.samples:
            label_indices.append(int(np.argmax(soft)))

    n = len(label_indices)
    counts = Counter(label_indices)
    cls_w = {c: n / (cfg.NUM_CLASSES * cnt) for c, cnt in counts.items()}

    # ── THE FIX: clamp every class weight to the cap ──
    capped = {c: min(w, cfg.CLASS_WEIGHT_CAP) for c, w in cls_w.items()}

    print("  Class weights (capped at "
          f"{cfg.CLASS_WEIGHT_CAP}x):")
    for c in sorted(capped):
        raw = cls_w[c]
        flag = "  <- capped" if raw > cfg.CLASS_WEIGHT_CAP else ""
        print(f"    {cfg.IDX_TO_EMOTION[c]:<10} n={counts[c]:<6} "
              f"raw={raw:.2f}  used={capped[c]:.2f}{flag}")

    weights = np.array([capped[label_indices[i]] for i in range(n)], dtype=np.float64)

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=n,
        replacement=True,
    )
