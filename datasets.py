"""
datasets.py — full rebuild.

Static image loaders (AffectNet, CK+, FER+, RAF-DB) are unchanged in behaviour
from before — same folder assumptions, verified against disk.

NEW: DFEWFramesDataset and FERV39KFramesDataset sample a few frames per video
clip and feed them into the SAME static Track A pipeline as regular images.
This is deliberate — Track A is a per-frame classifier, so a sampled frame
from a clip is just another labeled face image to it.

Both new loaders are built from the exact structure confirmed via
inspect_datasets_deep.py:
  DFEW    : data/DFEW/Clip/clip_224x16f/<5-digit id>/{1..16}.jpg
            labels in data/DFEW/EmoLabel_DataSplit/{train,test}/set_1.csv
            (video_name,label) — label is DFEW's own 1-7 convention, remapped below.
  FERV39K : data/FERV39k/2_ClipsforFaceCrop-002/2_ClipsforFaceCrop/<Scene>/<Emotion>/<clip>/*.jpg
            split membership in .../4_setups/All_scenes/{train_All,test_All}.csv
            format per line: "Action/Sad/0010 Sad"
"""

import cv2
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional

import config

# ─────────────────────────────────────────────────────────────
#  LABEL MAPS
# ─────────────────────────────────────────────────────────────

RAFDB_MAP = {
    1: "surprise", 2: "fear", 3: "disgust", 4: "happy",
    5: "sad", 6: "angry", 7: "neutral",
}

CKPLUS_FOLDER_MAP = {
    "anger": "angry", "angry": "angry", "contempt": None,
    "disgust": "disgust", "fear": "fear", "happy": "happy",
    "happiness": "happy", "neutral": "neutral", "sad": "sad",
    "sadness": "sad", "surprise": "surprise",
}

STOCK2FER_FOLDER_MAP = {
    "Angry": "angry", "Disgust": "disgust", "Fear": "fear", "Happy": "happy",
    "Neutral": "neutral", "Sad": "sad", "Surprise": "surprise",
}

CK48_FOLDER_MAP = {
    "anger": "angry", "angry": "angry", "fear": "fear", "happy": "happy",
    "sadness": "sad", "sad": "sad", "surprise": "surprise",
}

# DFEW's own label convention (per their EmoLabel README): 1-7 → our contract
DFEW_LABEL_MAP = {
    1: "happy", 2: "sad", 3: "neutral", 4: "angry",
    5: "surprise", 6: "disgust", 7: "fear",
}

# FERV39K split-file label words → our contract (same casing as stock2fer)
FERV39K_LABEL_MAP = {
    "Angry": "angry", "Disgust": "disgust", "Fear": "fear", "Happy": "happy",
    "Neutral": "neutral", "Sad": "sad", "Surprise": "surprise",
}


def one_hot(label_str: str) -> np.ndarray:
    v = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    idx = config.EMOTION_TO_IDX.get(label_str)
    if idx is not None:
        v[idx] = 1.0
    return v


def evenly_spaced_indices(total: int, k: int) -> list:
    """Pick k evenly-spaced indices out of `total` items (frame sampling)."""
    if total <= 0:
        return []
    if total <= k:
        return list(range(total))
    step = total / k
    return [int(i * step) for i in range(k)]


# ─────────────────────────────────────────────────────────────
#  BASE DATASET
# ─────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, transform=None, dataset_name: str = ""):
        self.samples      = []
        self.transform    = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, soft_label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(soft_label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
#  AFFECTNET  (YOLO format)
# ─────────────────────────────────────────────────────────────

def _load_affectnet_class_map() -> dict:
    if not config.AFFECTNET_YAML.exists():
        print(f"  [AffectNet] data.yaml not found at {config.AFFECTNET_YAML}")
        return {}
    with open(config.AFFECTNET_YAML) as f:
        yml = yaml.safe_load(f)
    names = yml.get("names", {})
    if isinstance(names, list):
        raw_map = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        raw_map = {int(k): v for k, v in names.items()}
    else:
        print("  [AffectNet] Unrecognised 'names' format in data.yaml")
        return {}
    result = {}
    for cid, name in raw_map.items():
        n = name.strip().lower()
        if n in ("anger", "angry"):          n = "angry"
        elif n in ("happiness", "happy"):    n = "happy"
        elif n in ("sadness", "sad"):        n = "sad"
        elif n in ("surprise", "surprised"): n = "surprise"
        if n in config.EMOTION_TO_IDX:
            result[cid] = n
    return result


class AffectNetDataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "affectnet")
        split_dir_map = {
            "train": config.AFFECTNET_TRAIN_DIR,
            "val":   config.AFFECTNET_VAL_DIR,
            "test":  config.AFFECTNET_TEST_DIR,
        }
        base_dir   = split_dir_map.get(split, config.AFFECTNET_TRAIN_DIR)
        imgs_dir   = base_dir / "images"
        labels_dir = base_dir / "labels"

        if not imgs_dir.exists():
            print(f"  [AffectNet] images/ not found at {imgs_dir} — skipping")
            return
        if not labels_dir.exists():
            print(f"  [AffectNet] labels/ not found at {labels_dir} — skipping")
            return

        class_map = _load_affectnet_class_map()
        if not class_map:
            print("  [AffectNet] empty class map — skipping")
            return

        loaded = 0
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            try:
                with open(label_path) as lf:
                    first_line = lf.readline().strip()
                if not first_line:
                    continue
                class_id = int(first_line.split()[0])
            except (ValueError, IndexError):
                continue
            emotion = class_map.get(class_id)
            if emotion is None:
                continue
            self.samples.append((img_path, one_hot(emotion)))
            loaded += 1

        print(f"  [AffectNet] {split}: {loaded} samples")


# ─────────────────────────────────────────────────────────────
#  CK+
# ─────────────────────────────────────────────────────────────

class CKPlusDataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "ckplus")
        root = config.CKPLUS_DIR
        if not root.exists():
            print(f"  [CK+]    Dir not found at {root} — skipping")
            return

        all_samples = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            label_str = CKPLUS_FOLDER_MAP.get(folder.name.lower())
            if label_str is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    all_samples.append((img_path, one_hot(label_str)))

        if not all_samples:
            print(f"  [CK+]    No images found in {root} — check folder names")
            return

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_samples))
        cut = int(0.85 * len(all_samples))
        chosen = idx[:cut] if split == "train" else idx[cut:]
        self.samples = [all_samples[i] for i in chosen]
        print(f"  [CK+]    {split}: {len(self.samples)} samples")


# ─────────────────────────────────────────────────────────────
#  FER+ — three sub-datasets
# ─────────────────────────────────────────────────────────────

class FERCk48Dataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "fer_ck48")
        root = config.FERPLUS_CK48_DIR
        if not root.exists():
            print(f"  [FER/CK48] Dir not found at {root} — skipping")
            return

        all_samples = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            label_str = CK48_FOLDER_MAP.get(folder.name.lower())
            if label_str is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    all_samples.append((img_path, one_hot(label_str)))

        if not all_samples:
            print(f"  [FER/CK48] No images found — skipping")
            return

        rng = np.random.RandomState(43)
        idx = rng.permutation(len(all_samples))
        cut = int(0.85 * len(all_samples))
        chosen = idx[:cut] if split == "train" else idx[cut:]
        self.samples = [all_samples[i] for i in chosen]
        print(f"  [FER/CK48] {split}: {len(self.samples)} samples")


class FERKaggle7Dataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "fer_k7")
        split_dir = config.FERPLUS_KAGGLE7_DIR / ("train" if split == "train" else "test")
        if not split_dir.exists():
            print(f"  [FER/K7]   Dir not found at {split_dir} — skipping")
            return

        loaded = 0
        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue
            label_str = CKPLUS_FOLDER_MAP.get(folder.name.lower())
            if label_str is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    self.samples.append((img_path, one_hot(label_str)))
                    loaded += 1

        print(f"  [FER/K7]   {split}: {loaded} samples")


class FERStockDataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "fer_stock")
        root = config.FERPLUS_STOCK_DIR
        if not root.exists():
            print(f"  [FER/Stock] Dir not found at {root} — skipping")
            return

        all_samples = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            label_str = STOCK2FER_FOLDER_MAP.get(folder.name) or \
                        CKPLUS_FOLDER_MAP.get(folder.name.lower())
            if label_str is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    all_samples.append((img_path, one_hot(label_str)))

        if not all_samples:
            print(f"  [FER/Stock] No images found — skipping")
            return

        rng = np.random.RandomState(44)
        idx = rng.permutation(len(all_samples))
        cut = int(0.85 * len(all_samples))
        chosen = idx[:cut] if split == "train" else idx[cut:]
        self.samples = [all_samples[i] for i in chosen]
        print(f"  [FER/Stock] {split}: {len(self.samples)} samples")


# ─────────────────────────────────────────────────────────────
#  RAF-DB
# ─────────────────────────────────────────────────────────────

class RAFDBDataset(EmotionDataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__(transform, "rafdb")
        subfolder = "train" if split == "train" else "test"
        split_dir = config.RAFDB_IMGS / subfolder

        if not split_dir.exists():
            print(f"  [RAF-DB]  {subfolder}/ not found at {split_dir} — skipping")
            return

        loaded = 0
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            try:
                label_int = int(class_folder.name)
            except ValueError:
                continue
            label_str = RAFDB_MAP.get(label_int)
            if label_str is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in class_folder.glob(ext):
                    self.samples.append((img_path, one_hot(label_str)))
                    loaded += 1

        print(f"  [RAF-DB]  {split}: {loaded} samples")


# ─────────────────────────────────────────────────────────────
#  DFEW  (NEW) — frame-sampled from 16-frame clips
# ─────────────────────────────────────────────────────────────

class DFEWFramesDataset(EmotionDataset):
    """
    Reads data/DFEW/EmoLabel_DataSplit/{train,test}/set_1.csv for clip_id -> label,
    then samples config.FRAMES_PER_CLIP evenly-spaced frames per clip from
    data/DFEW/Clip/clip_224x16f/<5-digit clip_id>/{1..16}.jpg (not zero-padded).

    "val" split maps to DFEW's own "test" csv (DFEW has no separate val folder).
    """
    def __init__(self, split: str = "train", transform=None, frames_per_clip: Optional[int] = None):
        super().__init__(transform, "dfew")
        frames_per_clip = frames_per_clip or config.FRAMES_PER_CLIP

        csv_split = "train" if split == "train" else "test"
        csv_path  = config.DFEW_SPLIT_ROOT / csv_split / f"{config.DFEW_FOLD}.csv"

        if not csv_path.exists():
            print(f"  [DFEW]   split csv not found at {csv_path} — skipping")
            return
        if not config.DFEW_CLIP_ROOT.exists():
            print(f"  [DFEW]   clip root not found at {config.DFEW_CLIP_ROOT} — skipping")
            return

        df = pd.read_csv(csv_path)
        loaded, clips_used = 0, 0

        for _, row in df.iterrows():
            try:
                clip_id   = int(row["video_name"])
                label_int = int(row["label"])
            except (ValueError, KeyError):
                continue

            label_str = DFEW_LABEL_MAP.get(label_int)
            if label_str is None:
                continue

            clip_dir = config.DFEW_CLIP_ROOT / f"{clip_id:05d}"
            if not clip_dir.exists():
                continue

            frame_paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if not frame_paths:
                continue

            for i in evenly_spaced_indices(len(frame_paths), frames_per_clip):
                self.samples.append((frame_paths[i], one_hot(label_str)))
                loaded += 1
            clips_used += 1

        print(f"  [DFEW]   {split}: {loaded} frames from {clips_used} clips "
              f"(fold={config.DFEW_FOLD}, {frames_per_clip}/clip)")


# ─────────────────────────────────────────────────────────────
#  FERV39K  (NEW) — frame-sampled, folder labels + official split csv
# ─────────────────────────────────────────────────────────────

class FERV39KFramesDataset(EmotionDataset):
    """
    Reads data/FERV39k/.../4_setups/All_scenes/{train_All,test_All}.csv
    Each line: "Action/Sad/0010 Sad"  →  relative clip path + label word.
    Frames live at data/FERV39k/.../2_ClipsforFaceCrop/<rel_path>/*.jpg
    (zero-padded 2-digit names, variable count per clip).
    """
    def __init__(self, split: str = "train", transform=None, frames_per_clip: Optional[int] = None):
        super().__init__(transform, "ferv39k")
        frames_per_clip = frames_per_clip or config.FRAMES_PER_CLIP

        csv_name = "train_All.csv" if split == "train" else "test_All.csv"
        csv_path = config.FERV_SETUP_ROOT / csv_name

        if not csv_path.exists():
            print(f"  [FERV39K] split file not found at {csv_path} — skipping")
            return
        if not config.FERV_FACE_ROOT.exists():
            print(f"  [FERV39K] face-crop root not found at {config.FERV_FACE_ROOT} — skipping")
            return

        loaded, clips_used = 0, 0
        with open(csv_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rel_path, label_word = line.rsplit(" ", 1)
                except ValueError:
                    continue

                label_str = FERV39K_LABEL_MAP.get(label_word)
                if label_str is None:
                    continue

                clip_dir = config.FERV_FACE_ROOT / rel_path
                if not clip_dir.exists():
                    continue

                frame_paths = sorted(clip_dir.glob("*.jpg"))
                if not frame_paths:
                    continue

                for i in evenly_spaced_indices(len(frame_paths), frames_per_clip):
                    self.samples.append((frame_paths[i], one_hot(label_str)))
                    loaded += 1
                clips_used += 1

        print(f"  [FERV39K] {split}: {loaded} frames from {clips_used} clips "
              f"({frames_per_clip}/clip)")


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str):
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.45, p=0.6),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=25, p=0.4),
            A.OneOf([
                A.GridDistortion(num_steps=4, distort_limit=0.15, p=1.0),
                A.ElasticTransform(alpha=30, sigma=5, p=1.0),
                A.Perspective(scale=(0.03, 0.08), p=1.0),
            ], p=0.35),
            A.GaussNoise(p=0.25),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.Sharpen(alpha=(0.1, 0.3), p=1.0),
            ], p=0.25),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                p=0.35,
            ),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────
#  BUILD COMBINED DATASET + WEIGHTED SAMPLER
# ─────────────────────────────────────────────────────────────

def build_dataset(split: str = "train") -> ConcatDataset:
    tf = get_transforms(split)

    parts = [
        AffectNetDataset(split, tf),
        CKPlusDataset(split,    tf),
        FERCk48Dataset(split,   tf),
        FERKaggle7Dataset(split,tf),
        FERStockDataset(split,  tf),
        RAFDBDataset(split,     tf),
        DFEWFramesDataset(split,    tf),   # NEW
        FERV39KFramesDataset(split, tf),   # NEW
    ]

    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise RuntimeError(
            "No datasets loaded — check your data/ paths in config.py"
        )

    combined = ConcatDataset(parts)
    print(f"\n  Total {split}: {len(combined)} samples from {len(parts)} datasets\n")
    return combined


def build_weighted_sampler(dataset: ConcatDataset) -> WeightedRandomSampler:
    from collections import Counter

    dataset_names = []
    label_indices = []

    for ds in dataset.datasets:
        for _, soft in ds.samples:
            dataset_names.append(ds.dataset_name)
            label_indices.append(int(np.argmax(soft)))

    n = len(label_indices)
    counts = Counter(label_indices)
    cls_w  = {c: n / (config.NUM_CLASSES * cnt) for c, cnt in counts.items()}
    ds_w   = config.DATASET_OVERSAMPLE

    extra_w = {
        config.EMOTION_TO_IDX[e]: w
        for e, w in config.CLASS_EXTRA_WEIGHT.items()
        if e in config.EMOTION_TO_IDX
    }

    weights = np.array([
        cls_w[label_indices[i]]
        * ds_w.get(dataset_names[i], 1.0)
        * extra_w.get(label_indices[i], 1.0)
        for i in range(n)
    ], dtype=np.float64)

    mean_w = weights.mean()
    cap = config.SAMPLE_WEIGHT_CAP_MULTIPLIER * mean_w
    n_capped = int((weights > cap).sum())
    weights = np.minimum(weights, cap)
    print(f"  Weight cap: {cap:.2f} ({config.SAMPLE_WEIGHT_CAP_MULTIPLIER}x mean) — capped {n_capped} samples")

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=n,
        replacement=True,
    )

    
