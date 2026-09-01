"""
config.py — full rebuild. Paths verified against your actual disk layout
via inspect_datasets.py / inspect_datasets_deep.py (Aug 2026).

Actual layout confirmed on disk:
  data/
  ├── AffectNet/YOLO_format/
  │   ├── train/images/  train/labels/
  │   ├── valid/images/  valid/labels/
  │   ├── test/images/   test/labels/
  │   └── data.yaml
  ├── CK+/                          ← confirmed WITH the '+' (not "CK")
  │   └── anger/ contempt/ disgust/ fear/ happy/ sadness/ surprise/
  ├── FER+/                         ← confirmed WITH the '+' (not "FER")
  │   ├── CK+48/
  │   ├── kaggle3/                  ← still skipped, only 3 emotions
  │   ├── kaggle7/{train,test}/
  │   └── stock2fer/
  ├── RAF-DB/
  │   └── DATASET/{train,test}/{1..7}/
  ├── DFEW/
  │   ├── Clip/clip_224x16f/<clip_id 5-digit>/{1.jpg..16.jpg}   (not zero-padded)
  │   └── EmoLabel_DataSplit/{train,test}/set_1.csv..set_5.csv  (video_name,label)
  └── FERV39k/
      ├── 2_ClipsforFaceCrop-002/2_ClipsforFaceCrop/<Scene>/<Emotion>/<clip 4-digit>/*.jpg
      └── drive-download-20260721T053535Z-1-003/4_setups/All_scenes/{train_All.csv,test_All.csv}

MAFW is intentionally excluded — archives confirmed broken/partial, links expired.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  DATA_DIR — points at your actual datasets, wherever the code lives.
#
#  Code runs from D:\video\files (BASE_DIR). Datasets physically live
#  under D:\archive\code\code\data via your existing junctions.
#  If you ever move the datasets, this is the ONE line to change.
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path(r"D:\archive\code\code\data")

# ─────────────────────────────────────────────────────────────
#  STATIC IMAGE DATASETS  (unchanged from before, paths verified)
# ─────────────────────────────────────────────────────────────

AFFECTNET_DIR       = DATA_DIR / "AffectNet" / "YOLO_format"
AFFECTNET_YAML      = AFFECTNET_DIR / "data.yaml"
AFFECTNET_TRAIN_DIR = AFFECTNET_DIR / "train"
AFFECTNET_VAL_DIR   = AFFECTNET_DIR / "valid"
AFFECTNET_TEST_DIR  = AFFECTNET_DIR / "test"

CKPLUS_DIR = DATA_DIR / "CK+"

FERPLUS_DIR         = DATA_DIR / "FER+"
FERPLUS_CK48_DIR    = FERPLUS_DIR / "CK+48"
FERPLUS_KAGGLE7_DIR = FERPLUS_DIR / "kaggle7"
FERPLUS_STOCK_DIR   = FERPLUS_DIR / "stock2fer"

RAFDB_DIR       = DATA_DIR / "RAF-DB"
RAFDB_IMGS      = RAFDB_DIR / "DATASET"
RAFDB_TRAIN_CSV = RAFDB_DIR / "train_labels.csv"   # not used — folder number IS the label
RAFDB_TEST_CSV  = RAFDB_DIR / "test_labels.csv"

# ─────────────────────────────────────────────────────────────
#  VIDEO DATASETS (NEW) — sampled as frames for the static Track A model
#
#  FRAMES_PER_CLIP controls how many frames get pulled from each clip.
#  Kept low (3) deliberately: DFEW/FERV39K together have tens of thousands
#  of clips, and pulling too many near-duplicate frames per clip would
#  drown out AffectNet/CK+/FER+/RAF-DB's diversity within each class.
# ─────────────────────────────────────────────────────────────

FRAMES_PER_CLIP = 3

DFEW_DIR         = Path(r"D:\video\DFEW\data\DFEW")
DFEW_CLIP_ROOT   = DFEW_DIR / "Clip" / "clip_224x16f"
DFEW_SPLIT_ROOT  = DFEW_DIR / "EmoLabel_DataSplit"
DFEW_FOLD        = "set_1"   # DFEW ships 5 CV folds — using fold 1 as a single train/test split

FERV_DIR         = Path(r"D:\video\FERV39K")
FERV_FACE_ROOT   = FERV_DIR / "extracted_faces" / "2_ClipsforFaceCrop"
# NOTE: this folder name is a literal Google Drive export timestamp — fragile.
# If you ever re-download FERV39K this path will need updating.
FERV_SETUP_ROOT  = FERV_DIR / "drive-download-20260721T053535Z-1-003" / "4_setups" / "All_scenes"

# ─────────────────────────────────────────────────────────────
#  EMOTION CONTRACT — fixed 7-class order shared by all loaders
# ─────────────────────────────────────────────────────────────
EMOTIONS       = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}
NUM_CLASSES    = 7

# ─────────────────────────────────────────────────────────────
#  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE   = 160
BATCH_SIZE   = 96      # ↑ from 48 — you're on a local RTX 4090 (24GB), not the
                        # Kaggle T4 (16GB) these settings were originally tuned for.
                        # Combined with AMP in train.py this should still fit comfortably.
                        # Drop to 48-64 if you hit OOM.
EPOCHS       = 60
LR           = 1e-4
WEIGHT_DECAY = 2e-4

# Dataset-level oversampling weights — applied on top of per-class balancing.
# Tiny datasets get boosted so they aren't drowned out; DFEW/FERV39K get
# throttled DOWN because raw clip counts (15.9K and ~39K) would otherwise
# dominate every class purely by volume once frame-sampled.
#   ferv39k=0.3: ~39K clips × 3 frames ≈ 117K frames — largest single source
#                by far, needs the heaviest throttling
#   dfew=0.6:    ~15.9K clips × 3 frames ≈ 48K frames — still throttled, less severe
DATASET_OVERSAMPLE = {
    "affectnet": 1.0,
    "ckplus":    8.0,
    "fer_ck48":  6.0,
    "fer_k7":    2.0,
    "fer_stock": 4.0,
    "rafdb":     4.0,
    "dfew":      0.6,   # NEW
    "ferv39k":   0.3,   # NEW
}

SAMPLE_WEIGHT_CAP_MULTIPLIER = 4.0   # combined weight cannot exceed 4x the pool mean
# Per-class additional weights — applied on top of dataset weights AND on top
# of the sampler's automatic inverse-class-frequency weight (cls_w).
#
# IMPORTANT: these values were re-derived after the first DFEW/FERV39K retrain
# regressed fear (-4.0pts) and disgust (-5.9pts) despite adding more data for
# both. Root cause: combining datasets shrank fear/disgust's SHARE of the
# total pool, which increased their automatic cls_w — stacking the old
# extra_w on top pushed effective sampling weight to fear=3.72x, disgust=4.07x,
# right past the ≤4x ceiling already established from the DFEW temporal
# training disgust-overfitting lesson. Meanwhile angry/neutral got quietly
# diluted (effective weight dropped ~25%) since they didn't grow as fast as
# happy/surprise in the combined pool.
#
# Values below target effective weight (cls_w * extra_w) close to what
# produced the known-good 74.3% static baseline, using the ACTUAL new cls_w
# from the combined 8-source pool (see verify_setup.py class distribution):
#   fear:    cls_w=1.859 -> extra_w=1.3 -> effective 2.42x  (was 3.72x)
#   disgust: cls_w=2.714 -> extra_w=1.2 -> effective 3.26x  (was 4.07x, safely under 4x cap)
#   angry:   cls_w=0.846 -> extra_w=1.6 -> effective 1.35x  (restores diluted weight)
#   neutral: cls_w=0.657 -> extra_w=2.0 -> effective 1.31x  (restores diluted weight)
#   sad:     cls_w=0.852 -> extra_w=2.2 -> effective 1.87x  (minor restoration)
#   surprise:cls_w=1.363 -> extra_w=0.8 -> effective 1.09x  (was overshooting at 1.0)
CLASS_EXTRA_WEIGHT = {
    "happy":    1.0,
    "sad":      2.2,
    "angry":    1.6,
    "neutral":  2.0,
    "fear":     1.3,
    "disgust":  1.2,
    "surprise": 0.8,
}

CKPT_PATH     = CKPT_DIR / "best_model.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

