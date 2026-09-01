"""
config_temporal.py — Track B (BiGRU) configuration.

Kept separate from config.py so Track A is never touched by Track B work.

Known lessons baked in from the paused overfitting investigation:
  - Uncapped class weights previously gave disgust an 11.5x multiplier
    (only 116 DFEW training samples for that class in set_1).
    FIX: CLASS_WEIGHT_CAP = 4.0
  - Label smoothing added to soften hard-label overfitting on tiny classes.
  - Lower LR + higher weight decay vs a from-scratch temporal CNN, since
    we're only training a BiGRU head on frozen EfficientNet-B2 embeddings.
"""

from pathlib import Path
import sys

# reuse Track A's static config for image size, normalization, emotion contract
sys.path.insert(0, str(Path(__file__).parent))
import config as static_config

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  EMOTION CONTRACT — same 7-class order as Track A, shared contract
# ─────────────────────────────────────────────────────────────
EMOTIONS       = static_config.EMOTIONS
EMOTION_TO_IDX = static_config.EMOTION_TO_IDX
IDX_TO_EMOTION = static_config.IDX_TO_EMOTION
NUM_CLASSES    = static_config.NUM_CLASSES

IMAGE_SIZE     = static_config.IMAGE_SIZE
IMAGENET_MEAN  = static_config.IMAGENET_MEAN
IMAGENET_STD   = static_config.IMAGENET_STD

# ─────────────────────────────────────────────────────────────
#  DATA PATHS
# ─────────────────────────────────────────────────────────────

# DFEW
DFEW_ROOT        = Path("D:/video/DFEW/data/DFEW")
DFEW_CLIP_DIR    = DFEW_ROOT / "Clip" / "clip_224x16f"
DFEW_SPLIT_DIR   = DFEW_ROOT / "EmoLabel_DataSplit"
DFEW_FOLD        = "set_1"   # which of the 5 official CV folds to use

# Standard DFEW paper label convention (1-indexed).
# NOTE: not yet cross-checked against annotation.xlsx — flagged for a
# later spot check, but this is the widely-used convention.
DFEW_LABEL_MAP = {
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "angry",
    5: "surprise",
    6: "disgust",
    7: "fear",
}

# FERV39K
FERV_ROOT        = Path("D:/video/FERV39K")
FERV_CLIP_DIR    = FERV_ROOT / "extracted_faces" / "2_ClipsforFaceCrop"
FERV_SPLIT_DIR   = FERV_ROOT / "drive-download-20260721T053535Z-1-003" / "4_setups" / "22_scenes"
FERV_SCENES = [
    "Action", "Argue", "Business", "Conflict", "Contest", "Crime", "Crisis",
    "DailyLife", "ElegantArt", "Experiment", "History", "Interview",
    "Liveshow", "Medicine", "OfficialEvent", "ScholarReport", "School",
    "Social", "Speech", "Talkshow", "Terror", "War",
]
# FERV39K folder names are already contract emotion names, just wrong case
FERV_FOLDER_MAP = {
    "Angry":    "angry",
    "Disgust":  "disgust",
    "Fear":     "fear",
    "Happy":    "happy",
    "Neutral":  "neutral",
    "Sad":      "sad",
    "Surprise": "surprise",
}

# ─────────────────────────────────────────────────────────────
#  SEQUENCE / TEMPORAL SETTINGS
# ─────────────────────────────────────────────────────────────
SEQ_LEN = 16   # matches DFEW's native 16 frames/clip; FERV39K frames are
               # uniformly sampled down/up to this count

# ─────────────────────────────────────────────────────────────
#  TRAINING HYPERPARAMETERS  — tuned to avoid the prior overfit
# ─────────────────────────────────────────────────────────────
BATCH_SIZE      = 16      # sequences per batch (each is SEQ_LEN frames)
EPOCHS          = 40
LR              = 3e-4    # only training GRU + head, backbone frozen
WEIGHT_DECAY    = 5e-4    # raised from typical 2e-4 — extra regularisation
LABEL_SMOOTHING = 0.10    # softens hard-label overfitting on tiny classes
GRU_HIDDEN      = 128
GRU_LAYERS      = 1
HEAD_DROPOUT    = 0.4

# Class weight cap — hard lesson from the previous overfit run where
# disgust (116 samples) got an 11.5x multiplier. Never exceed this.
CLASS_WEIGHT_CAP = 4.0

STATIC_CKPT = static_config.CKPT_PATH   # frozen EfficientNet-B2 backbone source

CKPT_PATH_FROZEN      = CKPT_DIR / "best_model_temporal_frozen.pt"
CKPT_PATH_UNFROZEN    = CKPT_DIR / "best_model_temporal_unfrozen.pt"
CKPT_PATH_TRANSFORMER = CKPT_DIR / "best_model_temporal_transformer.pt"
CKPT_PATH             = CKPT_PATH_FROZEN   # default is the proven frozen backbone model
