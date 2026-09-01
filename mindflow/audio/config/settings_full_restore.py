"""
MindFlow - Phase 1 Data Pipeline
Central configuration: paths, audio standard, unified label schema.

Edit DATASETS_ROOT and PROCESSED_ROOT to match your machine
(you're on an RTX 4090 box with no storage constraints, so these
can just point at local disk).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — set to your actual machine (Windows, E: drive, per your screenshot)
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path(r"E:\Capstone116_Vaish\Audio\Datasets")
PROCESSED_ROOT = Path(r"E:\Capstone116_Vaish\Audio\processed")

RAW_DIRS = {
    "CREMA-D": DATASETS_ROOT / "Crema-D",   # matches your actual folder name
    "RAVDESS": DATASETS_ROOT / "RAVDESS",
    "SAVEE":   DATASETS_ROOT / "SAVEE",
    "TESS":    DATASETS_ROOT / "TESS",
    "MELD":    DATASETS_ROOT / "MELD" / "MELD-RAW" / "MELD.Raw",  # confirmed nested path
    "DAIC-WOZ": DATASETS_ROOT / "DAIC",     # matches your actual folder name
    "IEMOCAP": DATASETS_ROOT / "Iecomap" / "IEMOCAP_full_release",
}

PROCESSED_AUDIO_DIR = PROCESSED_ROOT / "audio"
METADATA_DIR = PROCESSED_ROOT / "metadata"
LABELS_DIR = PROCESSED_ROOT / "labels"
TRANSCRIPTS_DIR = PROCESSED_ROOT / "transcripts"

for d in (PROCESSED_AUDIO_DIR, METADATA_DIR, LABELS_DIR, TRANSCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio standard (Phase 1, step 2)
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1  # mono
TARGET_DTYPE = "float32"
VAD_AGGRESSIVENESS = 2  # webrtcvad: 0 (least aggressive) - 3 (most aggressive)
TRIM_TOP_DB = 30  # librosa silence trim threshold

# ---------------------------------------------------------------------------
# Unified emotion label schema
# ---------------------------------------------------------------------------
UNIFIED_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fear",
    "neutral",
    "surprise",
    "disgust",
]

# Datasets whose native labels map cleanly onto UNIFIED_EMOTIONS.
# DAIC-WOZ is deliberately excluded here — it carries PHQ-8 depression/stress
# scores, not discrete emotion categories, so it's handled by a separate
# metadata builder (see metadata/build_daic_metadata.py) and consumed at the
# stress-regression stage, not folded into the 7-way emotion label set.
CATEGORICAL_EMOTION_DATASETS = ["CREMA-D", "RAVDESS", "SAVEE", "TESS", "MELD"]

# TESS uses only 2 actresses -> not viable for speaker-disjoint splits.
# Flagged here so downstream split code can force TESS into train-only.
SPEAKER_DISJOINT_INCOMPATIBLE = ["TESS"]
