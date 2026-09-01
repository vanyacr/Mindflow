"""
MindFlow - Phase 1 Data Pipeline
Central configuration: paths, audio standard, unified label schema.
"""

from pathlib import Path

# Resolve base project root dynamically (e.g. D:\Capstone116_Vaish or E:\Capstone116_Vaish)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (_PROJECT_ROOT / "Audio").exists():
    # Fallback to D: drive or E: drive if running from nested folder
    for drive in ["D:", "E:", "C:"]:
        candidate = Path(f"{drive}/Capstone116_Vaish")
        if (candidate / "Audio").exists():
            _PROJECT_ROOT = candidate
            break

DATASETS_ROOT = _PROJECT_ROOT / "Audio" / "Datasets"
PROCESSED_ROOT = _PROJECT_ROOT / "Audio" / "Processed"
if not PROCESSED_ROOT.exists():
    PROCESSED_ROOT = _PROJECT_ROOT / "Audio" / "processed"

RAW_DIRS = {
    "CREMA-D": DATASETS_ROOT / "Crema-D",
    "RAVDESS": DATASETS_ROOT / "RAVDESS",
    "SAVEE":   DATASETS_ROOT / "SAVEE",
    "TESS":    DATASETS_ROOT / "TESS",
    "MELD":    DATASETS_ROOT / "MELD" / "MELD-RAW" / "MELD.Raw",
    "DAIC-WOZ": DATASETS_ROOT / "DAIC",
    "IEMOCAP": DATASETS_ROOT / "Iecomap" / "IEMOCAP_full_release",
}

PROCESSED_AUDIO_DIR = PROCESSED_ROOT / "audio"
METADATA_DIR = PROCESSED_ROOT / "metadata"
LABELS_DIR = PROCESSED_ROOT / "labels"
TRANSCRIPTS_DIR = PROCESSED_ROOT / "transcripts"

for d in (PROCESSED_AUDIO_DIR, METADATA_DIR, LABELS_DIR, TRANSCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_DTYPE = "float32"
VAD_AGGRESSIVENESS = 2
TRIM_TOP_DB = 30

UNIFIED_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fear",
    "neutral",
    "surprise",
    "disgust",
]

CATEGORICAL_EMOTION_DATASETS = ["CREMA-D", "RAVDESS", "SAVEE", "TESS", "MELD"]
SPEAKER_DISJOINT_INCOMPATIBLE = ["TESS"]
