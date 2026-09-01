"""
Training configuration.

Stage plan follows the project roadmap, with ONE deliberate change from the
original plan: DAIC-WOZ is NOT folded into Stage 2 emotion fine-tuning,
because it carries PHQ-8 depression/stress scores, not the 7-class emotion
schema (see metadata/dataset_scanners.py::scan_daic_woz and the pipeline
README). It gets its own regression stage instead — see STRESS_STAGE below.
If you've decided how you actually want to reconcile that, edit the stage
lists here; nothing else in the training code needs to change.
"""

from pathlib import Path

from config.settings import PROCESSED_ROOT, METADATA_DIR, UNIFIED_EMOTIONS

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
WAVLM_CHECKPOINT = "microsoft/wavlm-large"
NUM_CLASSES = len(UNIFIED_EMOTIONS)       # 7
LABEL2IDX = {label: i for i, label in enumerate(UNIFIED_EMOTIONS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}

ATTENTION_POOL_DIM = 256   # hidden size of the attention pooling projection
CLASSIFIER_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Audio windowing
# ---------------------------------------------------------------------------
MAX_AUDIO_SECONDS = 6.0        # clips longer than this get randomly cropped
SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
BATCH_SIZE = 16                 # WavLM Large is big; raise cautiously on a 4090
GRAD_ACCUM_STEPS = 2             # effective batch size = BATCH_SIZE * this
LEARNING_RATE_HEAD = 1e-4        # classifier head
LEARNING_RATE_BACKBONE = 1e-5    # WavLM backbone (much lower — it's pretrained)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = {
    "stage1": 15,
    "stage2": 8,
    "stage3": 8,
    "stress": 10,
}
EARLY_STOPPING_PATIENCE = 4
GRAD_CLIP_NORM = 1.0
FP16 = True                      # mixed precision — worthwhile on a 4090

# ---------------------------------------------------------------------------
# Stage definitions (dataset name -> which metadata_*.csv to pull from)
# ---------------------------------------------------------------------------
STAGE_1_DATASETS = ["CREMA-D", "RAVDESS", "SAVEE", "TESS"]   # acted baseline
STAGE_2_DATASETS = ["MELD"]                                    # naturalistic, still 7-class
STAGE_3_DATASETS = ["IEMOCAP", "MSP-IMPROV"]                   # once access arrives

# Separate track: DAIC-WOZ stress/depression regression (not emotion classes)
STRESS_STAGE_DATASET = "DAIC-WOZ"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_SEED = 42
# TESS has only 2 speakers -> cannot support a disjoint val/test split.
FORCE_TRAIN_ONLY_DATASETS = ["TESS"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_DIR = METADATA_DIR / "splits"
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")
EXPORT_DIR = Path("./exported_models")

for d in (SPLITS_DIR, CHECKPOINT_DIR, LOG_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"  # you're on an RTX 4090 — change to "cpu" only for debugging
