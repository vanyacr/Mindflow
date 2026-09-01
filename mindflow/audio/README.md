# MindFlow — Phase 1 Data Pipeline (Audio)

Covers Coding Roadmap Modules 1–4: dataset extraction, audio preprocessing,
metadata generation, and label mapping — the foundation the rest of the
audio pipeline (WavLM training, fusion) builds on.

## Setup

```bash
pip install -r requirements.txt
```

Point `DATASETS_ROOT` in `config/settings.py` at wherever your raw dataset
folders actually live, e.g.:

```
datasets/
├── CREMA-D/        # AudioWAV/*.wav + VideoDemographics.csv
├── RAVDESS/         # Actor_01/*.wav ... Actor_24/*.wav
├── SAVEE/           # DC_*.wav, JE_*.wav, JK_*.wav, KL_*.wav
├── TESS/            # OAF_angry/*.wav, YAF_happy/*.wav, ...
├── MELD/            # train_sent_emo.csv + audio/{split}/dia{d}_utt{u}.wav
└── DAIC-WOZ/        # {pid}_P/{pid}_AUDIO.wav, *split*.csv (PHQ8 labels)
```

## Run

```bash
# Process what you have downloaded right now:
python scripts/run_phase1.py --datasets CREMA-D RAVDESS SAVEE TESS

# Dry run (metadata only, skip the (slower) audio standardization pass):
python scripts/run_phase1.py --datasets CREMA-D RAVDESS SAVEE TESS --skip-standardize

# Once MELD is downloaded and DAIC-WOZ finishes syncing:
python scripts/run_phase1.py --datasets all

# Then generate the distribution plots:
python scripts/analyze_metadata.py
```

## Output

```
processed/
├── audio/{DatasetName}/*.wav      # 16kHz mono, normalized, VAD-trimmed
└── metadata/
    ├── metadata.csv                # combined, unified-label emotion set
    ├── metadata_crema_d.csv        # per-dataset
    ├── metadata_ravdess.csv
    ├── metadata_savee.csv
    ├── metadata_tess.csv
    ├── metadata_meld.csv
    ├── metadata_daic_woz.csv       # separate schema: PHQ8 score, not emotion
    └── plots/*.png
```

## Notes specific to your setup

- **DAIC-WOZ is handled separately.** It carries PHQ-8 depression/stress
  scores, not the 7-class emotion labels, so it's excluded from
  `metadata.csv` and written to its own file. Per your Phase 4 plan it
  feeds Stage 2 fine-tuning on the stress-regression side, not the
  emotion classifier directly — decide how you want to reconcile that
  with a single WavLM head before Stage 2 training.
- **TESS is train-only.** Only 2 actresses (OAF/YAF) means it can't
  support a speaker-disjoint split; the scanner tags speaker as
  `OAF`/`YAF` so your split logic can filter on it directly.
- **MELD scanner expects extracted audio.** MELD ships utterance-level
  `.mp4` clips, not `.wav`. You'll need an `ffmpeg` extraction pass
  (`ffmpeg -i dia{d}_utt{u}.mp4 -ar 16000 -ac 1 dia{d}_utt{u}.wav`) before
  `scan_meld` will find anything — happy to write that extraction script
  too when you're ready to download MELD.
- **IEMOCAP / MSP-IMPROV** aren't wired in yet since access is still
  pending — add a `scan_iemocap` / `scan_msp_improv` function to
  `metadata/dataset_scanners.py` and an entry in `label_maps.py` once
  the datasets land; the rest of the pipeline (standardization, CSV
  writing, plots) needs zero changes.
