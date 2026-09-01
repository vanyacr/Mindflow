# MindFlow Audio Pipeline — Setup & Run Guide (Windows / RTX 4090)

Matches your actual folder layout:
`E:\Capstone116_Vaish\Audio\Datasets\{Crema-D, DAIC, MELD, RAVDESS, SAVEE, TESS}`

Config is already pointed at this — see `config/settings.py`.

---

## 1. Tools to install (one-time)

| Tool | Why | How |
|---|---|---|
| **Python 3.10 or 3.11 (64-bit)** | everything runs on this | python.org installer — check "Add to PATH" during install |
| **NVIDIA driver** (recent) | your RTX 4090 needs a current driver for CUDA 12.x | `nvidia-smi` in PowerShell should show your GPU + driver version; update via GeForce Experience or nvidia.com if old |
| **CUDA-enabled PyTorch** | training runs on GPU, not CPU | see step 3 below — do NOT `pip install torch` plain, it grabs a CPU-only build |
| **ffmpeg** | MELD ships video clips; you need to pull audio out | `winget install ffmpeg`, then restart your terminal and confirm with `ffmpeg -version` |
| **VS Code** (already using) | editing/running scripts | you have this |

## 2. Project setup

```powershell
cd E:\Capstone116_Vaish\Audio
# unzip the mindflow_pipeline.zip you were given here, so you have:
#   E:\Capstone116_Vaish\Audio\mindflow_pipeline\

cd mindflow_pipeline
python -m venv venv
.\venv\Scripts\activate
```

## 3. Install dependencies

```powershell
# Install the CUDA build of PyTorch FIRST (before requirements.txt), matched
# to your driver. cu121 works for most current drivers — check
# https://pytorch.org/get-started/locally/ if unsure.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then everything else
pip install -r requirements.txt
```

Verify the GPU is actually visible to PyTorch:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This must print `True` and `NVIDIA GeForce RTX 4090` before you go further —
if it prints `False`, the CUDA wheel didn't match your driver; don't proceed
to training until this is fixed, or it'll silently run on CPU and take days
instead of hours.

## 4. Verify each dataset folder's internal structure

The scanners expect specific layouts inside each top-level folder. Open each
one and confirm:

- **Crema-D**: a folder of `.wav` files like `1001_DFA_ANG_XX.wav`, plus
  ideally `VideoDemographics.csv` in the same root (for gender labels — not
  required, but you lose the gender column without it).
- **RAVDESS**: `Actor_01/` ... `Actor_24/` subfolders, each full of
  `03-01-06-...wav` style files.
- **SAVEE**: flat folder of files like `DC_a1.wav`, `JE_sa3.wav`.
- **TESS**: subfolders like `OAF_angry/`, `YAF_happy/`, each full of `.wav`.
- **MELD**: needs `train_sent_emo.csv`, `dev_sent_emo.csv`,
  `test_sent_emo.csv` at the root, plus a folder of `.mp4` clips per split
  (folder name varies by release — commonly `train_splits/`,
  `dev_splits_complete/`, `output_repeated_splits_test/`).
- **DAIC**: participant folders named `{id}_P/` (e.g. `300_P/`), each
  containing `{id}_AUDIO.wav`, plus a split label CSV at the root (e.g.
  `train_split_Depression_AVEC2017.csv`) with a `PHQ8_Score` column.

If any of these don't match what you actually see, tell me the real
structure and I'll adjust the relevant scanner in
`metadata/dataset_scanners.py` — don't try to rename thousands of files by
hand to match the script.

## 5. Run the pipeline, in order

```powershell
# A. Extract audio from MELD's video clips (one-time, only needed for MELD)
python scripts\extract_meld_audio.py

# B. Run Phase 1 on everything you have — scans, standardizes, labels, writes metadata.csv
python scripts\run_phase1.py --datasets CREMA-D RAVDESS SAVEE TESS MELD DAIC-WOZ

# C. Sanity-check the result before spending hours training on it
python scripts\analyze_metadata.py
```
Open `processed\metadata\plots\*.png` and actually look at them — check the
emotion distribution isn't wildly broken (e.g. one class with 5 clips) and
the duration histogram doesn't have a spike at 0 seconds (a sign VAD ate
entire clips somewhere).

```powershell
# D. Build speaker-disjoint train/val/test splits
python data\splits.py

# E. Train Stage 1 (acted baseline) — start here, don't skip to Stage 2
python training\train.py --stage stage1
```

Watch the first epoch closely. `train_acc` climbing above chance (~14% for
7 classes) within the first epoch and `val_acc` not immediately collapsing
to 0 both mean the pipeline end-to-end is wired correctly. If GPU memory
errors out, lower `BATCH_SIZE` in `config\train_config.py` (16 → 8) before
anything else.

```powershell
# F. Once Stage 1 finishes and you're happy with val_acc, fine-tune on MELD
python training\train.py --stage stage2 --resume checkpoints\stage1_best.pt

# G. Evaluate properly (accuracy, F1, confusion matrix, ROC — not just val_acc)
python training\evaluate.py --checkpoint checkpoints\stage2_best.pt

# H. DAIC-WOZ stress regression, separate from the emotion classifier
python training\train_stress.py --resume checkpoints\stage2_best.pt

# I. Export for the fusion team
python training\export.py --checkpoint checkpoints\stage2_best.pt --formats pt onnx
```

## 6. What NOT to do

- Don't run Stage 2 before confirming Stage 1's val_acc looks reasonable —
  fine-tuning a broken baseline just gives you a broken fine-tuned model.
- Don't skip step C (the plots). A silent labeling bug in one dataset is
  much cheaper to catch as a weird bar chart than after a 6-hour training run.
- Don't train on CPU by accident — recheck step 3's `torch.cuda.is_available()`
  check if an epoch is taking more than a few minutes on CREMA-D+RAVDESS+SAVEE+TESS alone.
