# MindFlow — Video Processing Module — Agent Context

Read this in full before touching any file. This is a B.Tech capstone project
(MindFlow: multimodal real-time student state estimation). I own the **video
processing module** end-to-end. Do not assume defaults that contradict this
document — ask me if something here seems to conflict with the code you see.

## Team & scope (do not modify these people's modules)

- **Me (owner)**: video module — static classifier (Track A), temporal model
  (Track B), inference pipeline, calibration, integration handoff.
- **Srujana**: fusion/integration layer + dashboard/gamification. Consumes my
  JSON output contract. Don't redesign the contract without flagging it to me
  first — she may already be building against the current shape.
- **Vaishnavi**: audio/speech emotion. Not my code.
- **Satwik**: text/NLP emotion. Not my code.

7-class contract shared across all modalities:
`["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]`
Framed downstream as "student state," not raw emotion.

Fusion baseline weights: video 0.40, audio 0.35, text 0.25 (static — confidence-
weighted dynamic fusion is planned but not yet implemented, that's Srujana's work).

## Output contract (do not break silently)

Per-window JSON: `timestamp`, `modality`, `emotion`, `confidence`, `all_scores`
(softmax, sums to 1.0), `window_ms`, `features`, `error`.

`features` includes: `landmarks` (empty unless explicitly requested — biometric,
excluded by default), `action_units`, `frame_idx`, `subtle_expr`, `blink`,
`head_pose`, `gaze`, `posture`. These behavioral features run on **different
timescales** than the emotion smoothing window (blink/drowsiness are 60s rolling
aggregates, gaze is smoothed over 30 samples, emotion is a short window) —
don't collapse them into one "instant" without noting that.

## Environment — do not deviate

- Windows, working directory `D:\video\files`, venv at `D:\video\.venv`.
- Python 3.12, PyTorch 2.6.0+cu124, RTX 4090 (24GB VRAM).
- Static datasets reached via Windows directory junctions (`mklink /J`) at
  `D:\archive\code\code\data`. Video datasets: DFEW at
  `D:\video\DFEW\data\DFEW`, FERV39K at `D:\video\FERV39K`.
- **MAFW is permanently dropped** (broken archives, expired access). Do not
  suggest re-adding it or "just re-downloading" it.
- Always verify CUDA is actually being used after any environment change —
  CPU-only PyTorch builds install silently and fail silently (train "works"
  but is unusably slow with no error).

## Track A — static classifier (current state: healthy, don't regress)

EfficientNet-B2 backbone, trained on 6 static sources (AffectNet, CK+, FER+
variants, RAF-DB; ~179K samples). **Confirmed healthy at ~74.6% static-only
accuracy** via `eval_confusion_static.py`.

**Critical**: the blended eval number (~55% when video-source frames are
mixed into the val set) is a **known population-mix artifact, not a
regression**. If a script or agent run reports a lower blended number, do
not treat that as a bug — always separately check static-only accuracy
before concluding anything changed.

Fear/disgust accuracy has a **data-quality ceiling** from annotator-level
label noise in FER+/AffectNet. This is not fixable by resampling — don't
propose more oversampling weight on fear/disgust as a fix; the ceiling is in
the labels, not the sampler.

Sampler weight cap: `SAMPLE_WEIGHT_CAP_MULTIPLIER = 4.0x` in the weighted
sampler — this exists deliberately as code hygiene. `CLASS_EXTRA_WEIGHT`
values in config.py are calibrated for a specific pool composition — if the
data pool changes significantly (new source added, source dropped), the
effective weights must be re-checked against actual `cls_w` output from the
sampler, not just eyeballed. Always verify against the sampler's real output
tensor, never by reimplementing the weighting formula by hand.

## Track B — temporal model (rebuilt from scratch, in progress)

Four files: `config_temporal.py`, `datasets_temporal.py`, `model_temporal.py`,
`train_temporal.py`. Architecture: **frozen** EfficientNet-B2 backbone + BiGRU
head (~330K/8.8M trainable params). Training data: 40,444 train / 10,188 val
clips, DFEW + FERV39K combined.

Baseline (mean-pooling over GRU timesteps): best val accuracy **~49.7%**.
Diagnosed problem: mean-pooling dilutes signal for flat/low-drama clips
(neutral, sad) - confirmed pattern: BiGRU helps transient classes (surprise
+16.7pp, happy +7.5pp, fear +5.2pp) but hurts stable ones (neutral -12.2pp,
sad -7.3pp).

**Fix implemented, not yet confirmed**: replaced mean-pooling with learned
attention pooling (`Linear(gru_out_dim -> 64) -> Tanh -> Linear(64 -> 1)`,
softmax over time, weighted sum). Checkpoint save/load updated to include
`attn` state dict key. A fresh training run was launched at the end of the
last session - **first task in any new session is to check whether this run
finished and what the result was**, before proposing anything else for Track B.

Disgust in DFEW has very few native samples (~116 before FERV39K was added
to the pool) - this is a data-scarcity issue to solve via the combined
dataset, not via hyperparameter tuning.

## Hard rules for any agent working in this repo

1. **Never share config state between Track A and Track B.** They are
   deliberately separate (`config.py` vs `config_temporal.py`). A task that
   touches both in the same edit is almost certainly wrong - stop and ask.
2. **Never trigger a multi-hour GPU training run without me reviewing the
   diagnostic first.** Write a short standalone `.py` diagnostic script,
   show me its output, wait for go-ahead, then train. This applies to
   `train.py`, `train_temporal.py`, `finetune_head_cpu.py`, and
   `finetune_contrastive.py` - all of them.
3. Before any structural code change (new layer, new loss, changed sampler
   logic), run `ast.parse()` or an equivalent syntax/sanity check and a small
   targeted verification script first. Don't go straight from edit to full run.
4. Write diagnostics as `.py` files, not inline multiline `cmd.exe` commands -
   Windows shell multiline handling is unreliable here.
5. When re-establishing ground truth after a break (new session, resumed
   work), re-run the relevant eval script first to confirm the checkpoint is
   still healthy before building anything on top of it. Don't assume the last
   known number still holds.
6. Do not propose "just retrain with different hyperparameters" as a fix for
   an accuracy problem without first identifying whether the issue is (a) a
   real bug, (b) a population-mix artifact, or (c) a label-noise ceiling -
   these have different fixes and only (a) is a coding task.
7. POSTER++ (if ever explored) handles landmarks internally via MobileFaceNet
   - do not add external MediaPipe landmark input for it, that's redundant.

## Current backlog (roughly in priority order)

1. Confirm attention-pooled Track B run - does neutral/stable-class accuracy
   recover as expected?
2. Comparative eval: attention-pooled BiGRU vs. naive per-frame static
   averaging (`eval_temporal_vs_static.py` - baseline was 49.74% BiGRU vs.
   50.76% static avg, so the bar to clear is "beats simple averaging").
3. Consolidate `inference.py`, `inference_old1.py`, `inference_old2.py`,
   `inference_old3.py` into a single pipeline. Keep: TTA, CLAHE, gamma
   correction, glasses-aware EAR calibration, neutral suppression, gaze
   tracker, drowsiness monitor, high-visibility HUD (from the current/latest
   `inference.py`). Drop: the older grey-HUD, non-Tasks-API MediaPipe usage.
4. Personal calibration fine-tuning pipeline (`user_profile.py`) - guided
   expression session -> head-only fine-tune -> save to
   `profiles/<user_id>_head.pt`.
5. Optional bounded comparisons: EfficientNet-B4, POSTER++.
6. Final full-pipeline test and cleanup.

## Novel contributions to keep visible in code/comments (panel/report material)

Per-user calibration pipeline, NeutralSuppressor algorithm, forehead-furrow
frustration estimator, confidence-weighted dynamic fusion (Srujana's side, but
video module supplies the confidence), stress-recovery gamification (feeds
from `features.blink`/`features.gaze`/`drowsiness`), privacy-by-design local
processing (mapped to DPDP Act 2023 - no raw frames/landmarks persisted by
default).
