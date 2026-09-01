# MindFlow — Video Module Handoff Prompt (for Antigravity IDE)

Paste everything below into Antigravity as your opening task/system prompt.

---

## Project context

I'm building **MindFlow**, a multimodal real-time student state estimation
system for my B.Tech capstone. I personally own the **video processing
module** end-to-end: static image classifier, temporal (video) model,
inference pipeline, and the JSON output contract handed off to the fusion
layer.

Teammates own other modalities — you don't need to touch their code:
- **Srujana** — fusion/integration layer + dashboard/gamification
- **Vaishnavi** — audio/speech emotion recognition
- **Satwik** — text/NLP emotion classification

**Target:** 7-class emotion recognition — happy, sad, angry, neutral, fear,
disgust, surprise — shared across all modalities.

**Output contract** (per-window JSON, already finalized, don't change it
without discussion): `timestamp`, `modality`, `emotion`, `confidence`,
`all_scores` (softmax, sums to 1.0), `window_ms`, `features`, `error`.

## Environment

- Windows, working dir `D:\video\files`, venv at `D:\video\.venv`
- RTX 4090 (24GB VRAM), Python 3.12, PyTorch 2.6.0+cu124
- Static datasets mounted via `mklink /J` junctions at
  `D:\archive\code\code\data`
- DFEW at `D:\video\DFEW\data\DFEW`; FERV39K at `D:\video\FERV39K`
- MAFW is dropped for good (broken archives, expired access) — don't
  suggest re-adding it

## Current state

### Track A — static image classifier (DONE, stable)
EfficientNet-B2 backbone, trained on AffectNet, CK+, FER+ (CK+48/kaggle7/
stock2fer), RAF-DB (~179K samples). **Confirmed healthy at ~74.6%
static-only accuracy** via `eval_confusion_static.py`. Blended accuracy
including video-source frames reads lower (~55%) — that's a known
population-mix artifact, not a regression, so always evaluate static-only
separately. Fear/disgust have a data-quality ceiling from annotator noise
in the source datasets — don't try to fix this with more sampling weight.
Sampler weight is capped at 4.0× (`SAMPLE_WEIGHT_CAP_MULTIPLIER`).

### Track B — temporal (video) model (in progress)
Rebuilt from scratch: `config_temporal.py`, `datasets_temporal.py`,
`model_temporal.py`, `train_temporal.py`. Architecture: frozen
EfficientNet-B2 backbone + BiGRU head. Trained on 40,444 train / 10,188 val
clips from DFEW + FERV39K combined.

- Mean-pooling baseline: ~49.7% val accuracy, but it diluted signal on
  flat/low-drama classes (neutral, sad dropped vs. static baseline).
- **Fix implemented:** replaced mean-pooling with learned attention
  pooling (`Linear(gru_out→64) → Tanh → Linear(64→1)`, softmax over time,
  weighted sum). Checkpoint save/load already updated for the new `attn`
  key.
- **A training run with the attention-pooled model was launched but
  results were not yet confirmed as of my last session.**

## What I need you to do first

1. Check whether the attention-pooled Track B training run finished and
   what checkpoint exists in the temporal checkpoints dir. If it finished,
   run eval and report per-class accuracy — specifically whether
   neutral/sad recovered relative to the mean-pooling baseline.
2. Run the comparative eval: attention-pooled BiGRU vs. naive per-frame
   static averaging (`eval_temporal_vs_static.py`). Baseline before the
   attention fix was 49.74% BiGRU vs. 50.76% static avg — I want to know
   if attention pooling closed that gap.

## Backlog after that (in rough priority order)

- Consolidate the multiple `inference.py` versions (`inference_old1/2/3.py`
  exist as history — only `inference.py` is current) into a single clean
  pipeline
- Personal calibration fine-tuning pipeline (`user_profile.py` already has
  the profile/baseline plumbing) — guided expression session, head-only
  fine-tune, save to `profiles/<user_id>_head.pt`
- Optional bounded comparisons: EfficientNet-B4 backbone, POSTER++
  (POSTER++ handles landmarks internally via MobileFaceNet — no external
  MediaPipe landmarks needed if we go that route)
- Final full-pipeline test and cleanup
- Integration handoff to Srujana — the Track A JSON output contract is
  ready now; she doesn't need to wait on Track B

## Key technical learnings (don't relitigate these)

- Blended val accuracy is misleading whenever video-source frames are in
  the pool — always check static-only accuracy separately
- `CLASS_EXTRA_WEIGHT` values calibrated for a small pool can silently
  blow past the 4× sampler cap when the pool composition changes — always
  rebalance against the actual `cls_w` tensor after a big data change,
  don't just eyeball it
- Fear/disgust accuracy ceiling is a data-quality problem, not something
  sampling or loss weighting fixes
- Mean-pooling over GRU timesteps hurts stable/low-drama classes; attention
  pooling is the fix, already implemented
- BiGRU helps transient classes (surprise, happy, fear) but hurt stable
  ones (neutral, sad) before the attention fix
- Disgust in DFEW alone has very few samples (~116) — that's better fixed
  by adding FERV39K via ConcatDataset than by hyperparameter tuning
- PyTorch CUDA installs can silently fall back to CPU — always verify
  `torch.cuda.is_available()` after any environment setup

## How I like to work — please follow this workflow

- **Strict sequential steps:** run one script, show me the full terminal
  output, then propose exactly one next step. Don't skip ahead or batch
  multiple changes before I've seen results.
- **Diagnose before coding:** write a small standalone diagnostic `.py`
  script to confirm a hypothesis before making structural changes. Avoid
  multiline `cmd.exe` commands on Windows — write scripts instead.
- **Verify before training:** run an `ast.parse()` syntax check and a
  quick sanity/verification script before kicking off any multi-hour GPU
  training run.
- **Re-establish ground truth after a break:** when resuming, re-run eval
  scripts to confirm a checkpoint is still healthy before building on it.
- **Keep Track A and Track B configs fully separate** — no shared config
  state between the static and temporal pipelines.
- I want direct recommendations with your reasoning stated, not a menu of
  options for me to pick from — make the ML methodology call and explain
  why.

Start by checking the Track B temporal checkpoint status and reporting
back before touching anything else.
