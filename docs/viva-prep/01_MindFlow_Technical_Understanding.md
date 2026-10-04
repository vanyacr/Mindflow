# MindFlow: Technical Understanding, Pass 1

**Purpose:** A first-pass technical consolidation of the four module handoffs (Video, Audio, Text, Fusion). The team should check it before the final LLD, PPT and viva material are generated.
**Prepared:** 4 Oct 2026
**Status:** DRAFT FOR TEAM VERIFICATION. Nothing here is final until the gaps in Section 9 are closed.

---

## 0. How to read this document

Every claim carries a provenance tag:

| Tag | Meaning |
|---|---|
| **[H-V] [H-A] [H-T] [H-F]** | Stated in the Video / Audio / Text / Fusion handoff |
| **[C-V] [C-A] [C-T]** | Read directly in source code on GitHub (`vanyacr/Mindflow`). Video: branch `feature/video-module` @ `175fc48` (1 Sep). Audio: `feature/audio-module` @ `0df85b8` (1 Sep). Text: `feature/text-module` @ `5b1fda2` (31 Aug) |
| **⚠ CONFLICT** | Two sources disagree. The team must decide which is true |
| **[MISSING — VERIFY]** | Not in any handoff and not visible in the code available to me. It must be verified from code, configuration or an experiment log |
| **[ENG. REASONING]** | My engineering justification, not project evidence. Label it as such in the viva |

### A critical caveat on the source code

I checked the four handoffs against the GitHub repository. **The integrated system described in the Fusion handoff is not on GitHub.** That handoff cites branch `milestone-75-integration`, commit `1a881fe`. That branch does not exist on `origin`. The remote has only `main`, `develop`, `Audio_Training`, `feature/audio`, `feature/audio-module`, `feature/fusion-module`, `feature/text-module` and `feature/video-module`. On GitHub, `mindflow/fusion/*.py` holds empty 0-byte files.

What this means:

- **Fusion, personalisation, temporal engine, decision engine, intervention policy, feedback, runtime, adapters, storage, API and UI:** every detail comes from the Fusion handoff alone. **None of it is verifiable from the repository.**
- **Video, audio and text:** the GitHub copies may be **older** than the copies running inside the integrated system. The video handoff says the copies on `D:\video\files` are the source of truth. The Fusion handoff describes a text model that does **not exist** in the GitHub text code (see 4.3).

**Action #0 for the team:** push `milestone-75-integration` (including `video_module/`, `audio_module/`, `text_module/` as integrated) to GitHub. Until that happens, the team cannot show the panel the code behind the core contribution.

---

## 1. Understanding of the four handoffs

### 1.1 What each handoff is

| Handoff | Author / owner | Date / commit | What it really is | Reliability notes |
|---|---|---|---|---|
| Video | Vanya (owner) | 4 Oct 2026 | Working notes for the video owner: Track A (static CNN, stable), Track B (temporal, **unsolved research problem**), `inference.py` output contract, open integration issues | Honest about Track B. Some statements conflict with the code on GitHub (label order, embedding size) |
| Audio | Written by Vanya C R about Vaishnavi's module | 4 Oct 2026, built from `0df85b8` + laptop audit | The most rigorous of the four. Gives architecture, data, training config, results, and a list of corrections to older documents | Explicitly retracts r = 0.389 for the shipped model. Several training scripts exist only on the laptop |
| Text | Sathwik/Satwik | Undated | Short summary: DistilBERT SST-2 sentiment plus a keyword emotion scorer with 6 classes | Thin. **Contradicted by the Fusion handoff** on which text model runs |
| Fusion | Srujana | `1a881fe`, 1 Oct 2026 | Full system handoff: contract, fusion maths, personalisation, temporal engine, decision, policy, feedback, runtime, storage, UI/API, tests, results | The most detailed, but **its code is not on GitHub**. It quotes an audio stress number the audio handoff calls wrong |

### 1.2 One-paragraph synthesis (what MindFlow actually is today)

MindFlow is a **local, privacy-first desktop application** (Windows, Python 3.12) [H-F §22]. It watches a student studying at a laptop through three channels: a **webcam** (face emotion, eyes, head pose, gaze), a **microphone** (speech emotion plus a depression-severity proxy used as "stress"), and **chat messages** (text emotion, sentiment and heuristic stress) [H-F §1]. Each modality runs its own separately trained model, unmodified, in its own worker process. Adapters convert each output into a common 11-field `ModalityObservation` [H-F §5–6]. A **late, confidence- and freshness-weighted fusion engine** combines the per-modality emotion distributions and stress signals into a `FusedObservation`. That object holds a fused emotion and 8 dashboard parameters, each with a confidence [H-F §7]. The fused signals are then:

1. compared with the student's own calibrated baseline (robust median/MAD z-scores plus expression templates) [H-F §8];
2. accumulated over time by leaky evidence accumulators with hysteresis [H-F §9];
3. mapped by a rule-based decision engine to one student state and a candidate action [H-F §10];
4. gated by a rule-based intervention policy that decides **when** a popup is allowed [H-F §11];
5. written up by Gemini, or by a curated fallback library, which decides **what** the popup says [H-F §12].

Everything is logged as derived numbers only, in SQLite and JSONL [H-F §14], and shown in Tkinter, Streamlit or a FastAPI phone page [H-F §15]. **No fused or end-to-end accuracy has been measured yet.** The formal evaluation (experiments A–G) has not been run [H-F §17–18].

### 1.3 Ownership

| Module | Owner | Source |
|---|---|---|
| Video (Track A, Track B, `inference.py`) | Vanya | [H-V §1], [H-F §2] |
| Audio | Vaishnavi | [H-V §1], [H-F §2] |
| Text | Sathwik/Satwik (spelling differs across handoffs) | [H-V], [H-F] |
| Fusion, personalisation, temporal, decision, feedback, popups, dashboards, gamification, storage, integration, web/phone | Srujana | [H-F §2] |

Note: the `feature/video-module` commits on GitHub are authored by "Kshama Jain". Confirm who pushed them and whether that push matches `D:\video\files`.

### 1.4 Cross-handoff conflicts found so far (resolve these first)

| # | Topic | Source A | Source B | Impact |
|---|---|---|---|---|
| C1 | **Video emotion label order** | [H-V §3]: `happy, neutral, sad, angry, fear, disgust, surprise` | [C-V] `config.py` and [H-F §5.1]: `happy, sad, angry, neutral, fear, disgust, surprise` | Fusion keys by name, so fusion is safe. **Track B, the checkpoint head and any one-hot code depend on the order.** Confirm against `D:\video\files\config.py` |
| C2 | **Text model** | [H-T] and [C-T]: DistilBERT SST-2 sentiment + **keyword** scorer, 6 classes (`anxious, stressed, calm, motivated, frustrated, focused`) | [H-F §5.3]: **DistilRoBERTa** `j-hartmann/emotion-english-distilroberta-base`, 7 classes, plus SST-2 sentiment. Says "it was 6-class (GoEmotions); now 7-class" | Fusion needs a 7-class *probability* distribution; the keyword scorer does not produce one. **The integrated `text_module/` was evidently changed, but no handoff documents the change.** Also, the 6-class version was keyword-based, not GoEmotions-trained |
| C3 | **Audio stress score** | [H-A]: shipped head r = **0.185 ± 0.084**; r = 0.389 belongs to an unsaved, possibly leaked run | [H-F §5.2]: "session-level Pearson r ≈ 0.39" | **The Fusion handoff and any slides must say 0.185.** Reporting 0.389 for the shipped model is factually wrong |
| C4 | **Video static accuracy** | [H-V]: 74.6% | [C-V] `config.py` comment: "known-good 74.3%"; [H-F §25]: "74.4% … not recorded in the repo" | Pick one number and tie it to a log file and an eval script (`eval_confusion_static.py`) |
| C5 | **Video Track B result** | [H-V]: BiGRU ≈ 49–50%, **no better than** static averaging (50.76%); root cause unknown | [C-V] `README.md`: "Track B … **64.03% on DFEW** … Champion Model", "live accuracy **>90%**" | The README numbers contradict the owner's own handoff. **Do not quote 64.03% or >90%.** Remove them from the README |
| C6 | **Fusion feature vector** | [H-A]: "Fusion vector, as planned: `[audio 768 ‖ video 512 ‖ text 768]`" | [H-F §7.1]: late fusion of **predictions**; embeddings not stored or used. [C-V]: `get_embedding()` returns **256-d**, not 512 | The concatenation plan is obsolete. Do not present it |
| C7 | **Training set of ~179K images** | [H-V]: ~179K includes DFEW and FERV39K frames | [C-V] README: "~179k static face crops (AffectNet, RAF-DB, CK+, FER+)" | Confirm the count and its composition from the training log |
| C8 | **Old architecture in the repo README** | `main/README.md`: MFCC audio features, DAiSEE, CogLoad, "**feature-level aggregation**", Burnout Risk, Well-Being Index | All handoffs: WavLM, no DAiSEE/CogLoad, **late** fusion, 8 different parameters | The panel may remember the Review-1 claims. Prepare a "what changed since Review-1" slide |
| C9 | **Python version** | [H-A]: audio runs on Python 3.10/3.11 | [H-F §22]: one `.venv312` (Python 3.12) for everything | Confirm the audio module loads under 3.12 from `requirements-lock.txt` |
| C10 | **Keystroke modality** | Not in any of the four handoffs (the text handoff lists it only as future work) | [C-T]: `feature/text-module` contains a full `KEYSTROKE/` package, a trained model script, a background service and a `weighted_fusion()` for keystroke + text | Is keystroke in scope, descoped, or abandoned? Decide and say so in the viva |
| C11 | **Video PERCLOS window and EAR calibration duration** | [H-V]: blink/PERCLOS = 60 s rolling; EAR calibration = 5 s | [C-V] `inference.py`: trackers are sized for `fps=30` (`deque(maxlen=fps*60)` = 1800) but are **updated only every `step` = 15 frames (2 Hz)**. That makes the effective window 1800 / 2 Hz = **15 min**, and EAR calibration (150 samples) take **75 s**, not 5 s | Possibly a real bug in the standalone module. [H-F] lists `perclos_window_samples = 120` (60 s at 2 Hz), so the integrated worker may override it. **Verify in `runtime/video_worker.py`** |

---

## 2. Overall system architecture (actual, not conceptual)

### 2.1 End-to-end pipeline (from [H-F §4, §13], with verified module internals)

```
USER (student at laptop)
 │
 ├─ Webcam ──► VideoRuntime ──► video worker PROCESS (spawn)          [H-F §13]
 │               every 500 ms: gamma 1.8 → YOLOv8n-face → crop+CLAHE →
 │               EfficientNet-B2 (4-view TTA, T=1.2) → SoftmaxSmoother(5) →
 │               NeutralSuppressor → MediaPipe landmarks/pose → EAR/PERCLOS,
 │               head pose, gaze, "AU" proxies, engagement                   [H-V §6, C-V]
 │
 ├─ Mic ─────► AudioRuntime ──► audio worker PROCESS (spawn)
 │               16 kHz, 6 s window every 1 s → WebRTC VAD gate →
 │               (speech) trim 30 dB → peak-norm 0.95 → centre-crop/tile 6 s →
 │               WavLM-Large → attention pool → 768-d → emotion(7) + stress     [H-F §5.2, H-A, C-A]
 │               ≈ one result per 10.1 s on CPU while speaking
 │
 └─ Chat ────► text pipeline (main process, per message)
                 clean_text → sentiment (DistilBERT SST-2) + emotion
                 (DistilRoBERTa 7-class per [H-F]; keywords per [C-T]) → stress  ⚠ C2
 │
 ▼  Queues (producer–consumer) → ADAPTERS → ModalityObservation (11 fields)   [H-F §6]
 ▼
FusionEngine.update(obs, now) / get_latest(now) on a 1 s pipeline tick        [H-F §7.2, §13]
   1 sanitise → 2 freshness gate (3 s / 15 s / 20 s linear fade) →
   3 personal re-labelling of the video vote (ExpressionProfile, JS distance) →
   4 effective weight = base × certainty × availability →
   5 conflict handling (similarity, text agreement, negative consensus, total disagreement) →
   6 weighted fused distribution → label (≥ 0.30, hysteresis) → weighted-support confidence
   + stress fusion (voice median 60 s, personal ref, text share) + 8 parameters
 ▼
FusedObservation (emotion + 8 parameters, each {value, confidence}; indicators; votes)
 ▼
PersonalBaseline: robust z = (x − median) / max(MAD·1.4826, floor)            [H-F §8.1]
 ▼
TemporalStateEngine: leaky evidence accumulators + hysteresis per condition     [H-F §9]
 ▼
DecisionEngine: ONE student state + candidate action (+ cooldown/back-off)      [H-F §10]
 ▼
InterventionPolicy (WHEN: grace, gap, hourly cap, duty cycle, flow protection)  [H-F §11]
 ▼
FeedbackEngine (WHAT: Gemini 1–2 sentences, filters, curated fallback)          [H-F §12]
 ▼
Popup PROCESS (colour, 3 s click-through animation, chime; OK / Not now / ignore)
 ▼ response → logged → back-off (cooldown doubles)                              (feedback loop)
 ▼
Gamification (points from logged events) · SQLite + JSONL (derived numbers only)
 ▼
MultimodalRuntime.get_latest_state() (pure read) → Tk dashboard | FastAPI → Streamlit / phone
```

### 2.2 Each transition explained

| Transition | Mechanism | Why it exists | Source |
|---|---|---|---|
| Device → worker | Separate `spawn` processes for video and audio | Heavy models must not freeze the UI. The handoff modules use clashing generic names (`config`, `inference`) and CWD-relative paths. A crash in one worker does not kill the app | [H-F §13] |
| Worker → main | Queues | Producer–consumer decoupling | [H-F §13] |
| Raw output → `ModalityObservation` | Adapters (`adapters/`, `utils/observation.py`) | Fusion never needs to know which model produced the data. Emotions are keyed **by name** because the video and audio label orders differ | [H-F §6] |
| Observation → fusion | `update(obs, now)` with arrival-time clock; a 1 s self-tick calls `get_latest(now)` | Different rates (video 2 Hz, audio ~0.1 Hz, text sporadic). Stale data must expire without polling. Deterministic replay | [H-F §7.2–7.3, §13] |
| Fusion → baseline | Robust z-scores against the user's calibration | People differ ("their own normal") | [H-F §8] |
| Baseline → temporal | Evidence += dt × confidence while a condition holds; decays otherwise | One noisy reading cannot trigger or reset a state | [H-F §9] |
| Temporal → decision | Map of persistent conditions to states and actions | A single state authority | [H-F §10] |
| Decision → policy | Rule table with windows, duty cycles and cooldowns | Restraint: popups only for persistent, confident evidence | [H-F §11] |
| Policy → feedback → popup | Gemini text with filters; curated fallback; popup in its own process | Predictable timing, warm wording, always visible | [H-F §12] |
| Popup → back to policy | OK / Not now / ignored are logged; "Not now" doubles that situation's cooldown | User feedback loop | [H-F §11–12] |
| All → storage / UI | SQLite (1 Hz fused states), JSONL; UIs poll `get_latest_state()` once per second | Privacy (derived numbers only); offline replay | [H-F §13–15] |

---

## 3. Module breakdown

| # | Module | Code location (integrated) | Responsibility | Inputs | Outputs | Evidence level |
|---|---|---|---|---|---|---|
| M1 | Video perception | `video_module/` + `runtime/video_worker.py` | Face emotion, eyes, pose, gaze, engagement | Webcam frames | Per-window JSON (every 500 ms) | Model **[C-V]**; worker [H-F only] |
| M2 | Audio perception | `audio_module/` + `runtime/audio_worker.py` | Speech emotion, stress proxy, embedding | 16 kHz mic | `predict()` dict | Model **[C-A]**; worker and VAD gate [H-F only] |
| M3 | Text perception | `text_module/` | Sentiment, emotion, stress per message | Chat message | Dict | ⚠ C2 |
| M4 | Adapters / contract | `adapters/`, `utils/observation.py` | Normalise to `ModalityObservation` | Raw outputs | 11-field dataclass | [H-F only] |
| M5 | Fusion | `fusion/fusion_engine.py`, `fusion/config.py` | Fuse emotion and stress; 8 parameters | Observations + clock | `FusedObservation` | [H-F only] |
| M6 | Personalisation | `personalization/baseline.py`, `expression_profile.py` | Baseline z-scores; expression templates | Calibration data | z-scores; personal distribution | [H-F only] |
| M7 | Temporal state | `decision/temporal_engine.py` | Persistence with hysteresis | Indicators, z, confidence | Condition levels: NORMAL / SLIGHT / PERSISTENT | [H-F only] |
| M8 | Decision | `decision/decision_engine.py` | One state + action | Conditions | State, action, priority | [H-F only] |
| M9 | Intervention policy | `feedback/intervention_policy.py` | WHEN to interrupt | State, parameters | Popup trigger | [H-F only] |
| M10 | Feedback + popup | `feedback/feedback_engine.py`, `popup.py`, `mood_fx.py` | WHAT to say; show it | Situation, context | Message, popup, response | [H-F only] |
| M11 | Chatbot | `chatbot/` | Gemini chat; offline fallback; feeds M3 | User message | Reply + text observation | [H-F only] |
| M12 | Gamification | `gamification/engine.py` | Points, streaks, goals, badges | Logged events | Score state | [H-F only] |
| M13 | Storage | `storage/database.py`, `session_logger.py` | SQLite + JSONL | All derived data | Persistent logs | [H-F only] |
| M14 | Runtime / controller | `runtime/multimodal_runtime.py`, `worker_process.py` | Facade, process management, pipeline clock | Everything | `get_latest_state()` | [H-F only] |
| M15 | UI / API | `dashboard/`, `ui/streamlit_app.py`, `server/app.py` | Tk, Streamlit, FastAPI + HTTPS phone page | State | Views, REST | [H-F only] |
| M16 | Tools / evaluation | `tools/replay.py`, `evaluate.py`, `guided_session.py`, `rebuild_calibration.py` | Replay, experiments A–G | JSONL logs | Metrics | [H-F only]; **experiments not run** |
| M17 | Tests | `tests/`, `tests/hardware/` | 163 automated tests + device smoke tests | — | Pass/fail | [H-F §18, §25] "verified 2026-10-04" |
| (M18) | Keystroke | `KEYSTROKE/` on `feature/text-module` | Typing-dynamics stress | Key events | Stress score | [C-T]; **not in any handoff** ⚠ C10 |

---

## 4. Model-by-model breakdown (40-question matrix)

Legend: ✅ documented/verified · ⚠ conflict or caveat · ❌ **[MISSING — VERIFY]**

### 4.1 VIDEO, Track A: static face-emotion CNN (LIVE in the system)

| # | Question | Answer | Source |
|---|---|---|---|
| 1 | Problem solved | Per-frame 7-class facial emotion recognition | [H-V §3–4] |
| 2 | Why video | The face is the continuous, highest-rate signal (2 Hz). It also gives eyes, pose and gaze for attention and drowsiness, which voice and text cannot provide | [H-F §24] + [ENG. REASONING] |
| 3 | Why this model | EfficientNet-B2 beat B4 "everywhere at 2× params" (documented negative result) | [H-V §4] |
| 4 | Alternatives considered | EfficientNet-B4 (dropped). For temporal: BiGRU, Transformer (Track B) | [H-V §4–5], [C-V] `model_b4.py` |
| 5 | Why suitable | Accuracy vs size; on-device CPU inference ("2.5× faster than B4" per README, **not otherwise verified**) | [H-V]; README ❌ speed |
| 6 | Dataset | AffectNet (YOLO format), CK+, FER+ (CK+48, kaggle7, stock2fer subsets; kaggle3 skipped), RAF-DB, DFEW (frame-sampled), FERV39K (frame-sampled). MAFW dropped | [C-V] `config.py`, `datasets.py` |
| 7 | Why those | Mix of lab (CK+), web (AffectNet, RAF-DB, FER) and in-the-wild video frames (DFEW, FERV39K) for robustness | [ENG. REASONING]; the code comments explain the oversampling only |
| 8 | Samples | "~179K" ⚠ C7. Per-dataset counts ❌ | [H-V] |
| 9 | Classes | 7: happy, sad, angry, neutral, fear, disgust, surprise (CK+ "contempt" dropped) | [C-V] ⚠ C1 order |
| 10 | Labels | One-hot soft vectors (`one_hot()`); mixup makes them soft | [C-V] |
| 11 | Split | **Per-dataset, not unified:** AffectNet uses its own `valid/`; CK+, FER-CK48 and FER-stock use a **random 85/15 split, seed 42 (not subject-disjoint)**; FER kaggle7 and RAF-DB use their **test** folders as val; DFEW uses `set_1` test as val; FERV39K uses `test_All.csv` as val. **No separate held-out test set** | [C-V] `datasets.py` |
| 12 | Preprocessing (training) | BGR→RGB, resize to **160×160**, ImageNet mean/std normalisation | [C-V] |
| 13 | Feature representation | EfficientNet-B2 pooled features, **1408-d** → head 512 → 256 (embedding) | [C-V] `model.py` |
| 14 | Input shape | (B, 3, 160, 160) | [C-V] `IMAGE_SIZE = 160` |
| 15 | Output shape | (B, 7) logits → softmax with T = 1.2 at inference | [C-V] |
| 16 | Architecture | timm `efficientnet_b2` (ImageNet-pretrained, `num_classes=0`) + head: Linear(1408→512), BN, ReLU, Dropout(0.4), Linear(512→256), BN, ReLU, Dropout(0.2), Linear(256→7) | [C-V] |
| 17 | Layer count | Backbone depth per timm B2 (block count ❌ not documented). Head = 3 linear layers | [C-V] |
| 18 | Layer roles | Backbone = spatial features. BN = stabilises the head. Dropout = regularisation. 256-d = embedding available for fusion (`get_embedding`) but **not used** | [C-V], [H-F] |
| 19 | Key hyperparameters | Batch 96, max 60 epochs, LR 1e-4, WD 2e-4, label smoothing 0.1, mixup α = 0.3 (p = 0.5), grad-clip 2.0, early-stop patience 15, AMP | [C-V] `config.py`, `train.py` |
| 20 | Optimizer | AdamW | [C-V] |
| 21 | LR / schedule | 1e-4. `CosineAnnealingWarmRestarts(T_0=8, T_mult=2, eta_min=5e-7)` | [C-V] |
| 22 | Batch size | 96 (comment: "drop to 48–64 if OOM") | [C-V] |
| 23 | Epochs | Max 60 + early stopping. **Actual epochs run ❌** (in `checkpoints/train_log.csv`) | [C-V] |
| 24 | Loss | `SoftCrossEntropyLoss` (cross-entropy on label-smoothed soft targets, smoothing 0.1) | [C-V] |
| 25 | Why that loss | Soft targets are needed for mixup. Smoothing counters label noise (the handoff notes fear/disgust annotator noise) | [C-V] + [H-V §4] |
| 26 | Activations | ReLU in the head; backbone SiLU/Swish (standard for EfficientNet ❌ not stated in project docs); softmax at output | [C-V] |
| 27 | Transfer learning | Yes, ImageNet-pretrained backbone | [C-V] |
| 28 | Frozen layers | **None.** `train.py` trains `model.parameters()`, a full fine-tune | [C-V] |
| 29 | Fine-tuned | Entire network | [C-V] |
| 30 | Augmentation | HFlip 0.5; Rotate ±20° (0.5); BrightnessContrast (0.35/0.45, 0.6); HSV (0.4); one of Grid/Elastic/Perspective; GaussNoise 0.25; one of Blur/MotionBlur/Sharpen (0.25); CoarseDropout 2–8 holes of 8–24 px (0.35); mixup | [C-V] `get_transforms` |
| 31 | Why augmentation | Webcam lighting, pose, blur and occlusion robustness | [ENG. REASONING] (no project rationale written) |
| — | Sampling | `WeightedRandomSampler` = inverse class freq × dataset oversample (AffectNet 1.0, CK+ 8.0, FER-CK48 6.0, FER-K7 2.0, FER-stock 4.0, RAF-DB 4.0, DFEW 0.6, FERV39K 0.3) × class extra weight, **capped at 4× the mean** | [C-V] |
| 32 | Metrics | Accuracy (overall, per class); confusion matrices | [C-V] `train.py`, `eval_confusion*.py` |
| 33 | Why those | ❌ Not justified anywhere. **Macro-F1 is not reported for video, but it should be** (class imbalance) |  |
| 34 | Results | **Static-only val accuracy ≈ 74.6%** (6 static sources, `eval_confusion_static.py`). Blended 8-source val ≈ 55% (in-the-wild frames). ⚠ C4 | [H-V], [C-V] |
| 35 | Weaknesses | Fear/disgust capped by label noise. Reported accuracy is on the **same validation sets used for checkpoint selection** (optimistic). CK+ random split leaks subjects across train/val. Leans towards negative classes on some cameras [H-F §21] | [H-V], [C-V], [H-F] |
| 36 | Failure cases | Talking face → surprise/disgust/angry [H-F §19]. Resting face read as "sad". Looking down at notes. Glasses and lighting affect EAR | [H-F] |
| 37 | Inference | See Section 4.1.1 | [C-V] |
| 38 | Model output | `emotion`, `confidence`, `all_scores{7}`, plus drowsiness, engagement and features | [H-V §6] |
| 39 | Confidence | Top probability of the 5-window **averaged** softmax (`SoftmaxSmoother`). If the top probability is below 0.30 the label becomes "neutral" but the confidence stays at that low value. If `NeutralSuppressor` overrides neutral, confidence = the mean probability of the overriding emotion over 3 windows. **Not calibrated** (T = 1.2 is fixed; how it was chosen ❌) | [C-V] |
| 40 | Into fusion | Video adapter → `emotion_scores` (by name), `confidence` → certainty (floor 0.25), base weight 0.6, freshness 3 s. Optionally re-labelled by the user's ExpressionProfile first | [H-F §6–7] |

#### 4.1.1 Video inference, exactly ([C-V] `inference.py`)

1. Webcam frame (the code assumes 30 fps). Analysis runs every `step = fps × 500 / 1000 = 15` frames, i.e. **2 Hz**.
2. **Gamma LUT, γ = 1.8** (brightens) → **YOLOv8n-face** bounding box (Ultralytics auto-downloads it).
3. Crop → resize 160 → **CLAHE** (clipLimit 3.0, tile 4×4, on the L channel of LAB).
4. **TTA, 4 views**: original, horizontal flip, 95% centre zoom, fixed +0.15 brightness / +0.10 contrast. Logits / **T = 1.2** → softmax → mean of the 4 views. `--no_tta` uses 1 view.
5. **SoftmaxSmoother(window 5)**: mean of the last 5 distributions. If top < `CONF_THRESHOLD = 0.30`, label = neutral.
6. **NeutralSuppressor**: if the label is neutral but some non-neutral emotion has ≥ **0.25** in each of the last **3** windows, output the strongest such emotion.
7. **MediaPipe Tasks** (0.10.33) face landmarker and pose landmarker (detection/presence/tracking confidence 0.4) → EAR (landmarks 33,160,158,133,153,144 / 362,385,387,263,373,380), blink rate, PERCLOS, head pose (pitch/yaw/roll), gaze_x ∈ [−1, 1], posture, optical-flow "subtle expression".
8. **"AU04/AU06/AU12" are geometric landmark distances** (e.g. AU04 = |y(70) − y(33)|). They are **not FACS action-unit intensities.** Say "AU proxies" in the viva.
9. **Drowsiness level** from PERCLOS: ALERT < 0.15, MILD < 0.25, DROWSY < 0.40, CRITICAL ≥ 0.40.
10. **Engagement (0–100)**: 50 + 20·(happy + surprise) − 15·(0.8 sad + 0.5 fear + 0.3 neutral) ± 8 for blink rate (+8 in 10–18 bpm; −8 if < 5 or > 25) − 8/−15 for |yaw| > 20°/30° − drowsiness penalty (0/5/20/40). Smoothed over 90 values. *(Fusion does not use it for focus; see [H-F §7.10].)*
11. EAR threshold = 65% of the median resting EAR (auto-calibration) or the `user_profile` baseline. ⚠ C11 on effective durations.

#### 4.1.2 Video output contract ([H-V §6], top-level fields verified in [C-V])

`timestamp, modality="video", emotion, confidence, all_scores{7}, window_ms (500), tta_enabled, drowsiness{perclos, level, alert_count}, engagement (0–100), features{landmarks, action_units{AU04,AU06,AU12}, frame_idx, subtle_expr, blink{ear_left, ear_right, blink_rate_bpm, ear_avg}, head_pose{pitch,yaw,roll}, gaze{gaze_x}, posture{shoulder_raise, forward_lean, asymmetry}, personal_deltas}, error`

Timescales differ: blink/PERCLOS long window, gaze 30-sample smoothing, emotion 5-window. **No embedding is emitted.**

### 4.2 VIDEO, Track B: temporal model (NOT integrated; research in progress)

| Item | Value | Source |
|---|---|---|
| Architecture | Frozen Track A B2 → per-frame **256-d** embeddings × 16 frames → BiGRU (hidden 128, 1 layer, bidirectional → 256) → attention pooling (Linear 256→64, tanh, Linear 64→1, softmax over time) → head (Linear 256→128, BN, ReLU, Dropout 0.4, Linear 128→7) = `temp_delta` | [C-V], [H-V §5] |
| Output | `static_avg_logits + residual_scale × temp_delta`. `residual_scale` is learnable, initialised at 0.1. `static_avg` = mean over frames of B2's final Linear(256→7) | [C-V] |
| Config | SEQ_LEN 16, batch 16, 40 epochs, AdamW LR 3e-4, WD 5e-4, CosineAnnealingLR (T_max 40, eta_min 1e-6), label smoothing 0.10, class-weight cap 4.0, DFEW fold `set_1`. Optional focal loss γ = 1.5 | [C-V] |
| Variants | Unfrozen last block; Transformer (1408→512, learnable positional embedding, 2 layers, 8 heads, d_ff 1024, GELU, pre-norm, attention pooling, residual over log static-average probabilities) | [C-V] |
| Results | Mean-pool BiGRU 49.32% vs static frame-average 50.76%. Ensemble (w = 0.9 static) +0.17 pp. Attention run ≈ 50% | [H-V §5] ⚠ C5 vs README |
| Hypothesis | Attention collapsed to uniform (≈ 1/16); `residual_scale` → 0 | [H-V] **not confirmed** |
| Status | **Not integrated** in the live system | [H-F §5.1, §21] |

**Viva position:** "Track B is a documented research attempt. It has not yet beaten frame-averaging, and the diagnostic is pending. The live system uses Track A with temporal persistence handled downstream (smoothing, fusion hysteresis, temporal engine)."

### 4.3 TEXT (⚠ two different descriptions)

#### 4.3a What the GitHub code and the Text handoff describe ([C-T], [H-T])

| # | Question | Answer |
|---|---|---|
| 1 | Problem | Per-message sentiment, emotional tone, heuristic stress |
| 3 | Model | `distilbert-base-uncased-finetuned-sst-2-english` (HF pipeline), **off-the-shelf, not fine-tuned by MindFlow**. Loaded `local_files_only` unless `MINDFLOW_ALLOW_MODEL_DOWNLOAD=1` |
| 6 | Datasets | GoEmotions and DepressionEmo were **downloaded only** (`text_datasets.py`). The DepressionEmo ID is uncertain: the script tries 6 candidate IDs and falls back to `"emotion"`. **No training on either dataset exists in the code** |
| 9 | Classes | Sentiment: POSITIVE/NEGATIVE (no neutral). Emotion: 6 keyword classes `anxious, stressed, calm, motivated, frustrated, focused` |
| 12 | Preprocessing | lowercase → replace `[^a-z0-9\s]` with space → collapse whitespace → trim. Emoji, apostrophes and non-ASCII are lost. No lemmatisation, negation handling or stop-words |
| 13 | Representation | Sentiment: DistilBERT tokenizer (WordPiece, uncased). Emotion: token counts after crude suffix stripping (`ing, ed, ly, es, s`) |
| 15 | Output | Dict: `sentiment_polarity, sentiment_score, stress_score, anxiety_prob, emotional_tone, motivation_level, all_emotions{6}, estimated_sentiment_accuracy, estimated_overall_text_accuracy` |
| — | Emotion score | Σ keyword hits + phrase hits (1.5) + boost terms, then `min(score / 3, 1)`. **Not probabilities. Classes are independent** (they do not sum to 1). If all are 0, a fixed fallback profile is derived from sentiment (≥ 0.8) |
| — | Stress formula | `clamp(0.45·anxious + 0.35·stressed + 0.2·frustrated + 0.35·max(0, 1 − sentiment_score))` |
| 39 | Confidence | **None.** `sentiment_score` is the probability of the *predicted* class, whatever its polarity |
| 34 | Results | **No evaluation exists.** The `estimated_*_accuracy` fields are **heuristic display ranges computed from the input itself**, not measured accuracy. **Never quote them as accuracy** |

⚠ **Design defect to be ready for:** `1 − sentiment_score` does not depend on polarity. A confidently **NEGATIVE** message (score 0.996) adds ≈ 0.001 to stress, the same as a confidently **POSITIVE** one. The term measures sentiment *uncertainty*, not negativity. The text team's own `TEXT_FUSION_HANDOFF.md` already notes that "its direction depends on polarity". Verify whether the integrated `text_module/` fixed this.

#### 4.3b What the Fusion handoff says runs in the integrated system ([H-F §5.3])

- `j-hartmann/emotion-english-distilroberta-base` (7 classes, mapped joy→happy, anger→angry, sadness→sad) **plus** DistilBERT SST-2 sentiment, **plus** stress from stress words + negative emotion + negative sentiment.
- "Changed since Review-1: was 6-class (GoEmotions), now 7-class."

**Every one of the 40 questions is [MISSING — VERIFY] for 4.3b**: who changed it, the code, the exact stress formula, the certainty input, any evaluation. This is off-the-shelf; no MindFlow training was done. Fusion text certainty uses "top probability", which only makes sense for 4.3b.

**THE TEAM CANNOT CURRENTLY DEFEND THE TEXT MODEL. SOURCE VERIFICATION REQUIRED.**

### 4.4 AUDIO: WavLM-Large speech emotion + stress head (LIVE)

| # | Question | Answer | Source |
|---|---|---|---|
| 1 | Problem | 7-class speech emotion + depression-severity proxy ("stress") + 768-d embedding | [H-A] |
| 2 | Why audio | Prosody carries affect when the face is neutral or out of view. Independent error source | [H-F §24] + [ENG. REASONING] |
| 3 | Why WavLM-Large | Self-supervised on large unlabelled speech with a denoising objective; upper layers encode prosody and speaker state; fine-tuning beats training from scratch on 33k clips | [H-A cheat sheet]: owner's reasoning, **no comparative experiment** |
| 4 | Alternatives | Earlier MFCC + Keras `.h5` model (branch `Audio_Training`, deleted from `main`); README mentions MFCC/pitch/spectral. **No documented comparison** with wav2vec2/HuBERT ❌ | [C] branch history |
| 6 | Datasets | Stage 1: MELD 13,706; IEMOCAP 7,529; CREMA-D 7,442; TESS 2,800 (train only); RAVDESS 1,440; SAVEE 480 = **33,397**. Stage 2: DAIC-WOZ (E-DAIC 2019), **266 sessions**, PHQ-8 | [H-A] |
| 9 | Classes / order | `happy, sad, angry, fear, neutral, surprise, disgust` (indices 0–6) | [C-A] `settings.py` |
| 11 | Split | Stage 1: speaker-disjoint `GroupShuffleSplit` 85/15, seed 42, grouped by dataset + speaker; TESS forced into train → **30,510 / 2,887**. Stage 2: 5-fold (v2); `StratifiedKFold` on PHQ-8 ≥ 10 (v3-CV) | [H-A], [C-A] |
| 12 | Preprocessing | librosa 16 kHz mono → trim at 30 dB (kept if > 0.5 s remains) → peak-normalise to 0.95 → crop/tile to 6 s (random crop in training, centre crop otherwise; **tile, not zero-pad**) → training only: ±6 dB random gain, clip to [−1, 1] | [H-A], [C-A] |
| — | VAD | Offline Phase 1 standardisation: `webrtcvad`, aggressiveness 2, 30 ms frames, **drops non-speech frames**. Runtime: a WebRTC VAD gate in `audio_worker` (its parameters ❌). ⚠ The handoff says preprocessing is "identical at training and inference", **but inference does not drop VAD frames**. Verify whether the Stage 1 metadata pointed at VAD-processed files | [C-A] `audio_standardize.py`, [H-F] |
| 13–14 | Input | Raw waveform (B, 96,000) = 6 s × 16 kHz | [C-A] |
| 16 | Architecture | WavLM-Large (`microsoft/wavlm-large`; CNN feature encoder **frozen**; 24 transformer layers, **bottom 12 frozen**) → `last_hidden_state` (B, T, 1024) → **AttentionPooling** (Linear 1024→512, tanh, Linear 512→1, masked softmax) → **projection** Linear(1024→768) + LayerNorm + GELU = embedding → heads: emotion Linear(768→7); stress Linear(768→64), ReLU, Dropout 0.3, Linear(64→1), Sigmoid; confidence Linear(768→1), Sigmoid (**untrained**) | [C-A] |
| 15 | Outputs | `embedding` (768), `emotion_logits` (7), `stress` ∈ [0, 1], `confidence` (ignore) | [C-A] |
| 19–23 | Stage 1 hyperparameters | AdamW; backbone LR 1e-5, head LR 1e-4; batch 32; 15 epochs; AMP; checkpoint on best val macro-F1 | [H-A], [C-A] |
| — | Scheduler | **None visible** in `train_stage1.py` (the "warmup + cosine" claim is retracted). Gradient clipping ❌ | [H-A] |
| 24 | Loss | Cross-entropy with inverse-frequency class weights normalised to sum to 7 | [C-A] |
| 25 | Why | Class imbalance (surprise has only 80 val clips) | [ENG. REASONING] + [C-A] comment |
| — | Stage 2 (shipped) | v1 `train_stage2_stress.py` (**local only**) trains the stress head on frozen Stage 1 embeddings → `stage2_stress_best.pt`. Target = PHQ-8 / 24. Loss, LR and epochs **❌** (script not on GitHub). Whether it trained on all 266 sessions or a split is **❌, an open question that decides leakage** | [H-A] |
| — | v3-CV (not shipped) | Unfreeze top 2 WavLM layers; batch 4; backbone LR 1e-6, head LR 5e-5; MSE weighted up for PHQ-8 ≥ 10; 5-fold stratified; fixed 4 epochs; **initialised from the already DAIC-trained head → possible leakage** | [H-A] |
| 27–29 | Transfer learning | Yes. Frozen: CNN feature encoder + layers 0–11. Fine-tuned: layers 12–23 + pooling + projection + emotion head (Stage 1); stress head only (Stage 2) | [C-A], [H-A] |
| 30 | Augmentation | Random crop + random gain ±6 dB only. Noise/reverb is roadmap item #7 | [H-A] |
| 32 | Metrics | Accuracy, macro-F1, weighted-F1, per-class P/R/F1; stress: Pearson r, MAE, binary accuracy/F1 at PHQ-8 ≥ 10 | [H-A], [C-A] reports |
| 34 | Results | **65.8% accuracy, macro-F1 0.631**, weighted-F1 0.657 (2,887 speaker-disjoint val clips; chance 14.3%). Best: angry F1 0.764; worst: surprise 0.520. **Shipped stress r = 0.185 ± 0.084** (v2 CV, from a docstring; MAE etc. not recorded). r = 0.389 is exploratory and possibly leaked | [H-A], [C-A] reports |
| 35 | Weaknesses | Validation both selects and reports the checkpoint (optimistic). Mostly acted data → calm natural speech reads as "sad" (~50% per [H-F §19]). Stress is session-trained but scored on 6 s clips. Report header wrongly lists 4 datasets | [H-A], [H-F] |
| 37 | Inference | `AudioInference.predict(path)` / `predict_waveform` (**laptop only**) → same preprocessing → softmax(logits / T), T = 1.0 by default → dict | [H-A], [C-A] |
| 38 | Output | `embedding[768], emotion, emotion_probs{7}, stress, confidence(ignore)`, + `emotion_margin` (wrapper); calibrated fields when a profile is passed | [H-A] |
| 39 | Confidence | `emotion_margin` = top-1 − top-2 probability. Fusion certainty = clip(margin / ⅓, 0.25, 1) × clip(speech_ratio / 0.5, 0.4, 1), floor 0.25 | [H-A], [H-F §7.5] |
| 40 | Into fusion | Base weight 0.4; freshness 15 s; unusable if `speech_detected` is False. Stress → `_voice_stress`: median of readings over 60 s, personalised against the reading-voice reference | [H-F §7.3, §7.9] |

**Audio user calibration [C-A] `user_calibration.py`:** a baseline of YIN pitch (50–400 Hz), RMS energy, pause ratio and the embedding. Live deltas: pitch %, energy dB, pause delta, cosine similarity. **Rule-based adjustments:** +0.12 stress if pitch < −15% and pause > +0.10; +0.15 if energy > +6 dB and pitch > +20%; −0.05 if consistent with the baseline (|dB| < 2, |pitch| < 5%, cos > 0.88). It moves 25% of "sad" to "neutral" for quiet speakers. ⚠ It labels these `PROSODIC_FLATTENING_DETECTED` / "clinical markers". **That conflicts with the non-diagnostic policy if surfaced.** Verify whether the integrated system uses the module's calibration or only Fusion's own reading-voice reference.

### 4.5 Fusion as an "algorithmic model" (no trained parameters)

Fusion is rule- and formula-based. **All weights and thresholds are engineering defaults, not learned or validated** [H-F §7.14]. Full detail is in Section 7.

### 4.6 External AI services and pretrained components (no MindFlow training)

| Component | Use | Source |
|---|---|---|
| YOLOv8n-face (`yolov8n-face.pt`) | Face detection | [H-V], [C-V] |
| MediaPipe Face Landmarker / Pose Landmarker (`.task`) | Landmarks, EAR, pose | [H-V], [C-V] |
| WebRTC VAD | Speech gate | [H-F], [C-A] |
| DistilBERT SST-2 | Sentiment | [C-T] |
| DistilRoBERTa j-hartmann | Text emotion (integrated, per [H-F]) | ⚠ C2 |
| Gemini (`GEMINI_MODEL` env, e.g. `gemini-3.5-flash-lite`) | Popup wording, chatbot | [H-F §12, §16, §22] |

---

## 5. All currently known parameters

### 5.1 Video
| Parameter | Value | Src |
|---|---|---|
| Image size | 160 | C-V |
| Normalisation | ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] | C-V |
| Batch / epochs / LR / WD | 96 / 60 (early-stop 15) / 1e-4 / 2e-4 | C-V |
| Scheduler | CosineAnnealingWarmRestarts T_0 8, T_mult 2, eta_min 5e-7 | C-V |
| Label smoothing / mixup | 0.1 / α 0.3 at p 0.5 | C-V |
| Grad clip | 2.0 | C-V |
| Head dropout | 0.4 / 0.2 | C-V |
| Frames per clip (DFEW/FERV39K) | 3 (evenly spaced) | C-V |
| Dataset oversample | AffectNet 1, CK+ 8, FER-CK48 6, FER-K7 2, FER-stock 4, RAF-DB 4, DFEW 0.6, FERV39K 0.3 | C-V |
| Class extra weight | happy 1.0, sad 2.2, angry 1.6, neutral 2.0, fear 1.3, disgust 1.2, surprise 0.8 | C-V |
| Sample weight cap | 4.0 × mean | C-V, H-V |
| Window | 500 ms (2 Hz) | H-V, C-V |
| Gamma | 1.8 | C-V |
| CLAHE | clip 3.0, tile 4×4 | C-V |
| TTA / temperature | 4 views / 1.2 | C-V |
| Smoother / threshold | 5 / 0.30 | C-V |
| NeutralSuppressor | 3 windows, ≥ 0.25 | C-V |
| EAR calibration | 65% of the median resting EAR; "5 s" (⚠ C11); default 0.20 | C-V |
| PERCLOS levels | 0.15 / 0.25 / 0.40 | C-V |
| MediaPipe confidences | 0.4 | C-V |
| Track B | SEQ 16, batch 16, 40 ep, LR 3e-4, WD 5e-4, LS 0.10, GRU 128 × 1, dropout 0.4, cap 4.0, residual init 0.1 | C-V |

### 5.2 Audio
| Parameter | Value | Src |
|---|---|---|
| Sample rate / channels | 16 kHz / mono | C-A |
| Trim | top_db 30, keep if > 0.5 s | H-A, C-A |
| Peak normalisation | 0.95 | C-A |
| Clip length | 6 s (crop/tile) | C-A |
| Gain augmentation | ±6 dB | C-A |
| Offline VAD | webrtcvad aggressiveness 2, 30 ms frames | C-A |
| Frozen | CNN encoder + 12/24 layers | C-A |
| Pooling | Attention 1024→512→1 | C-A |
| Embedding | 768 (Linear + LN + GELU) | C-A |
| Stress head | 768→64→1, dropout 0.3, sigmoid | C-A |
| Stage 1 | AdamW, LR 1e-5 / 1e-4, batch 32, 15 epochs, weighted CE, AMP, select on macro-F1 | C-A |
| Split | 85/15 GroupShuffleSplit, seed 42 | C-A |
| Inference temperature | 1.0 (default) | C-A |
| Runtime window | 6 s every 1 s; results ~every 10.1 s on CPU | H-F |
| Calibration pitch | YIN 50–400 Hz | C-A |
| Checkpoint size | ~1.26 GB each | H-A |
| Backbone params | ~316M (WavLM Large). **Trainable count ❌** (the model prints it, but no log is recorded) | H-A |

### 5.3 Text
| Parameter | Value | Src |
|---|---|---|
| Stress weights | 0.45 anxious, 0.35 stressed, 0.2 frustrated, 0.35·(1 − sent_score) | C-T |
| Keyword normalisation | score / 3, cap 1; phrase hit 1.5 | C-T |
| Sentiment-fallback threshold | 0.8 | C-T |
| Fusion text window | 20 s linear fade | H-F |

### 5.4 Fusion, personalisation, temporal, decision, policy, gamification
All from [H-F]; full tables in [H-F §7.14, §8, §9, §10, §11, §16]. Key values to memorise:

- **Fusion:** base weights 0.6 / 0.4 / 0.2 (V/A/T); freshness 3 / 15 / 20 s; certainty floor 0.25; audio margin full ⅓; speech full 0.5, floor 0.4; similarity 0.7·cos + 0.3·family; neutral family 0.55; text agreement 0.7 / 0.5 / 0.3 → factors 1.0 / 0.9 / 0.75 / 0.6; negative consensus ≥ 2 modalities and ratio 0.45 → neutral × 0.20; min emotion probability 0.30; switch margin 0.10 / hold 1.25 s; voice stress window 60 s, 6 expected readings, spread 0.25; text stress share 0.4; PERCLOS full scale 0.40; windows: attention 5 s, distraction 2 s, drowsiness 20 s, fatigue 300 s; head 5 / 20 / 30°; gaze 0.10 / 0.25 / 0.50; edge score 0.75; distraction yaw/pitch 25°.
- **Baseline:** MAD × 1.4826; floors 0.05 / 3°; |z| ≥ 2; ≥ 20 samples; 1 Hz.
- **Expression profile:** ≥ 8 readings per template; JS distance; stressed template usable if LOO ≥ 0.70.
- **Temporal:** the table in [H-F §9]. Fallbacks: drowsy PERCLOS 0.25, stress 0.55, negative affect 0.55, engagement low 0.30, personal match 0.5; trends over 5 min (≥ 60 s).
- **Decision:** cooldown 60 s; dismissed ×2 (max 900 s); break after 45 focused min; encourage every 20 min; away ≥ 120 s = break.
- **Policy:** grace 2 min; gap 3 min (60 s urgent); max 6/h; duty 70%; coverage ≥ 90%; confidence ≥ 0.5; flow ≥ 0.7 over 2 min; 20-20-20 after 20 min; long session 50 min; real break 5 min; auto-close 25 s; rule table [H-F §11].
- **Feedback:** ≤ 45 words / 240 chars; Gemini timeout 6 s.
- **Gamification:** 2 points/focused min, 10/break, 25/daily goal (60 min), 15/calibration; streak tolerance 30 s.
- **Security:** PBKDF2-SHA256, 200,000 iterations, salted.
- **Runtime:** tick 1 s; start-up ~30 s; HTTPS 8443; Streamlit 8501.

---

## 6. All currently known datasets

| Dataset | Modality | Used for | Size (as documented) | Split used | Labels mapped to | Status / concerns |
|---|---|---|---|---|---|---|
| AffectNet (YOLO-format variant) | Video | Track A | ❌ | own train/valid | 7 (contempt dropped) | Which AffectNet release/variant (YOLO export)? Licence ❌ |
| CK+ | Video | Track A | ❌ | random 85/15, seed 42 | 7 (contempt dropped) | **Not subject-disjoint** |
| FER+ subsets (CK+48, kaggle7, stock2fer) | Video | Track A | ❌ | CK48/stock random 85/15; kaggle7 train/test | 7 | "FER+" here is a folder of mixed sources; kaggle3 skipped. Is "CK+48" a duplicate of CK+? ❌ |
| RAF-DB | Video | Track A | ❌ | train / **test as val** | 7 | — |
| DFEW | Video | Track A (3 frames/clip), Track B | ~15.9K clips (code comment) | fold `set_1` train/test | 7 via `DFEW_LABEL_MAP` | Label map **not spot-checked** against `annotation.xlsx` [H-V] |
| FERV39K | Video | Track A (3 frames/clip), Track B | ~39K clips (comment) | `train_All` / `test_All` | 7 | Fragile Drive-export path |
| MAFW | Video | — | — | — | — | **Dropped** (broken archives) |
| MELD | Audio | Stage 1 | 13,706 | speaker-disjoint | 7 | Audio extracted with ffmpeg |
| IEMOCAP | Audio | Stage 1 | 7,529 | speaker-disjoint | 7 | Mapping of excited/frustrated ❌ |
| CREMA-D | Audio | Stage 1 | 7,442 | speaker-disjoint | 7 | Acted |
| TESS | Audio | Stage 1 | 2,800 | **train only** | 7 | 2 speakers |
| RAVDESS | Audio | Stage 1 | 1,440 | speaker-disjoint | 7 (calm → ? ❌) | Acted |
| SAVEE | Audio | Stage 1 | 480 | speaker-disjoint | 7 | Acted, 4 male speakers ❌ |
| DAIC-WOZ / E-DAIC 2019 | Audio | Stage 2 stress | 266 sessions | 5-fold CV | PHQ-8 / 24; binary ≥ 10 | Positive rate ❌; licence/EULA ❌ |
| GoEmotions | Text | **Downloaded only** | ~58K | — | — | Not used for training |
| DepressionEmo (ID uncertain) | Text | **Downloaded only** | ❌ | — | — | Exact HF ID unknown; may have fallen back to `emotion` |
| j-hartmann training corpus | Text (integrated) | Pretrained by a third party | — | — | 7 | Per the model card; not MindFlow data |
| MindFlow logged sessions | System | Real-use stats; future replay eval | 9 users, 36 sessions, 12,436 observations, 5,790 fused states | — | — | Consent process? Labelled? ❌ |

---

## 7. All currently known algorithms

### 7.1 Fusion equations (as stated in [H-F §7]; code not on GitHub)

Notation: modality m ∈ {V, A, T}; base weight b_m; distribution p_m(e).

1. **Usable:** exists ∧ available ∧ error = None ∧ age ≤ max_age_m ∧ has scores (∧ speech_detected for audio).
2. **Certainty**
   - c_V = max(0.25, confidence_V)
   - c_A = max(0.25, clip(margin / ⅓, 0.25, 1) · clip(speech_ratio / 0.5, 0.4, 1))
   - c_T = max(0.25, max_e p_T(e)) · agree_factor · (1 − age / 20)
   - ❌ verify whether the 0.25 floor is applied before or after the text decay and agreement factors
3. **Effective weight:** w_m = b_m · c_m · 1[usable_m]
4. **Similarity:** S(i, j) = 0.7 · cos(p_i, p_j) + 0.3 · F(i, j), where F = 1 for the same valence family, 0.55 if one side is neutral, 0 if opposite (negative = {sad, angry, fear, disgust}, positive = {happy}). *Not stated: where surprise belongs ❌. Whether the family is taken from each side's argmax ❌*
5. **Text agreement:** ā = mean over live V/A of S(T, m) → factor 1.0 / 0.9 / 0.75 / 0.6 at ≥ 0.7 / 0.5 / 0.3 / below. Factor 1.0 if no V/A is live.
6. **Negative consensus:** if #{live m with a negative label} ≥ 2 and NegEv ≥ 0.45 · NeuEv, then p_m(neutral) ← 0.20 · p_m(neutral) for each neutral-reading m. From the worked example, NegEv = Σ w_m · p_m(top negative label) and NeuEv = Σ w_m · p_m(neutral) over neutral-reading modalities. ❌ **verify: top negative class only, or the sum of negative classes?** The worked example uses the top class only.
7. **Total disagreement:** if ≥ 2 voters and no pair "agrees", the label = argmax_m w_m's label. ❌ **the definition of "agree" (same label? S ≥ threshold?)**
8. **Fused distribution:** P(e) = Σ_m w_m p'_m(e) / Σ_m w_m, renormalised to sum to 1. (p' = after the neutral discount; **not** renormalised per modality before mixing. This is consistent with the worked example.)
9. **Label:** e* = argmax P. If e* ≠ neutral and P(e*) < 0.30 → neutral. Hysteresis: switch immediately if the lead ≥ 0.10, otherwise after 1.25 s on top.
10. **Confidence:** C = Σ_m (w_m / Σw) · p_m(e*) (uses the *un-discounted* p_m, per the worked example ❌ verify)
11. **All-zero weights:** P = 0, label "neutral", C = 0 → downstream treats this as "unknown".

**Why base weights summing to 1.2 is not a problem** (video open issue #1): fusion divides by Σw, so only the **ratios 3 : 2 : 1** matter. The sum is irrelevant after normalisation.

### 7.2 Stress
- Voice: s_v = median(readings in 60 s); personal value s_v' = (s_v − ref) / (1 − ref); conf_v = min(1, n / 6) · (1 − spread / 0.25) · freshness. ❌ the definitions of "spread" (range? IQR? std?) and "freshness".
- Decision stress: voice only → s_v'; text only → s_T · decay; both → (1 − σ) s_v' + σ s_T with σ = 0.4 · decay.
- Dashboard stress: confidence-weighted mean of voice, face (similarity to the stressed template over 5 s, conf × template LOO accuracy; or the neutral-relative fallback at half confidence) and text. Confidence = (1 − Π(1 − c_i)) · (1 − (max − min) / 2).

### 7.3 Dashboard parameters
- Attention per frame: min(pose score, gaze score), piecewise linear (head ≤ 5° → 1, 5–20° → 0.75, 20–30° → 0; gaze ≤ 0.10 → 1, 0.25 → 0.75, 0.50 → 0); mean over 5 s.
- Distraction: share of frames with an on-screen score < 0.5 or no face, over 2 s.
- Drowsiness: share of eyes-closed frames (calibrated EAR, only when facing the screen, |head| ≤ 20°) over 20 s ÷ 0.40, capped at 1.
- Fatigue: the same over 5 min. Focus = attention × (1 − drowsiness). Engagement = the video module's score × face share.
- Window confidence = coverage = received / expected.

### 7.4 Personalisation
- Robust z = (x − median) / max(1.4826 · MAD, floor).
- Expression templates: mean video distribution per {neutral, happy, stressed, speaking}; match by Jensen–Shannon distance; softmax over −distance / τ with τ = the typical within-template distance; LOO accuracy gate of 0.70 for "stressed". ❌ **How the template distribution is converted back into a 7-class vote for fusion.**

### 7.5 Temporal engine
E ← E + dt·c while the condition holds; E ← max(0, E − dt·decay) otherwise; "fades slowly" when unknown (rate ❌). NORMAL → SLIGHT at E ≥ slight_s → PERSISTENT at E ≥ persist_s; leave PERSISTENT only at E ≤ exit_s.

### 7.6 Decision, policy, feedback
Rule tables [H-F §10–12]. ❌ **Tie-break when several conditions are PERSISTENT** (e.g. drowsy and stress_high are both "high" priority).

### 7.7 Model-side algorithms
Weighted random sampling with a cap; mixup; label-smoothed soft CE; cosine warm restarts; TTA; temperature scaling (fixed); moving-average smoothing; neutral suppression; PERCLOS; EAR; geometric head pose (method ❌, possibly solvePnP); iris-based gaze; Farnebäck optical flow (pyr 0.5, levels 3, win 15); attention pooling; speaker-disjoint grouped split; inverse-frequency weighted CE; YIN pitch; keyword scoring.

---

## 8. All currently known design decisions

| Decision | Stated reason | Alternatives rejected | Src |
|---|---|---|---|
| Late (decision-level) fusion | Separately trained models, no joint dataset; graceful degradation; independent errors | Early fusion (needs joint data and model changes). Intermediate fusion: **not discussed** ❌ | H-F §7.1, §20 |
| Confidence- and freshness-weighted fusion | Different rates and reliabilities; missing ≠ zero | Simple average (kept as the `strategy="average"` baseline) | H-F |
| Base weights 0.6 / 0.4 / 0.2 | **No empirical derivation documented.** "Engineering defaults, not validated optima" | — | H-F §7.14 |
| Name-keyed emotions + common contract | Different label orders; decoupling | Positional arrays | H-F §6 |
| Personalisation **after** fusion | Compare with the user's own normal | ❌ | H-F §7.1 |
| Median/MAD baseline | Robust to outliers in a short calibration (Leys et al. 2013) | Mean/std | H-F §8 |
| Templates + LOO gate | Use the personal stressed face only if it is provably distinguishable | — | H-F |
| Leaky accumulators + hysteresis | Ignore single noisy readings; no flicker (debouncing) | Instantaneous logic (experiment E) | H-F §9 |
| Rules decide WHEN, AI writes WHAT | Predictability, explainability | AI deciding when | H-F §11 |
| Non-diagnostic filter + tests | Wellbeing, not medicine | — | H-F |
| Worker processes; popup process | Responsive UI, isolation, fault tolerance | Monolith | H-F §13 |
| Silence = no evidence | Not speaking says nothing about emotion | Silence = neutral | H-F |
| Store derived numbers only | Privacy; replay | Raw storage; cloud | H-F §14 |
| Local inference | Privacy (DPDP Act 2023), latency | Cloud | H-V §1, H-F §20 |
| EfficientNet-B2 over B4 | B4 worse at 2× params | B4 | H-V |
| Track B frozen backbone + residual to static average | Few parameters; small DFEW classes; anchor to a proven baseline | Unfrozen; Transformer | H-V, C-V |
| WavLM-Large, bottom 12 frozen | Pretrained prosody; memory and overfitting | ❌ no experiment | H-A |
| Attention pooling (audio) | Down-weight silence/filler | Mean pooling (no comparison logged ❌) | H-A |
| Tiling instead of zero padding | Removed a "sad" bias | Zero padding | H-A |
| Emotion margin instead of the confidence head | Head untrained | — | H-A |
| Voice stress = 60 s median, personalised | Single windows swing 0–0.89; head trained at session level | Per-window value | H-F §7.9 |
| Graded attention | Binary attention saturated at 1.0 | Binary cone | H-F |
| Eye closure only while facing the screen | False "tired" popups when looking at notes | — | H-F |
| Mobile descoped to a TFLite/ONNX on-device demo | Deadline | Native app | H-V §8 |

---

## 9. Major technical gaps (ranked by viva risk)

**THE TEAM CANNOT CURRENTLY DEFEND THESE DETAILS. SOURCE VERIFICATION IS REQUIRED.**

### Tier 1: would sink a viva answer
1. **Integrated code is not on GitHub** (`milestone-75-integration`, `1a881fe`). Nothing in fusion, decision or policy can be shown.
2. **Which text model actually runs** (C2), its stress formula, and whether the polarity defect in `1 − sentiment_score` was fixed.
3. **Audio stress number:** the Fusion handoff says 0.39; the truth for the shipped head is 0.185 (C3). Also: was the v1/v2 head trained on all 266 sessions or a split?
4. **No fused, end-to-end or state-level evaluation.** Experiments A–G have not been run. No false-alerts-per-hour figure. No latency figure apart from audio's 10.1 s.
5. **Fusion weights 0.6 / 0.4 / 0.2 have no empirical basis.** "Why these weights?" currently has only an engineering-intuition answer.
6. **Video accuracy provenance:** 74.3 / 74.4 / 74.6 (C4). The figure is measured on the same validation sets used for selection. CK+ split is not subject-disjoint. No macro-F1 or per-class F1 is recorded in the handoff.
7. **Video label order** (C1).
8. **README claims of 64.03% and >90% for video** (C5) and **MFCC/DAiSEE/feature-level fusion** in the main README (C8).

### Tier 2: precise questions you will be asked
9. Per-dataset sample counts (video), train/val sizes, and the actual number of epochs run (video Track A).
10. Parameter counts: total and trainable for B2 + head (README says 8.9M, unverified) and for WavLM (trainable count not logged).
11. Stage 2 (v1) training script: loss, LR, epochs, input embeddings.
12. Runtime WebRTC VAD settings and the definition of `speech_ratio` in `audio_worker.py`.
13. Whether the Stage 1 training audio had VAD frames removed (train/inference mismatch).
14. Exact fusion code semantics: floor order, negative-evidence definition, "agree" definition, the family of surprise, the confidence formula using discounted or raw p.
15. How the ExpressionProfile distribution maps to a 7-class video vote.
16. Spread and freshness definitions in voice-stress confidence.
17. Decision-engine tie-breaking between simultaneous persistent conditions.
18. Video worker effective PERCLOS and EAR-calibration durations (C11).
19. How the temperature T = 1.2 was chosen (any calibration/ECE measurement?).
20. Per-frame video latency (CPU, with and without TTA). "52 ms" is unverified.
21. Memory (RAM/VRAM) footprint of the running system.
22. Which stress value the policy's "stress ≥ 0.6" rule reads (dashboard stress or decision stress).

### Tier 3: completeness
23. AffectNet variant and licences (AffectNet, DAIC-WOZ, IEMOCAP and MELD all have EULAs).
24. Label mappings: IEMOCAP excited/frustrated, RAVDESS calm, MELD joy, CK+48 vs CK+ overlap.
25. DAIC-WOZ positive rate (PHQ-8 ≥ 10).
26. Keystroke status (C10).
27. Consent and ethics process for the 9 logged users.
28. Review-1 suggestions: **not in any handoff**. The team must supply them.
29. Track B diagnostic outcome (attention entropy, `residual_scale` value).
30. Test-suite breakdown (163 tests across which files).
31. Python 3.12 compatibility of the audio stack (C9).
32. Gemini model name and prompt template; banned-word list.

---

## 10. Exact source/code files to inspect to fill the gaps

### A. Integrated system (Srujana; **must be pushed first**)
| File | Fills gaps |
|---|---|
| `fusion/fusion_engine.py` | #14 (floor order, NegEv, agreement, family map, confidence), #22 |
| `fusion/config.py`, `mindflow_settings.py`, `data/settings.json` (if present) | All defaults; confirm the tables |
| `utils/observation.py`, `adapters/video_adapter.py`, `adapters/audio_adapter.py`, `adapters/text_adapter.py` | Contract; how `confidence`, `emotion_margin`, `speech_ratio`, `speech_detected` are filled |
| `runtime/audio_worker.py` | #12 VAD aggressiveness/frame size, speech_ratio, window/hop; which calibration is used |
| `runtime/video_worker.py` | #18 fps passed to trackers; TTA on/off at runtime; `perclos_window_samples` |
| `runtime/multimodal_runtime.py`, `runtime/worker_process.py` | Pipeline clock, queues, start-up |
| `personalization/expression_profile.py`, `baseline.py` | #15, τ, LOO computation |
| `decision/temporal_engine.py`, `decision_engine.py` | #17, unknown-fade rate |
| `feedback/intervention_policy.py`, `feedback_engine.py` | #22, banned words, prompt |
| `text_module/` (the integrated copy) | #2 (the actual model, stress formula) |
| `requirements-lock.txt` | #31, versions for the dependency slide |
| `tests/` (esp. `test_fusion_matrix.py`) | #30, evidence for robustness claims |
| `data/mindflow.db`, `data/sessions/*.jsonl` | Re-derive the stats in [H-F §17]; latency from timestamps |

### B. Video (Vanya; `D:\video\files` is the source of truth)
| File | Fills |
|---|---|
| `config.py` (disk copy) | #7 label order, hyperparameters |
| `checkpoints/train_log.csv` | #9 epochs actually run, best epoch, curves |
| `eval_confusion_static.py` output, `eval_per_source.py` output | #6 per-source and per-class accuracy; compute macro-F1 |
| `verify_setup.py` output | #9 class distribution, per-dataset counts |
| `inference.py` (disk copy) | #18, #19, #20 (add timing) |
| `diag_temporal_run.py` output | #29 |
| `DFEW annotation.xlsx` vs `DFEW_LABEL_MAP` | Label-map correctness |

### C. Audio (Vaishnavi)
| File | Fills |
|---|---|
| `train_stage2_stress.py`, `train_stage2_stress_v2.py` (**laptop only**) | #3, #11 |
| `audio_interface.py` with `predict_waveform` (laptop) | Inference contract |
| `metadata/metadata_<name>.csv` + `dataset_scanners.py` | #13, #24 (paths: VAD-processed or raw?) |
| Stage 1 training console log | Per-epoch curves, trainable parameters |
| `train_stage2_stress_v3_cv.py` console log | #25 positive rate |

### D. Text (Sathwik/Satwik)
| File | Fills |
|---|---|
| Integrated `text_module/*` | #2 |
| `data/text_datasets/depression_emo/` dataset_info | Which dataset actually downloaded |
| `KEYSTROKE/*`, `train_keystroke_model.py` | #26 |

### E. Team members
Review-1 suggestions and the changes made since (#28); consent process (#27); licences (#23); final decision on keystroke (#26); who answers which viva section.

---

## 11. Proposed LLD structure (to be generated after verification)

1. **Introduction:** purpose, scope, definitions (state, condition, observation, modality), references
2. **Requirements traceability:** FR/NFR table → module → class → test
3. **Architecture overview:** layered view; process view (main, video worker, audio worker, popup, server)
4. **Diagrams**
   - 4.1 Use-case diagram (Student: sign up, calibrate, study session, chat, respond to popup, view history; System actors: Gemini, devices)
   - 4.2 Master class diagram (`MultimodalRuntime`, `WorkerProcess`, adapters, `ModalityObservation`, `FusionEngine`, `FusionConfig`, `FusedObservation`, `PersonalBaseline`, `ExpressionProfile`, `TemporalStateEngine`, `DecisionEngine`, `InterventionPolicy`, `FeedbackEngine`, `GamificationEngine`, `Database`, `SessionLogger`, `ConversationManager`, `ResponseGenerator`; plus the model classes `EmotionModel`, `AudioModel`, `AudioInference`, `UserProfileCalibrator`)
   - 4.3 Sequence diagrams: (a) one second of the pipeline; (b) video frame → observation; (c) audio window → observation; (d) chat message → reply + text observation; (e) calibration wizard; (f) popup lifecycle with OK / Not now; (g) replay evaluation
   - 4.4 Package diagram (folders in [H-F §4])
   - 4.5 Deployment diagram (single Windows laptop; processes; ports 8501 / 8443; optional phone over HTTPS; Gemini over the internet)
   - 4.6 State-machine diagram (NORMAL → SLIGHT → PERSISTENT with exit hysteresis; student states)
   - 4.7 Activity diagram for fusion (steps 1–6)
5. **Module designs** (one sub-chapter per M1–M17): responsibility, classes, methods with signatures, data structures, algorithm + pseudocode, I/O specification, error handling, configuration, dependencies
6. **ML module specifications** (Video A/B, Audio S1/S2, Text): dataset → preprocessing → architecture table (layer, in/out shape, parameters) → training configuration → inference → output contract → evaluation → known limitations
7. **Data design:** `ModalityObservation`, `FusedObservation`, profile JSONs, SQLite schema (tables in [H-F §14]; columns ❌), JSONL record format
8. **Interface design:** REST endpoints (request/response schemas ❌), internal Python APIs, queue message formats
9. **Configuration:** every setting with default, unit, rationale and override path
10. **Error handling and fault tolerance:** per modality (missing, stale, error, silent), crash isolation, Gemini fallback
11. **Security and privacy:** what is never stored; PBKDF2; local-only; deletion
12. **Testing:** unit, contract, fusion matrix, replay determinism, hardware smoke tests; evaluation plan A–G
13. **Constraints, assumptions, dependencies**
14. **Traceability matrix and open issues** (from Section 9)

---

## 12. Proposed PPT structure (Progress Review, following the university template)

| # | Slide | Core content | Visual |
|---|---|---|---|
| 1 | Title | Project, team, guide | — |
| 2 | Abstract & Scope | Problem; 3 modalities; personalised, non-diagnostic, local. In/out of scope (no diagnosis, no cloud, mobile descoped) | Icon strip |
| 3 | Suggestions from Review-1 → Action taken | ❌ **Team must supply the list** | 2-column table |
| 4 | What changed since Review-1 | MFCC → WavLM; 6-class keyword text → 7-class; feature-level → late fusion; new personalisation/temporal/policy layers; DAiSEE/CogLoad dropped | Before/after |
| 5 | System architecture | The pipeline in Section 2.1 | Layered diagram |
| 6 | Design approach | Layered + modular, contract-driven, late fusion; benefits, drawbacks, alternatives | Table |
| 7 | Video model | Pipeline, B2 architecture, data, training config, results 74.6% (with caveat) | Architecture + confusion matrix |
| 8 | Video temporal (Track B) | Honest negative/ongoing result | Small table |
| 9 | Audio model | WavLM diagram, 2-stage training, 65.8% / 0.631; stress r = 0.185 | Diagram + per-class F1 bars |
| 10 | Text model | Verified pipeline (after C2 is resolved) | Flow |
| 11 | Common contract + adapters | The 11 fields; name keying | Class box |
| 12 | Fusion algorithm | Weights, certainty, freshness, conflict | Equations |
| 13 | Fusion worked example | The [H-F §7.8] numbers | Step table |
| 14 | Personalisation | Median/MAD; templates; LOO 0.61 → 0.96 | Chart |
| 15 | Temporal + decision + policy | Accumulator, hysteresis, rule table | State diagram |
| 16 | Feedback, popups, gamification, chatbot | WHEN vs WHAT | Screenshots |
| 17 | System/runtime/deployment | Processes, storage, API | Deployment diagram |
| 18 | Constraints, assumptions, dependencies | Section 10 of the final document | Table |
| 19 | Results so far | Module metrics; 163 tests; real-use stats; **no fused accuracy yet** | Tables |
| 20 | Does the approach need changing? | Yes/no per module + new approach + benefits/drawbacks | Table |
| 21 | Demonstration | Order of the demo segments | Checklist |
| 22 | Timeline for pending tasks | Evaluation A–G, Track B diagnostic, stress rerun, text verification, test set | Gantt to 26 Oct |
| 23 | Conclusion | 3 takeaways | — |
| 24 | References | Baltrušaitis 2019; Leys 2013; Lin 1991; WavLM; EfficientNet; dataset papers; PERCLOS | — |

---

## 13. Team training syllabus (12 sessions)

Each session: **learn → answer → draw from memory → memorise → equations → demo.** About 60–90 minutes each. The module owner teaches and a different member presents it back.

| # | Session | Topics | Answer without notes | Draw | Memorise | Equations | Demo practice |
|---|---|---|---|---|---|---|---|
| 1 | Overall architecture | Section 2 pipeline; processes; one second of MindFlow | "Walk me from webcam frame to popup" | Pipeline + process diagram | Rates: 2 Hz / ~10 s / per message; tick 1 s | — | `main.py` start-up, dashboard |
| 2 | Requirements | FR/NFR derivation; privacy; non-diagnostic | "Which requirement drives late fusion?" | Use-case diagram | Never stored: frames, audio, embeddings, text | — | Show `data/` contents (derived only) |
| 3 | Video | Track A pipeline, training, inference, outputs; Track B status | 40-question matrix (4.1) | B2 + head; inference chain | 160 px, batch 96, LR 1e-4, T 1.2, window 5, 0.30, γ 1.8, PERCLOS levels | Softmax with T; EAR; PERCLOS; engagement formula | `inference.py --webcam --consent` |
| 4 | Audio | WavLM, pooling, 2 stages, calibration, margin | 4.4 matrix; "why is stress r low?" | AudioModel diagram | 33,397 / 30,510 / 2,887; 65.8% / 0.631; r = 0.185; 6 s, 16 kHz, 0.95, 30 dB | Attention pooling; margin; PHQ-8/24 | Gradio dashboard with 3 backup WAVs |
| 5 | Text | Preprocessing, SST-2, emotion model (after C2), stress | "Is `sentiment_score` a confidence?" | Text flow | Stress weights; 20 s fade | Stress formula | Chat message → state change |
| 6 | Fusion | All of Section 7.1–7.2 | "Why late? Why 0.6/0.4/0.2? What if all disagree?" | Fusion activity diagram | Every value in §5.4 Fusion | All 11 equations; redo the worked example by hand | Replay one session with `average` vs `confidence` |
| 7 | Personalisation + temporal | Median/MAD, templates, LOO, accumulators | "Why MAD? Why hysteresis?" | Accumulator state machine | 1.4826; |z| ≥ 2; ≥ 20 samples; template LOO 0.70; temporal table | Robust z; JS distance; accumulator update | Calibration wizard |
| 8 | Decision + interventions | States, actions, policy rules, feedback, gamification | "How do you avoid false alarms?" (3 layers) | Decision table | Grace 2 min, gap 3 min, 6/h, duty 70%; points | Duty/coverage | Trigger a "distracted" popup; click Not now |
| 9 | System architecture | Patterns, processes, contract, storage, API | "Name 5 design patterns and where they are used" | Class + package diagrams | Endpoints; tables; PBKDF2 200k | — | FastAPI `/api/state` |
| 10 | Deployment | venv, models, ports, HTTPS, Gemini fallback | "What happens offline?" | Deployment diagram | Python 3.12, ports 8501/8443, `.env` | — | Fresh-machine run from a checklist |
| 11 | Results + limitations | What is measured vs not; honest numbers; gaps | "What is your fused accuracy?" → "Not measured yet; the replay harness exists; experiments A–G pending" | Results table | Every number in §9 and §25 of [H-F], corrected | — | `pytest` run (163 tests) |
| 12 | Full mock viva | Rotate roles; panel asks across modules | Random question from the bank | Any diagram on demand | — | — | Full integrated demo + backup plan |

---

## Fusion worked example: independent re-check

I recomputed [H-F §7.8] by hand using the stated equations. It **reproduces exactly**: w = 0.420 / 0.288 / 0.105; cos(T,V) = 0.699, cos(T,A) = 0.952; agreement 0.811; fused distribution sad 0.547, neutral 0.199, fear 0.119, angry 0.102, happy 0.033; confidence 0.42. Reproducing those numbers requires three things:

1. the neutral discount is applied **without renormalising the video distribution** before mixing;
2. "negative evidence" uses each negative modality's **top** negative class weighted by w;
3. the confidence uses the **raw** (un-discounted) video p(sad) = 0.30.

These three semantic details must be confirmed in `fusion_engine.py` (gap #14).

---

**Next step:** the team answers or verifies the items in Section 9 (start with Tier 1 and conflicts C1–C11). Then the final LLD, PPT, demo plan, question bank and cheat sheet are generated from verified facts only.
