# MindFlow TEXT/NLP to Fusion Handoff

**Recipient:** Srujana, Fusion developer  
**Date inspected:** 2026-08-13  
**Scope:** Current implementation only. No NLP redesign or retraining is assumed.

## 1. CURRENT TEXT PIPELINE

Source files: `TEXT/text_preprocess.py`, `TEXT/text_model.py`, and `TEXT/text_pipeline.py`.

```text
Raw text
  -> clean_text(raw_text)
  -> DistilBERT sentiment analysis
  -> keyword/rule-based emotion scoring
  -> heuristic stress calculation
  -> run_text_pipeline() dictionary
```

### Preprocessing

`clean_text()` performs the following operations, in order:

1. `None` becomes the empty string `""`.
2. Every other input is converted with `str()`.
3. Text is lowercased.
4. Every character not matching `[a-z0-9\\s]` is replaced with a space.
5. Consecutive whitespace is collapsed to one space.
6. Leading and trailing whitespace is removed.

Consequences:

| Input characteristic | Actual behavior |
|---|---|
| Uppercase | Converted to lowercase |
| ASCII letters | Preserved |
| ASCII digits `0-9` | Preserved |
| Punctuation | Replaced with spaces |
| Apostrophes | Replaced with spaces; `can't` becomes `can t` |
| Special characters | Replaced with spaces |
| Emojis | Replaced with spaces |
| Non-English/non-ASCII characters | Replaced with spaces |
| Multiple spaces | Collapsed |
| `None` | Becomes empty string |
| Empty or whitespace-only input | Remains empty string after cleaning |

There is no minimum-token check, lemmatization, stopword removal, emoji handling, or negation handling.

### Tokenizer and model

The code uses the Hugging Face `pipeline("sentiment-analysis", ...)`, which automatically loads the tokenizer associated with:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

The model is a pretrained DistilBERT model fine-tuned by its original publisher for SST-2 binary sentiment classification. There is no evidence that MindFlow fine-tuned it. The model is not trained for the six MindFlow emotion categories and is not a stress classifier.

The cached model configuration reports:

- Maximum positional sequence length: `512`
- Tokenizer model maximum length: `512`
- Lowercase tokenizer: yes
- Model labels: `NEGATIVE`, `POSITIVE`

The inference call specifies `truncation=True`. The code does not specify an explicit `max_length`; the model/tokenizer default limit is therefore used. The code does not specify padding. No device is specified, so the Hugging Face pipeline defaults to CPU.

The model-level output is approximately:

```python
{"label": "POSITIVE" or "NEGATIVE", "score": float}
```

The source code does not explicitly name a tokenizer class. Hugging Face `pipeline()` automatically selects the tokenizer associated with the model. `test_text.py` separately requests `AutoTokenizer.from_pretrained()` for the same model name. Therefore, the exact tokenizer class is not explicitly fixed by the TEXT implementation; the model-associated DistilBERT uncased tokenizer is used when available.

If model loading fails, `load_sentiment_model()` returns `None` and the code uses its keyword-based fallback sentiment scorer. Loading exceptions are caught; inference exceptions are not caught by `text_pipeline.py`.

### Emotion scoring

Emotion scoring is entirely local keyword/rule scoring. It is not a trained emotion model. The exact categories are:

```text
anxious, stressed, calm, motivated, frustrated, focused
```

For each requested label, the code:

1. Lowercases the cleaned text.
2. Extracts tokens with `[a-z']+`.
3. Normalizes simple suffixes: `ing`, `ed`, `ly`, `es`, and `s`.
4. Counts matching single-word keywords.
5. Matches multiword keywords as substrings.
6. Adds category-specific stress/support boosts.
7. Divides by `3.0`.
8. Caps the result at `1.0`.
9. Rounds it to four decimal places.

Multiple categories can be active simultaneously. Scores are heuristic normalized scores, not probabilities.

### Stress calculation

The current pipeline calculates:

```text
stress_score = clamp(
    0.45 * anxious_score
  + 0.35 * stressed_score
  + 0.20 * frustrated_score
  + 0.35 * max(0, 1 - sentiment_score),
    0, 1
)
```

This is a project heuristic. It is not the output of a trained stress model.

### Sentiment output details

The sentiment classes are exactly `POSITIVE` and `NEGATIVE`. There is no `NEUTRAL` class in the SST-2 model or in the current pipeline. The model's `score` is the probability-like predicted-class score returned by the Hugging Face pipeline. The pipeline stores that value unchanged, rounded to four decimals, as `sentiment_score`.

For a `POSITIVE` result, the score represents the model score for `POSITIVE`; for a `NEGATIVE` result, it represents the model score for `NEGATIVE`. It is not converted to a signed sentiment value. The pipeline determines polarity by uppercasing the model's `label`.

If the transformer cannot be loaded, fallback sentiment counts words from its hardcoded positive and negative sets. The fallback chooses `POSITIVE` when positive count is greater than or equal to negative count and returns `(positive + 1) / (positive + negative + 2)`. Otherwise it chooses `NEGATIVE` and returns `(negative + 1) / (positive + negative + 2)`. This fallback value is also a keyword-derived score, not calibrated confidence.

Concrete example: the actual input `I am stressed about exams and deadlines.` produced `sentiment_polarity: NEGATIVE` and `sentiment_score: 0.996`; this means the loaded SST-2 classifier assigned a score of `0.996` to its `NEGATIVE` class.

## 2. EXACT OUTPUT SCHEMA

`run_text_pipeline(raw_text)` returns exactly these nine top-level fields:

| Field | Type | Range/values | Actual meaning | Fusion status |
|---|---|---|---|---|
| `sentiment_polarity` | `str` | `POSITIVE` or `NEGATIVE` | Predicted SST-2 sentiment label | Potential candidate |
| `sentiment_score` | `float` | `0.0-1.0` | Score for the predicted sentiment class | Potential candidate; not general confidence |
| `stress_score` | `float` | `0.0-1.0` | Current heuristic text stress score | Strong candidate, with limitations |
| `estimated_sentiment_accuracy` | `str` | For example `92-98%` | Heuristic display range from sentiment score | Do not use for Fusion |
| `estimated_overall_text_accuracy` | `str` | For example `87-93%` | Heuristic display range from sentiment and emotion separation | Do not use for Fusion |
| `anxiety_prob` | `float` | `0.0-1.0` | Heuristic score for the `anxious` category | Potential candidate; not a probability |
| `emotional_tone` | `str` | Six labels, or fallback `calm` | Highest-scoring emotion label | Potential categorical feature |
| `motivation_level` | `float` | `0.0-1.0` | Heuristic score for `motivated` | Potential candidate |
| `all_emotions` | `dict[str, float]` | Six keys; each `0.0-1.0` | All keyword/rule-based emotion scores | Strong candidate as evidence vector |

The pipeline does not return `timestamp`, `datetime`, `modality`, `confidence`, `window_ms`, `features`, `error`, `modality_present`, `text_length`, `id`, or `mask`.

## 3. FUSION-RELEVANT FIELDS

### Exact emotion keywords

```python
EMOTION_KEYWORDS = {
    "anxious": (
        "anxious", "anxiety", "worried", "worry", "nervous", "panic",
        "panicked", "uneasy", "overwhelmed", "restless", "afraid", "fearful",
    ),
    "stressed": (
        "stressed", "stress", "pressure", "deadline", "deadlines", "exam",
        "exams", "workload", "burnout", "exhausted", "fatigued", "tired",
    ),
    "calm": (
        "calm", "relaxed", "peaceful", "steady", "balanced", "clear",
        "centered", "okay", "fine", "stable",
    ),
    "motivated": (
        "motivated", "determined", "driven", "focused", "consistent",
        "productive", "ready", "inspired", "improve", "improving", "progress",
    ),
    "frustrated": (
        "frustrated", "annoyed", "angry", "upset", "irritated", "stuck",
        "drained", "fed up", "hopeless", "depressed", "sad", "helpless",
    ),
    "focused": (
        "focused", "concentrate", "concentrated", "attentive", "organized",
        "on track", "alert", "productive", "discipline", "goal",
    ),
}
```

Additional stress boosts are applied to `anxious`, `stressed`, and `frustrated`:

```text
deadline/deadlines: 0.8
exam/exams: 0.8
pressure: 0.7
overwhelmed: 0.9
panic: 1.0
stress: 1.0
burnout: 0.9
exhausted: 0.9
hopeless: 1.0
```

Additional support boosts are applied to `calm`, `motivated`, and `focused`:

```text
calm: 0.5
ready: 0.5
focus: 0.7
focused: 0.8
motivated: 0.8
improve: 0.7
consistent: 0.6
productive: 0.7
```

The fallback emotion profile is used when no explicit emotion cue is found and sentiment is available. It can produce nonzero values even for empty or meaningless text.

### Fusion classification

| Output | Classification | Reason |
|---|---|---|
| `stress_score` | A. Strong candidate | Direct text-side stress evidence, but heuristic |
| `all_emotions` | A. Strong candidate | Complete current text emotion evidence |
| `anxiety_prob` | B. Potential candidate | Anxiety lexical evidence, but not a probability |
| `sentiment_polarity` | B. Potential candidate | Useful categorical sentiment evidence |
| `sentiment_score` | B. Potential candidate | Predicted-class score; its direction depends on polarity |
| `emotional_tone` | B. Potential candidate | Dominant heuristic category |
| `motivation_level` | B. Potential candidate | Motivation lexical evidence |
| `estimated_sentiment_accuracy` | D. Do not use | Not measured accuracy or confidence |
| `estimated_overall_text_accuracy` | D. Do not use | Not measured accuracy or confidence |

The current TEXT code does not directly predict Focus, Productivity, Well-being, Burnout, or Risk.

## 4. CONFIDENCE STATUS

TEXT MODULE CURRENTLY DOES NOT PROVIDE A RELIABLE GENERAL CONFIDENCE FIELD.

`sentiment_score` comes from the DistilBERT sentiment pipeline and is a score for the predicted class. It is not a general reliability value. For a negative prediction, `0.996` means high confidence in `NEGATIVE`, not positive sentiment `0.996`.

The two `estimated_*_accuracy` fields are computed by `_estimate_display_accuracy()` from the sentiment score and emotion-score separation. They are uncalibrated display heuristics, not benchmark accuracy and not model confidence.

Fusion should retain `sentiment_score` as classifier evidence, but should not use it as the single modality confidence. No trained or calibrated confidence field exists in the current TEXT module.

## 5. TIMESTAMP/WINDOW STATUS

TEXT PIPELINE DOES NOT CURRENTLY GENERATE A TIMESTAMP.

The pipeline has no timestamp, datetime, observation time, event time, `window_ms`, duration, session duration, or sequence timing.

The integration adapter can attach timestamp and window metadata externally. The timestamp should be generated at the point where the text observation is received or adapted. The window policy must be decided by Fusion; it is not present in the current NLP implementation.

## 6. MISSING-MODALITY/ERROR STATUS

### Actual edge-case behavior

With the locally cached model, the following inputs all return a dictionary:

| Input | Current result |
|---|---|
| `None` | Converted to empty text and inferred |
| `""` | Empty text is inferred |
| `"   "` | Cleaned to empty text and inferred |
| Very short text | Inferred normally |
| Punctuation-only text | Cleaned to empty text and inferred |
| Emoji-only or non-English text | Cleaned to empty text and inferred |
| Apostrophe text | Apostrophes are removed before inference |

Empty input does not return zero values or an error. With the available model it returned a default positive prediction and nonzero fallback emotion values. These outputs should be treated as invalid/missing by the adapter rather than as meaningful evidence.

### Error behavior

- Model loading exceptions are caught and trigger keyword sentiment fallback.
- `text_pipeline.py` has no `try/except` around the complete pipeline.
- Inference exceptions propagate to the caller.
- There is no built-in `error` field.
- There is no built-in `modality_present` field.
- There is no built-in missing-text marker.

The adapter should validate input before inference and catch exceptions around `run_text_pipeline()`.

## 7. SAMPLE OUTPUTS

These were executed on 2026-08-13 with the locally cached model `distilbert-base-uncased-finetuned-sst-2-english`.

### 1. `I feel motivated and ready to study.`

```python
{
    "sentiment_polarity": "POSITIVE",
    "sentiment_score": 0.9995,
    "stress_score": 0.0002,
    "estimated_sentiment_accuracy": "92-98%",
    "estimated_overall_text_accuracy": "90-96%",
    "anxiety_prob": 0.0,
    "emotional_tone": "motivated",
    "motivation_level": 1.0,
    "all_emotions": {
        "anxious": 0.0, "stressed": 0.0, "calm": 0.4333,
        "motivated": 1.0, "frustrated": 0.0, "focused": 0.4333
    }
}
```

### 2. `I am stressed about exams and deadlines.`

```python
{
    "sentiment_polarity": "NEGATIVE",
    "sentiment_score": 0.996,
    "stress_score": 1.0,
    "estimated_sentiment_accuracy": "92-98%",
    "estimated_overall_text_accuracy": "87-93%",
    "anxiety_prob": 1.0,
    "emotional_tone": "anxious",
    "motivation_level": 0.0,
    "all_emotions": {
        "anxious": 1.0, "stressed": 1.0, "calm": 0.0,
        "motivated": 0.0, "frustrated": 1.0, "focused": 0.0
    }
}
```

### 3. `Today I feel calm and focused.`

```python
{
    "sentiment_polarity": "POSITIVE",
    "sentiment_score": 0.9996,
    "stress_score": 0.0001,
    "estimated_sentiment_accuracy": "92-98%",
    "estimated_overall_text_accuracy": "87-93%",
    "anxiety_prob": 0.0,
    "emotional_tone": "calm",
    "motivation_level": 1.0,
    "all_emotions": {
        "anxious": 0.0, "stressed": 0.0, "calm": 1.0,
        "motivated": 1.0, "frustrated": 0.0, "focused": 1.0
    }
}
```

### 4. `I feel tired and emotionally overwhelmed.`

```python
{
    "sentiment_polarity": "NEGATIVE",
    "sentiment_score": 0.9993,
    "stress_score": 0.5669,
    "estimated_sentiment_accuracy": "92-98%",
    "estimated_overall_text_accuracy": "83-89%",
    "anxiety_prob": 0.6333,
    "emotional_tone": "anxious",
    "motivation_level": 0.0,
    "all_emotions": {
        "anxious": 0.6333, "stressed": 0.6333, "calm": 0.0,
        "motivated": 0.0, "frustrated": 0.3, "focused": 0.0
    }
}
```

### 5. `I can improve if I stay consistent every day.`

```python
{
    "sentiment_polarity": "POSITIVE",
    "sentiment_score": 0.9971,
    "stress_score": 0.001,
    "estimated_sentiment_accuracy": "92-98%",
    "estimated_overall_text_accuracy": "90-96%",
    "anxiety_prob": 0.0,
    "emotional_tone": "motivated",
    "motivation_level": 1.0,
    "all_emotions": {
        "anxious": 0.0, "stressed": 0.0, "calm": 0.4333,
        "motivated": 1.0, "frustrated": 0.0, "focused": 0.4333
    }
}
```

## 8. COMMON-CONTRACT GAP ANALYSIS

| Contract field | Does Text provide it? | Exact existing field | Required integration behavior |
|---|---|---|---|
| `timestamp` | No | None | Add externally in adapter |
| `modality` | No | None | Add `"text"` in adapter |
| `emotion` | Partially | `emotional_tone` | Map without changing source semantics |
| `confidence` | No reliable value | `sentiment_score` is classifier score only | Do not invent; retain separately |
| `all_scores` | Partially | `all_emotions` plus scalar fields | Package existing values in adapter |
| `window_ms` | No | None | Add only if Fusion defines a window |
| `features` | No | None | Adapter may package text evidence here |
| `error` | No | None | Adapter should catch and expose failures |
| `modality_present` | No | None | Adapter should set after input validation |
| `stress` evidence | Yes | `stress_score` | Pass through as heuristic text stress evidence |
| Raw text | No | None | Keep absent from Fusion logs |

## 9. MINIMUM CHANGES REQUIRED

No changes to the NLP model are required for basic integration.

The minimum adapter behavior is:

1. Reject `None`, empty, whitespace-only, and punctuation-only text as unavailable.
2. Call `run_text_pipeline()` only for valid text.
3. Add `timestamp` externally.
4. Add `modality: "text"`.
5. Add `modality_present`.
6. Preserve all current text fields without relabeling them.
7. Add `error: None` on success or an error description on failure.
8. Add `window_ms` only after Fusion defines the temporal policy.
9. Do not use either `estimated_*_accuracy` field as confidence.
10. Do not store raw text in the shared Fusion observation.

The minimum additional tests are:

- Exact nine-key output schema.
- Numeric range checks.
- Valid sample text.
- `None`, empty, whitespace-only, and punctuation-only text.
- Model fallback behavior.
- Adapter timestamp/modality/presence fields.
- Adapter exception-to-error behavior.
- Confirming that accuracy strings are not consumed as confidence.

Existing automated validation: `tests/test_text_pipeline_accuracy.py` passed with `1 passed` and one transformer deprecation warning.

### Already available from Satwik

- Current preprocessing implementation
- Model name and local cached availability
- Current tokenizer/model behavior
- Exact output dictionary
- Exact emotion categories and keyword/rule scoring
- Current stress formula
- Actual sample outputs
- Existing automated stress test

### Missing but required from Satwik/Fusion agreement

- Official timestamp format
- Observation ID policy, if Fusion requires one
- Empty/invalid text policy
- Error policy
- Whether Fusion consumes `stress_score` directly or transforms it
- Observation/window aggregation policy

### Optional

- Held-out benchmark metrics
- Sentiment calibration
- Project-trained stress classifier
- Trained emotion classifier
- Formal reliability model

### Not needed for basic Fusion integration

- NLP redesign
- DistilBERT retraining
- New datasets
- Raw text storage

## 10. FINAL HANDOFF TO SRUJANA

### WHAT SRUJANA SHOULD CONSUME

Use these current fields:

| Field | Type | Meaning | Recommended Fusion use |
|---|---|---|---|
| `stress_score` | `float`, `0-1` | Heuristic text stress evidence | Primary text stress feature |
| `all_emotions` | `dict[str, float]` | Six heuristic emotion scores | Preserve as text feature vector |
| `sentiment_polarity` | `str` | DistilBERT sentiment class | Categorical evidence |
| `sentiment_score` | `float`, `0-1` | Predicted-class DistilBERT score | Model evidence; not general confidence |
| `anxiety_prob` | `float`, `0-1` | Heuristic anxious score | Anxiety-related evidence only |
| `emotional_tone` | `str` | Highest heuristic emotion | Categorical evidence |
| `motivation_level` | `float`, `0-1` | Heuristic motivated score | Motivation-related evidence |

### Fusion-ready adapter output

For a valid text observation, Srujana can adapt the current output into this shape:

```python
{
    "timestamp": "<generated by integration layer>",
    "modality": "text",
    "modality_present": True,
    "emotion": text_output["emotional_tone"],
    "stress_score": text_output["stress_score"],
    "sentiment_polarity": text_output["sentiment_polarity"],
    "sentiment_score": text_output["sentiment_score"],
    "anxiety_prob": text_output["anxiety_prob"],
    "motivation_level": text_output["motivation_level"],
    "all_emotions": text_output["all_emotions"],
    "features": {
        "stress_score": text_output["stress_score"],
        "sentiment_score": text_output["sentiment_score"],
        "anxiety_prob": text_output["anxiety_prob"],
        "motivation_level": text_output["motivation_level"],
        "all_emotions": text_output["all_emotions"],
    },
    "window_ms": "<only if defined externally>",
    "error": None,
}
```

For invalid text:

```python
{
    "timestamp": "<generated by integration layer>",
    "modality": "text",
    "modality_present": False,
    "emotion": None,
    "features": None,
    "window_ms": "<only if defined externally>",
    "error": "missing_or_empty_text",
}
```

For an inference failure:

```python
{
    "timestamp": "<generated by integration layer>",
    "modality": "text",
    "modality_present": False,
    "emotion": None,
    "features": None,
    "window_ms": "<only if defined externally>",
    "error": "text_pipeline_inference_failed",
}
```

The adapter should not add a fabricated `confidence`, timestamp, window size, accuracy, stress model, or emotion probability. The current TEXT module supplies text-side evidence only; Fusion remains responsible for combining it with Video and Audio evidence to estimate Focus, Stress, Productivity, Well-being, and risk indicators.

## Provided-file and artifact status

The inspected TEXT files are `TEXT/text_model.py`, `TEXT/text_pipeline.py`, `TEXT/text_preprocess.py`, `TEXT/text_datasets.py`, `TEXT/__init__.py`, `test_text.py`, and `tests/test_text_pipeline_accuracy.py`. The relevant configuration is `requirements.txt`; relevant documentation includes `README.md`, `TEXT_AND_KEYSTROKE_MODALITY_REPORT.md`, `TEXT_DATASET_AND_PREPROCESSING_OVERVIEW.md`, `STRESS_ESTIMATOR_TABLE.md`, and `novelty/text_section/README.md`.

The following requested items are not implemented or present in the repository: a project-local TEXT checkpoint, sample-output file, TEXT configuration file, Fusion implementation, and a non-empty text-section README. The repository contains a file named `FUSION`, but it is whitespace-only rather than a Fusion directory or implementation. The Hugging Face model used during testing exists in the machine's external local cache, not as a repository checkpoint.
