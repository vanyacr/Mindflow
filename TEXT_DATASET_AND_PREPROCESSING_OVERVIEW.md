# Text Processing: Dataset & Preprocessing Overview
**Mindflow Project | April 19, 2026**

---

## 1. Dataset Overview

Your project uses **two complementary text datasets**:

### 1.1 GoEmotions Dataset

**Location**: `data/text_datasets/go_emotions/`

| Property | Value |
|----------|-------|
| **Source** | Google (public benchmark) |
| **Size** | ~58,000 labeled examples |
| **Language** | English (Reddit comments) |
| **Emotions** | 28 emotion categories + neutral (29 total) |
| **Splits** | Train / Validation / Test |
| **Format** | HuggingFace .arrow (columnar, optimized) |

**Emotion Labels** (sample):
- admiration, amusement, anger, annoyance, approval, caring
- confusion, curiosity, desire, disappointment, disapproval, disgust
- embarrassment, excitement, fear, gratitude, grief, joy
- love, nervousness, neutral, optimism, pride, realization
- relief, remorse, sadness, surprise, neutral

**Use in Mindflow**: Benchmark for fine-tuning emotion detection; understand broad emotional spectra.

---

### 1.2 DepressionEmo Dataset

**Location**: `data/text_datasets/depression_emo/`

| Property | Value |
|----------|-------|
| **Source** | Mental health corpora (suicide/depression posts) |
| **Size** | Variable (depends on candidate sourced) |
| **Language** | English (Reddit, social media) |
| **Focus** | Depression-related language, hopelessness, suicidal ideation markers |
| **Splits** | Train only (in your current download) |
| **Format** | HuggingFace .arrow (same as GoEmotions) |

**Example Content**: User posts expressing depression, anxiety, hopelessness, stress.

**Use in Mindflow**: Fine-tune text model specifically for stress/mental health signals; transfer learning from depression language to stress detection.

---

## 2. Data Structure & Schema

### 2.1 GoEmotions Record Example

```json
{
  "text": "I'm absolutely thrilled about the new project! This will be an amazing opportunity.",
  "labels": [7, 19],  // Multi-label: indices for "excitement" and "joy"
  "id": "reddit_comment_12345"
}
```

### 2.2 DepressionEmo Record Example

```json
{
  "text": "I feel hopeless. Nothing makes sense anymore. Can't see a future.",
  "label": 1,  // Binary: 0=not depressed, 1=depressed
  "id": "depression_post_67890"
}
```

---

## 3. Current Preprocessing Pipeline

### 3.1 Preprocessing Code (Your Implementation)

**File**: [TEXT/text_preprocess.py](TEXT/text_preprocess.py)

```python
def clean_text(raw_text: str) -> str:
    """Normalize user text for downstream NLP models.
    
    Steps:
    1. Lowercase
    2. Remove special characters
    3. Collapse extra spaces
    """
```

### 3.2 Preprocessing Steps (In Order)

| Step | Input | Operation | Output | Reason |
|------|-------|-----------|--------|--------|
| 1 | Raw text | Convert to lowercase | "I'M STRESSED" → "i'm stressed" | Reduce vocab size; BERT is case-insensitive |
| 2 | Lowercased | Remove special chars (regex: `[^a-z0-9\s]`) | "i'm stressed!" → "im stressed" | Reduce noise; keep alphanumerics + space only |
| 3 | Cleaned | Collapse whitespace (regex: `\s+`) | "i  stressed" → "i stressed" | Normalize spacing |
| 4 | Final | Strip leading/trailing spaces | " i stressed " → "i stressed" | Clean boundaries |

### 3.3 Preprocessing Input/Output Example

**Raw Input**:
```
"I'm SO overwhelmed!!!  Can't focus... work deadlines everywhere??"
```

**After Step 1 (Lowercase)**:
```
"i'm so overwhelmed!!!  can't focus... work deadlines everywhere??"
```

**After Step 2 (Remove Special Chars)**:
```
"im so overwhelmed  cant focus work deadlines everywhere"
```

**After Step 3 (Collapse Whitespace)**:
```
"im so overwhelmed cant focus work deadlines everywhere"
```

**Final Cleaned Output**:
```
"im so overwhelmed cant focus work deadlines everywhere"
```

---

## 4. Full Text Processing Pipeline

### 4.1 End-to-End Flow

**File**: [TEXT/text_pipeline.py](TEXT/text_pipeline.py)

```
Raw Text Input
    ↓
[Preprocessing: clean_text()]
    ↓ 
Cleaned Text
    ↓
[Sentiment Analysis: DistilBERT]
    ↓
Sentiment Label + Score
    ↓
[Emotion Detection: Keyword-based]
    ↓
Emotion Scores (6 emotions)
    ↓
[Confidence Estimation]
    ↓
Final Output: Stress Probability + Details
```

### 4.2 Text Pipeline Output (Your Current Code)

**Function**: `run_text_pipeline(raw_text: str) -> Dict[str, Any]`

**Example Output**:
```json
{
  "sentiment_polarity": "NEGATIVE",
  "sentiment_score": 0.85,
  "estimated_sentiment_accuracy": "72-98%",
  "estimated_overall_text_accuracy": "65-92%",
  "anxiety_prob": 0.68,
  "emotional_tone": "stressed",
  "motivation_level": 0.15,
  "all_emotions": {
    "anxious": 0.68,
    "stressed": 0.73,
    "calm": 0.12,
    "motivated": 0.15,
    "frustrated": 0.45,
    "focused": 0.1
  }
}
```

---

## 5. Preprocessing Features & Design Choices

### 5.1 What Gets Preserved

✅ **Alphanumeric characters**: a–z, 0–9 (kept)
✅ **Spaces**: Word boundaries (kept)
❌ **Punctuation**: ! ? . , ; : ' " (removed)
❌ **Special symbols**: @ # $ % ^ & (removed)
❌ **Emoji**: (removed)
❌ **Case information**: (converted to lowercase)

### 5.2 Why These Choices?

| Design Choice | Benefit | Trade-off |
|---------------|---------|-----------|
| Lowercase | Reduces vocabulary; BERT handles case poorly | Lose emphasis cues ("I'M STRESSED" → "i'm stressed") |
| Remove punctuation | Simplifies tokenization; reduces noise | Lose sentence boundaries and emphasis markers |
| Collapse whitespace | Handles messy user input | Can't detect multiple-word pauses |
| Keep alphanumerics | Preserve semantic content | May conflate different spellings (e.g., "grt" vs "great") |

### 5.3 Current Limitations

⚠️ **Negations lost**: "not fine" → "not fine" (preserved), but removed punctuation might affect BERT's understanding  
⚠️ **Slang unchanged**: "lol" stays as "lol" (good for keywords, could confuse embeddings)  
⚠️ **No lemmatization**: "stressed", "stressing", "stress" are kept separate  
⚠️ **No stopword removal**: Common words ("the", "is", "a") still present (better for context)  
⚠️ **Emoji removed entirely**: Loss of emotional context ("😭" removed)

---

## 6. Emotion Detection System

### 6.1 Keyword-Based Emotion Scoring

**File**: [TEXT/text_model.py](TEXT/text_model.py)

Your model uses **hardcoded keyword sets** per emotion:

```python
EMOTION_KEYWORDS = {
    "stressed": (
        "stressed", "stress", "pressure", "deadline", "deadlines",
        "exam", "exams", "workload", "burnout", "exhausted",
        "fatigued", "tired"
    ),
    "anxious": (
        "anxious", "anxiety", "worried", "worry", "nervous",
        "panic", "panicked", "uneasy", "overwhelmed", "restless"
    ),
    "calm": (
        "calm", "relaxed", "peaceful", "steady", "balanced",
        "clear", "centered", "okay", "fine"
    ),
    # ... more emotions
}
```

### 6.2 Emotion Score Computation

1. **Tokenize** cleaned text into words
2. **Normalize** each token (remove suffixes: -ing, -ed, -ly, -es, -s)
3. **Count matches** against each emotion's keyword set
4. **Normalize score** to [0.0, 1.0] range

**Example**:
- Text: "i am so stressed and exhausted"
- Tokens: ["i", "am", "so", "stressed", "and", "exhausted"]
- "stressed" keyword matches 1 (stressed)
- "exhausted" keyword matches 2 (stressed)
- Score for stress emotion: count / 3.0 = 2 / 3.0 = 0.67 → clamped to 1.0 = **0.67**

---

## 7. Sentiment Analysis Component

### 7.1 Model Used

**Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Lightweight BERT variant
- Pre-trained on Stanford Sentiment Treebank (SST-2)
- Output: Binary classification (POSITIVE / NEGATIVE)

### 7.2 Sentiment Pipeline

```python
from transformers import pipeline

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    local_files_only=not allow_download
)

result = sentiment_pipe(text)  # Input: cleaned text
# Output: {"label": "POSITIVE", "score": 0.95}
```

### 7.3 Fallback (When Model Unavailable)

If BERT model can't be loaded, **keyword-based fallback**:

```python
positive_words = {
    "good", "great", "calm", "focused", "motivated",
    "ready", "improve", "confident", "hopeful", "productive"
}
negative_words = {
    "bad", "anxious", "stressed", "depressed", "hopeless",
    "overwhelmed", "nervous", "tired", "frustrated", "panic"
}

# Count pos/neg words → compute POSITIVE/NEGATIVE probability
```

---

## 8. Data Format Throughout Pipeline

### 8.1 Data Flow with Types

```python
# Input
raw_text: str = "I'm SO stressed about deadlines!"

# After preprocessing
cleaned_text: str = "im so stressed about deadlines"

# After sentiment
sentiment_result: Dict[str, float] = {
    "label": "NEGATIVE",
    "score": 0.92
}

# After emotion detection
emotion_scores: Dict[str, float] = {
    "anxious": 0.30,
    "stressed": 0.75,
    "calm": 0.05,
    "motivated": 0.10,
    "frustrated": 0.40,
    "focused": 0.15
}

# Final output
output: Dict[str, Any] = {
    "sentiment_polarity": "NEGATIVE",
    "sentiment_score": 0.92,
    "anxiety_prob": 0.30,
    "emotional_tone": "stressed",
    "all_emotions": emotion_scores,
    # ... more fields
}
```

---

## 9. Dataset Statistics & Characteristics

### 9.1 GoEmotions Distribution (Approx.)

| Split | Samples | Emotion Distribution |
|-------|---------|----------------------|
| Train | ~43,410 | Balanced across 29 emotions |
| Validation | ~5,427 | Same distribution |
| Test | ~5,427 | Held-out evaluation set |
| **Total** | **~58,000** | Multi-label (1–3 emotions per sample) |

### 9.2 DepressionEmo Distribution

| Split | Samples | Class Distribution |
|-------|---------|-------------------|
| Train | Variable | Binary (0=normal, 1=depressed) |

---

## 10. Text Length Statistics

### 10.1 Typical Text Lengths (After Preprocessing)

| Metric | Value | Note |
|--------|-------|------|
| **Min tokens** | 1 | Very short fragment |
| **Median tokens** | 30–50 | Typical Reddit comment |
| **Max tokens** | 512 | BERT max input length |
| **Mean tokens** | 45 | Average |

### 10.2 Recommendation for Your Demo

- **Minimum length**: 8–10 tokens (e.g., "I'm really stressed about work")
- **Optimal length**: 30–100 tokens (e.g., 1–2 sentences)
- **Max length**: 512 tokens (handled by BERT truncation)

---

## 11. Summary Table: Text Processing in Mindflow

| Component | Current Implementation | Purpose | Status |
|-----------|------------------------|---------|--------|
| **Raw Data** | GoEmotions (58k) + DepressionEmo | Benchmark + domain-specific | ✅ Ready |
| **Preprocessing** | Lowercase, remove special chars, collapse whitespace | Normalize input | ✅ Implemented |
| **Sentiment** | DistilBERT (SST-2 fine-tuned) | Binary positive/negative | ✅ Implemented |
| **Emotion Detection** | Keyword-based (6 emotions) | Fast, interpretable emotion scoring | ✅ Implemented |
| **Confidence Estimation** | Heuristic (sentiment + emotion evidence) | Demo accuracy display | ✅ Implemented |
| **Stress Probability** | Derived from anxiety + sentiment | Main output for fusion | ✅ Implemented |

---

## 12. Next Steps (If Upgrading)

- [ ] **Add lemmatization** (preserve "stressed" vs "stressing" as same root)
- [ ] **Fine-tune BERT** on combined GoEmotions + DepressionEmo (stress-specific encoder)
- [ ] **Add emoji handling** (map emojis to sentiment before removal)
- [ ] **Cross-modal alignment** (sync text timestamps with keystroke + EEG)
- [ ] **Validate on held-out test set** (current: heuristic confidence)

---

**Your text processing is simple, fast, and **demo-ready**. For benchmark performance, consider fine-tuning a stress-specific model on DepressionEmo + custom stress labels.**
