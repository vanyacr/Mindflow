# StressEstimator — Fuses Text + Keystroke into 0-100 Score

## Text Modality Signals

| Signal | Stress Direction | Source | Evidence |
|--------|------------------|--------|----------|
| **Sentiment Polarity** | NEGATIVE = more stress | DistilBERT | Negative language correlates with stress/anxiety |
| **Anxiety Keywords** | ↑ = more stress | Keyword matching | Explicit stress indicators: "anxious", "worried", "overwhelmed", "stressed" |
| **Emotional Tone** | Anxious/Frustrated = stress | 6-label emotion detector | Negative emotions (frustrated, anxious) indicate distress |
| **Motivation Level** | ↓ = more stress | Keyword analysis | Low motivation ("can't do it", "tired") signals stress/depression |
| **Confidence Score** | ↓ = uncertainty/stress | Model confidence | Lower confidence in sentiment indicates ambiguity/mixed emotions |

---

## Keystroke Modality Signals

| Signal | Stress Direction | Baseline | Stress Indicator |
|--------|------------------|----------|------------------|
| **Typing Velocity (WPM)** | ↓ = more stress | 162.37 WPM | Slow typing (anxiety → hesitation) |
| **Key Dwell Time (ms)** | ↑ = more stress | 98.65 ms | Holding keys longer (tension) |
| **Dwell Variability (std)** | ↑ = more stress | 29.30 ms | Inconsistent pressure (loss of control) |
| **Key-to-Key Latency (ms)** | ↑ = more stress | 307.59 ms | Longer pauses between keys (cognitive load) |
| **Pause Frequency** | ↑ = more stress | 14.39 pauses/min | More thinking pauses (uncertainty) |
| **Error Count** | ↑ = more stress | 5.80 errors/min | More typos (distraction/rushing) |

---

## Fusion Logic (Multimodal Score)

```
Raw Scores:
  - Text_Score (0-100) ← sentiment + emotions + keywords
  - Keystroke_Score (0-100) ← z-score deviations from baseline
  
Weighted Fusion:
  Final_Stress = (0.60 × Text_Score) + (0.40 × Keystroke_Score)
  
Output Range: 0-100
  - 0-30:    Low stress (normal)
  - 30-60:   Moderate stress (caution)
  - 60-100:  High stress (alert)
```

---

## Example Scenarios

### Scenario 1: Text Says "OK" but Keystroke Reveals Stress
```
Text Input: "I'm fine, everything is good"
- Sentiment: POSITIVE (0.95)
- Emotions: calm, motivated
- Text Score: 15 (low stress)

Keystroke Pattern:
- Velocity: 80 WPM (↓ from 162) → +25 stress
- Pause frequency: 25 (↑ from 14) → +20 stress  
- Errors: 12 (↑ from 5.8) → +15 stress
- Keystroke Score: 60 (high stress)

FUSION: (0.60 × 15) + (0.40 × 60) = 33 (MODERATE STRESS)
✓ System catches hidden stress despite positive text
```

### Scenario 2: True Stress in Both Modalities
```
Text Input: "I'm so anxious, can't focus, everything is overwhelming"
- Sentiment: NEGATIVE (0.02)
- Emotions: anxious (0.85), stressed (0.70)
- Text Score: 85 (high stress)

Keystroke Pattern:
- Velocity: 70 WPM (↓ from 162)
- Pause frequency: 30 (↑ from 14)
- Error count: 15 (↑ from 5.8)
- Keystroke Score: 75 (high stress)

FUSION: (0.60 × 85) + (0.40 × 75) = 81 (HIGH STRESS)
✓ Both modalities agree = high confidence alert
```

### Scenario 3: Normal Typing, Positive Text
```
Text: "I accomplished my goals today and feel great!"
- Sentiment: POSITIVE (0.98)
- Emotions: motivated (0.9), calm (0.8)
- Text Score: 10 (very low stress)

Keystroke:
- Velocity: 165 WPM (≈ baseline 162) → 0 deviation
- Pause freq: 14 (= baseline) → 0 deviation
- Errors: 5.5 (≈ baseline 5.8) → 0 deviation
- Keystroke Score: 5 (very low stress)

FUSION: (0.60 × 10) + (0.40 × 5) = 8 (NORMAL)
✓ System confirms no stress
```

---

## Why This Works

| Advantage | Explanation |
|-----------|-------------|
| **Catches Hidden Stress** | Keystroke reveals stress even when user denies/masks it |
| **Detects Text Anomalies** | Text sentiment alone could be misled by sarcasm; keystroke validates |
| **Personalized Baseline** | Your unique typing signature (velocity, dwell, pauses) makes detection accurate |
| **Continuous Monitoring** | Real-time keystroke collection (5-min windows) = early warning system |
| **Robust to Single Failure** | If text model offline, keystroke alone still detects stress; vice versa |
| **Research-Backed** | Both text emotion and keystroke dynamics proven stress indicators in literature |
