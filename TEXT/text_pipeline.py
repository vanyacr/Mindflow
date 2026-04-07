"""Main text pipeline for the Mindflow multimodal project."""

from __future__ import annotations

from typing import Any, Dict

from .text_model import analyze_sentiment, detect_emotions
from .text_preprocess import clean_text


def _round_prob(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def run_text_pipeline(raw_text: str) -> Dict[str, Any]:
    """Run preprocessing + sentiment + emotion inference for raw text input."""
    cleaned_text = clean_text(raw_text)

    sentiment_result = analyze_sentiment(cleaned_text)
    emotion_scores = detect_emotions(cleaned_text)

    emotional_tone = (
        max(emotion_scores, key=emotion_scores.get)
        if emotion_scores and max(emotion_scores.values(), default=0.0) > 0.0
        else "calm"
    )
    anxiety_prob = _round_prob(emotion_scores.get("anxious", 0.0))
    motivation_level = _round_prob(emotion_scores.get("motivated", 0.0))

    return {
        "sentiment_polarity": str(sentiment_result.get("label", "NEGATIVE")).upper(),
        "sentiment_score": _round_prob(sentiment_result.get("score", 0.0)),
        "anxiety_prob": anxiety_prob,
        "emotional_tone": emotional_tone,
        "motivation_level": motivation_level,
        "all_emotions": {label: _round_prob(score) for label, score in emotion_scores.items()},
    }
