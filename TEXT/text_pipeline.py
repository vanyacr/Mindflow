"""Main text pipeline for the Mindflow multimodal project."""

from __future__ import annotations

from typing import Any, Dict

from .text_model import analyze_sentiment, detect_emotions
from .text_preprocess import clean_text


def _round_prob(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _format_percent_range(center: float, spread: float = 3.0) -> str:
    lower = int(round(max(35.0, min(99.0, center - spread))))
    upper = int(round(max(35.0, min(99.0, center + spread))))
    if lower > upper:
        lower, upper = upper, lower
    return f"{lower}-{upper}%"


def _estimate_display_accuracy(sentiment_score: float, emotion_scores: Dict[str, float]) -> tuple[str, str]:
    """Return per-input demo estimates from confidence + emotion evidence.

    These are heuristic display estimates, not benchmark metrics.
    """
    confidence = abs((2.0 * sentiment_score) - 1.0)  # 0 near neutral, 1 near confident edges
    ranked = sorted((float(v) for v in emotion_scores.values()), reverse=True)
    top_emotion = ranked[0] if ranked else 0.0
    top_gap = (ranked[0] - ranked[1]) if len(ranked) > 1 else top_emotion

    sentiment_center = 65.0 + (30.0 * confidence)
    overall_center = 60.0 + (18.0 * confidence) + (12.0 * top_emotion) + (5.0 * max(0.0, top_gap))

    return _format_percent_range(sentiment_center), _format_percent_range(overall_center)


def run_text_pipeline(raw_text: str) -> Dict[str, Any]:
    """Run preprocessing + sentiment + emotion inference for raw text input."""
    cleaned_text = clean_text(raw_text)

    sentiment_result = analyze_sentiment(cleaned_text)
    emotion_scores = detect_emotions(cleaned_text, sentiment_result=sentiment_result)

    emotional_tone = (
        max(emotion_scores, key=emotion_scores.get)
        if emotion_scores and max(emotion_scores.values(), default=0.0) > 0.0
        else "calm"
    )
    sentiment_score = _round_prob(sentiment_result.get("score", 0.0))
    estimated_sentiment_accuracy, estimated_overall_text_accuracy = _estimate_display_accuracy(
        sentiment_score,
        emotion_scores,
    )
    anxiety_prob = _round_prob(emotion_scores.get("anxious", 0.0))
    motivation_level = _round_prob(emotion_scores.get("motivated", 0.0))

    return {
        "sentiment_polarity": str(sentiment_result.get("label", "NEGATIVE")).upper(),
        "sentiment_score": sentiment_score,
        "estimated_sentiment_accuracy": estimated_sentiment_accuracy,
        "estimated_overall_text_accuracy": estimated_overall_text_accuracy,
        "anxiety_prob": anxiety_prob,
        "emotional_tone": emotional_tone,
        "motivation_level": motivation_level,
        "all_emotions": {label: _round_prob(score) for label, score in emotion_scores.items()},
    }
