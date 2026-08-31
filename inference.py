"""Fusion-ready inference entrypoint for text + keystroke scores."""

from __future__ import annotations

from typing import Dict, Optional

from keystroke import KeystrokeTracker
from sentiment import RealTimeSentimentScorer, extract_sentiment, sentiment_to_stress_score


def weighted_fusion(
    score_visual: Optional[float] = None,
    score_audio: Optional[float] = None,
    score_keystroke: Optional[float] = None,
    score_text: Optional[float] = None,
    w_visual: float = 0.35,
    w_audio: float = 0.25,
    w_keystroke: float = 0.2,
    w_text: float = 0.2,
) -> Dict[str, object]:
    """Late-fusion weighted average with missing-modality handling."""
    items = [
        ("visual", score_visual, w_visual),
        ("audio", score_audio, w_audio),
        ("keystroke", score_keystroke, w_keystroke),
        ("text", score_text, w_text),
    ]

    active = [(name, float(score), float(weight)) for name, score, weight in items if score is not None]
    if not active:
        return {
            "final_score": 0.0,
            "used_modalities": [],
            "weights_used": {},
            "mask": {"visual": 0, "audio": 0, "keystroke": 0, "text": 0},
        }

    total_weight = sum(weight for _, _, weight in active)
    norm = [(name, score, weight / total_weight) for name, score, weight in active]

    final_score = sum(score * w for _, score, w in norm)
    return {
        "final_score": round(final_score, 4),
        "used_modalities": [name for name, _, _ in norm],
        "weights_used": {name: round(w, 4) for name, _, w in norm},
        "mask": {
            "visual": int(score_visual is not None),
            "audio": int(score_audio is not None),
            "keystroke": int(score_keystroke is not None),
            "text": int(score_text is not None),
        },
    }


def run_text_and_keystroke_inference(
    tracker: KeystrokeTracker,
    text: Optional[str] = None,
    model_path: Optional[str] = None,
    keystroke_window_seconds: int = 30,
    text_scorer: Optional[RealTimeSentimentScorer] = None,
) -> Dict[str, object]:
    """Return two continuous [0,1] scores to feed fusion layer."""
    keystroke_out = tracker.keystroke_stress_score(
        window_seconds=keystroke_window_seconds,
        model_path=model_path,
    )

    text_out = None
    if text is not None and str(text).strip():
        if text_scorer is not None:
            text_out = text_scorer.process_if_due(text)
        else:
            sentiment = extract_sentiment(text)
            text_out = sentiment_to_stress_score(sentiment)

    fusion = weighted_fusion(
        score_keystroke=keystroke_out["keystroke_score"],
        score_text=(text_out["text_score"] if text_out else None),
    )

    return {
        "keystroke": keystroke_out,
        "text": text_out,
        "fusion": fusion,
    }
