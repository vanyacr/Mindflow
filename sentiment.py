"""Text sentiment module for stress scoring.

This module is privacy-aware by design:
- It returns derived features and scores.
- Raw input text should not be stored in logs/databases.
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional

from transformers import pipeline


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

EMOTION_KEYWORDS = {
    "frustration": {"frustrated", "annoyed", "irritated", "stuck", "angry"},
    "anxiety": {"anxious", "nervous", "worried", "panic", "overwhelmed"},
    "confusion": {"confused", "unclear", "lost", "unsure", "doubt"},
    "confidence": {"confident", "sure", "capable", "ready"},
    "calm": {"calm", "relaxed", "steady", "peaceful"},
    "happy": {"happy", "excited", "joy", "great", "good"},
}


@lru_cache(maxsize=1)
def _get_classifier():
    allow_download = os.getenv("MINDFLOW_ALLOW_MODEL_DOWNLOAD", "0") == "1"
    try:
        return pipeline("sentiment-analysis", model=MODEL_NAME, local_files_only=not allow_download)
    except Exception:
        return None


def _find_emotion_keywords(text: str) -> List[str]:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    found: List[str] = []
    for label, words in EMOTION_KEYWORDS.items():
        if tokens.intersection(words):
            found.append(label)
    return found


def extract_sentiment(text: str) -> Dict[str, object]:
    """Extract normalized sentiment in [0, 1] and emotion keywords.

    Returns:
    - sentiment: 0 negative, 0.5 neutral-ish, 1 positive
    - emotion_keywords: matched labels
    - confidence: classifier confidence
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return {"sentiment": 0.5, "emotion_keywords": [], "confidence": 0.0}

    classifier = _get_classifier()
    keywords = _find_emotion_keywords(cleaned)

    if classifier is None:
        neg_words = {"stressed", "anxious", "worried", "panic", "hopeless", "frustrated", "tired"}
        pos_words = {"calm", "ready", "confident", "good", "great", "happy", "focused"}
        tokens = re.findall(r"[a-z']+", cleaned.lower())
        neg = sum(1 for t in tokens if t in neg_words)
        pos = sum(1 for t in tokens if t in pos_words)
        sentiment = (pos + 1.0) / (pos + neg + 2.0)
        confidence = (max(pos, neg) + 1.0) / (pos + neg + 2.0)
        return {
            "sentiment": float(round(sentiment, 4)),
            "emotion_keywords": keywords,
            "confidence": float(round(confidence, 4)),
        }

    result = classifier(cleaned, truncation=True)[0]
    label = str(result["label"]).upper()
    score = float(result["score"])

    sentiment = score if label == "POSITIVE" else (1.0 - score)

    return {
        "sentiment": float(round(sentiment, 4)),
        "emotion_keywords": keywords,
        "confidence": float(round(score, 4)),
    }


def sentiment_to_stress_score(sentiment_dict: Dict[str, object]) -> Dict[str, object]:
    """Map sentiment + emotion keywords into stress score in [0,1]."""
    sentiment = float(sentiment_dict.get("sentiment", 0.5))
    keywords = [str(k) for k in sentiment_dict.get("emotion_keywords", [])]
    confidence = float(sentiment_dict.get("confidence", 0.0))

    # Piecewise map as requested.
    if sentiment <= 0.33:
        # 0 -> 1.0, 0.33 -> 0.7
        base = 1.0 - ((sentiment / 0.33) * 0.3)
    elif sentiment <= 0.67:
        # 0.33 -> 0.7, 0.67 -> 0.3
        base = 0.7 - (((sentiment - 0.33) / 0.34) * 0.4)
    else:
        # 0.67 -> 0.3, 1.0 -> 0.0
        base = 0.3 - (((sentiment - 0.67) / 0.33) * 0.3)

    boosts = {
        "frustration": 0.3,
        "anxiety": 0.4,
        "confusion": 0.25,
        "confidence": -0.25,
        "calm": -0.25,
        "happy": -0.2,
    }
    adjust = sum(boosts.get(k, 0.0) for k in keywords)

    stress = max(0.0, min(1.0, base + adjust))

    return {
        "modality": "text",
        "text_score": float(round(stress, 4)),
        "confidence": float(round(confidence, 4)),
        "sentiment": float(round(sentiment, 4)),
        "emotion_keywords": keywords,
        "privacy_note": "Only derived scores should be stored. Raw text/transcript should be discarded.",
    }


class RealTimeSentimentScorer:
    """Compute text stress score only when new text arrives and interval elapsed."""

    def __init__(self, min_interval_seconds: int = 300) -> None:
        self.min_interval_seconds = max(60, int(min_interval_seconds))
        self._last_emit_ts = 0.0
        self._last_text = ""

    def process_if_due(self, text: str) -> Optional[Dict[str, object]]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None

        now = time.time()
        is_new_text = cleaned != self._last_text
        interval_elapsed = (now - self._last_emit_ts) >= self.min_interval_seconds

        if not is_new_text:
            return None

        if not interval_elapsed and self._last_emit_ts > 0.0:
            return None

        sentiment = extract_sentiment(cleaned)
        out = sentiment_to_stress_score(sentiment)
        out["interval_seconds"] = self.min_interval_seconds

        self._last_text = cleaned
        self._last_emit_ts = now
        return out
