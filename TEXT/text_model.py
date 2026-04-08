"""Model loading and inference helpers for Mindflow text NLP."""

from __future__ import annotations

import os
from functools import lru_cache
import re
from typing import Dict, List

from transformers import pipeline


SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_LABELS: List[str] = [
    "anxious",
    "stressed",
    "calm",
    "motivated",
    "frustrated",
    "focused",
]
EMOTION_KEYWORDS = {
    "anxious": ("anxious", "anxiety", "worried", "nervous", "panic", "panicked", "uneasy", "overwhelmed"),
    "stressed": ("stressed", "stress", "pressure", "deadline", "deadlines", "exam", "exams", "workload", "burnout"),
    "calm": ("calm", "relaxed", "peaceful", "steady", "balanced", "clear", "centered"),
    "motivated": ("motivated", "determined", "driven", "focused", "consistent", "productive", "ready", "inspired"),
    "frustrated": ("frustrated", "annoyed", "angry", "upset", "irritated", "stuck", "drained", "fed up"),
    "focused": ("focused", "concentrate", "concentrated", "attentive", "organized", "on track", "alert", "productive"),
}


@lru_cache(maxsize=1)
def load_sentiment_model():
    """Load and cache the DistilBERT sentiment model if available locally.

    By default we avoid network downloads to keep CLI runs responsive.
    Set MINDFLOW_ALLOW_MODEL_DOWNLOAD=1 to allow online model fetch.
    """
    allow_download = os.getenv("MINDFLOW_ALLOW_MODEL_DOWNLOAD", "0") == "1"

    try:
        return pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            local_files_only=not allow_download,
        )
    except Exception:
        return None


def _fallback_sentiment(text: str) -> Dict[str, float]:
    """Keyword-based fallback when transformer weights are unavailable."""
    normalized = str(text or "").lower()
    positive_words = {
        "good",
        "great",
        "calm",
        "focused",
        "motivated",
        "ready",
        "improve",
        "confident",
        "hopeful",
        "productive",
    }
    negative_words = {
        "bad",
        "anxious",
        "stressed",
        "depressed",
        "hopeless",
        "overwhelmed",
        "nervous",
        "tired",
        "frustrated",
        "panic",
    }

    tokens = re.findall(r"[a-z']+", normalized)
    pos = sum(1 for token in tokens if token in positive_words)
    neg = sum(1 for token in tokens if token in negative_words)

    if pos >= neg:
        score = (pos + 1.0) / (pos + neg + 2.0)
        return {"label": "POSITIVE", "score": float(round(score, 4))}

    score = (neg + 1.0) / (pos + neg + 2.0)
    return {"label": "NEGATIVE", "score": float(round(score, 4))}


def analyze_sentiment(text: str) -> Dict[str, float]:
    """Return sentiment label and confidence score for input text."""
    sentiment_pipe = load_sentiment_model()
    if sentiment_pipe is None:
        return _fallback_sentiment(text)

    result = sentiment_pipe(text or "", truncation=True)[0]
    return {
        "label": str(result["label"]).upper(),
        "score": float(result["score"]),
    }


def detect_emotions(text: str, candidate_labels: List[str] | None = None) -> Dict[str, float]:
    """Return lightweight emotion scores for requested emotion labels.

    This keeps the text pipeline on a single transformer model by using
    DistilBERT for sentiment and a local keyword scorer for emotion tone.
    """
    labels = candidate_labels or EMOTION_LABELS
    normalized_text = str(text or "").lower()
    tokens = re.findall(r"[a-z']+", normalized_text)
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    def score_label(label: str) -> float:
        keywords = EMOTION_KEYWORDS.get(label, ())
        score = 0.0
        for keyword in keywords:
            if " " in keyword:
                score += 1.5 if keyword in normalized_text else 0.0
            else:
                score += float(token_counts.get(keyword, 0))

        # Keep scores in a simple 0.0-1.0 range without extra dependencies.
        return round(min(score / 3.0, 1.0), 4)

    return {label: score_label(label) for label in labels}
