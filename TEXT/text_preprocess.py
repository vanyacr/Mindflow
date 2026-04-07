"""Text preprocessing utilities for the Mindflow NLP pipeline."""

from __future__ import annotations

import re


_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(raw_text: str) -> str:
    """Normalize user text for downstream NLP models.

    Steps:
    1. Lowercase
    2. Remove special characters
    3. Collapse extra spaces
    """
    if raw_text is None:
        return ""

    text = str(raw_text).lower()
    text = _SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
