"""Main keystroke dynamics pipeline for stress detection."""

from __future__ import annotations

from typing import Any, Dict

from .keystroke_listener import KeystrokeBuffer, KeystrokeListener
from .keystroke_model import KeystrokeStressModel


def run_keystroke_pipeline(
    keystroke_buffer: KeystrokeBuffer | None = None,
    window_seconds: int = 60,
    user_baseline_wpm: float = 60.0,
) -> Dict[str, Any]:
    """Run keystroke analysis pipeline for stress detection.
    
    Args:
        keystroke_buffer: Pre-populated KeystrokeBuffer (if None, will check active listener)
        window_seconds: Time window for feature extraction
        user_baseline_wpm: User's typical typing speed for deviation calculation
    
    Returns:
        Dict with keystroke stress outputs ready for fusion layer
    """
    # Use provided buffer or create new empty one
    if keystroke_buffer is None:
        keystroke_buffer = KeystrokeBuffer()
    
    # Set user baseline if provided
    if user_baseline_wpm > 0:
        keystroke_buffer.user_baseline_wpm = user_baseline_wpm
    
    # Compute stress scores
    stress_result = KeystrokeStressModel.compute_keystroke_stress_probability(
        keystroke_buffer, window_seconds=window_seconds
    )
    
    # Format for fusion consumption
    return {
        "modality": "keystroke",
        "stress_probability": stress_result["keystroke_stress_probability"],
        "confidence": stress_result["keystroke_stress_confidence"],
        "component_scores": {
            "wpm": {
                "score": stress_result["wpm_stress_score"],
                "reason": stress_result["wpm_stress_reason"],
            },
            "pressure": {
                "score": stress_result["pressure_stress_score"],
                "reason": stress_result["pressure_stress_reason"],
            },
            "pauses": {
                "score": stress_result["pause_stress_score"],
                "reason": stress_result["pause_stress_reason"],
            },
        },
        "event_count": stress_result["event_count"],
        "window_seconds": window_seconds,
    }


def create_keystroke_listener() -> KeystrokeListener:
    """Factory function: create and return a new keystroke listener.
    
    In production, this can be extended to use pynput or pyxhook for
    real keyboard event capture.
    """
    return KeystrokeListener()
