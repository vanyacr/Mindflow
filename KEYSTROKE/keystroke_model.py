"""Keystroke-based stress scoring using behavioral feature heuristics."""

from __future__ import annotations

from typing import Dict

from .keystroke_features import KeystrokeFeatureExtractor
from .keystroke_listener import KeystrokeBuffer


def _normalize_to_stress_score(raw_value: float, min_val: float, max_val: float) -> float:
    """Map a raw feature value to [0, 1] stress likelihood.
    
    Values are clamped and scaled inversely if they correlate negatively with stress.
    """
    clamped = max(min_val, min(max_val, raw_value))
    normalized = (clamped - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    return round(max(0.0, min(1.0, normalized)), 4)


class KeystrokeStressModel:
    """Score stress likelihood from keystroke dynamics using heuristic rules."""

    WPM_STRESS_THRESHOLD = 25.0
    PRESSURE_STRESS_THRESHOLD = 150.0
    PAUSE_STRESS_THRESHOLD = 0.3

    @staticmethod
    def score_wpm_deviation(wpm_deviation: float) -> Dict[str, float]:
        """Score stress from WPM deviation (faster or slower is both concerning)."""
        abs_deviation = abs(wpm_deviation)
        wpm_stress = _normalize_to_stress_score(abs_deviation, 0.0, 50.0)

        return {
            "wpm_stress_score": wpm_stress,
            "wpm_stress_reason": (
                "elevated_typing_speed" if wpm_deviation > 0
                else "reduced_typing_speed" if wpm_deviation < -5 else "normal"
            ),
        }

    @staticmethod
    def score_key_pressure(avg_hold_ms: float, max_hold_ms: float) -> Dict[str, float]:
        """Score stress from key hold duration (hesitation indicator)."""
        avg_stress = _normalize_to_stress_score(avg_hold_ms, 50.0, 200.0)
        max_stress = _normalize_to_stress_score(max_hold_ms, 100.0, 400.0)
        pressure_stress = (avg_stress * 0.7) + (max_stress * 0.3)

        return {
            "pressure_stress_score": round(pressure_stress, 4),
            "pressure_stress_reason": (
                "prolonged_key_holds" if avg_hold_ms > 150 else "normal_pressure"
            ),
        }

    @staticmethod
    def score_pause_patterns(avg_pause_ms: float, pause_frequency: float) -> Dict[str, float]:
        """Score stress from inter-keystroke intervals (hesitation/rumination)."""
        pause_avg_stress = _normalize_to_stress_score(avg_pause_ms, 100.0, 800.0)
        pause_freq_stress = _normalize_to_stress_score(pause_frequency, 0.1, 0.5)
        pause_stress = (pause_avg_stress * 0.4) + (pause_freq_stress * 0.6)

        return {
            "pause_stress_score": round(pause_stress, 4),
            "pause_stress_reason": (
                "frequent_hesitation" if pause_frequency > 0.3
                else "prolonged_pauses" if avg_pause_ms > 500 else "normal_pauses"
            ),
        }

    @staticmethod
    def score_timing_variability(latency_std_ms: float, burst_ratio: float) -> Dict[str, float]:
        """Score erratic rhythm and rushed burst behavior."""
        latency_stress = _normalize_to_stress_score(latency_std_ms, 0.0, 350.0)
        burst_stress = _normalize_to_stress_score(burst_ratio, 0.2, 0.9)
        combined = (latency_stress * 0.6) + (burst_stress * 0.4)

        return {
            "latency_stress_score": round(combined, 4),
            "latency_stress_reason": (
                "erratic_timing" if latency_std_ms > 200 else "normal_timing"
            ),
            "burst_stress_score": round(burst_stress, 4),
            "burst_stress_reason": (
                "rushed_bursting" if burst_ratio > 0.55 else "steady_bursting"
            ),
        }

    @staticmethod
    def score_corrections(backspace_rate: float, key_variation: float) -> Dict[str, float]:
        """Score correction activity and text-switching patterns."""
        correction_stress = _normalize_to_stress_score(backspace_rate, 0.0, 12.0)
        variation_stress = _normalize_to_stress_score(abs(0.5 - key_variation), 0.0, 0.5)
        combined = (correction_stress * 0.75) + (variation_stress * 0.25)

        return {
            "correction_stress_score": round(combined, 4),
            "correction_stress_reason": (
                "high_correction_rate" if backspace_rate > 4 else "normal_corrections"
            ),
        }

    @staticmethod
    def compute_keystroke_stress_probability(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Compute overall keystroke-based stress probability."""
        features = KeystrokeFeatureExtractor.extract_all_features(buffer, window_seconds)

        wpm_scores = KeystrokeStressModel.score_wpm_deviation(features["wpm_deviation"])
        pressure_scores = KeystrokeStressModel.score_key_pressure(
            features["avg_key_hold_ms"], features["max_key_hold_ms"]
        )
        pause_scores = KeystrokeStressModel.score_pause_patterns(
            features["avg_pause_ms"], features["pause_frequency"]
        )
        timing_scores = KeystrokeStressModel.score_timing_variability(
            features.get("latency_std_ms", 0.0), features.get("burst_ratio", 0.0)
        )
        correction_scores = KeystrokeStressModel.score_corrections(
            features.get("backspace_rate", 0.0), features.get("key_variation", 0.0)
        )

        overall_stress = (
            (wpm_scores["wpm_stress_score"] * 0.2) +
            (pressure_scores["pressure_stress_score"] * 0.2) +
            (pause_scores["pause_stress_score"] * 0.2) +
            (timing_scores["latency_stress_score"] * 0.15) +
            (correction_scores["correction_stress_score"] * 0.15) +
            (timing_scores["burst_stress_score"] * 0.1)
        )

        event_count = features.get("keystroke_count", 0)
        confidence = min(1.0, max(0.2, event_count / 100.0))

        result = {
            "keystroke_stress_probability": round(overall_stress, 4),
            "keystroke_stress_confidence": round(confidence, 4),
            "wpm_stress_score": wpm_scores["wpm_stress_score"],
            "wpm_stress_reason": wpm_scores["wpm_stress_reason"],
            "pressure_stress_score": pressure_scores["pressure_stress_score"],
            "pressure_stress_reason": pressure_scores["pressure_stress_reason"],
            "pause_stress_score": pause_scores["pause_stress_score"],
            "pause_stress_reason": pause_scores["pause_stress_reason"],
            "latency_stress_score": timing_scores["latency_stress_score"],
            "latency_stress_reason": timing_scores["latency_stress_reason"],
            "burst_stress_score": timing_scores["burst_stress_score"],
            "burst_stress_reason": timing_scores["burst_stress_reason"],
            "correction_stress_score": correction_scores["correction_stress_score"],
            "correction_stress_reason": correction_scores["correction_stress_reason"],
            "event_count": int(event_count),
        }

        return result
