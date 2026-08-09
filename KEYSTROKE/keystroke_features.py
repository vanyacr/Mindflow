"""Feature extraction from keystroke dynamics."""

from __future__ import annotations

import math
from typing import Dict, List

from .keystroke_listener import KeystrokeBuffer, KeystrokeEvent


BACKSPACE_KEYS = {"\b", "Key.backspace", "backspace", "Key.delete", "delete"}


class KeystrokeFeatureExtractor:
    """Extract behavioral features from keystroke patterns."""

    @staticmethod
    def compute_wpm_data(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Compute words-per-minute and deviation from user baseline."""
        events = buffer.get_events_in_window(window_seconds)

        if not events:
            return {
                "wpm": 0.0,
                "wpm_baseline": buffer.user_baseline_wpm,
                "wpm_deviation": 0.0,
                "keystroke_count": 0,
            }

        key_presses = sorted((e for e in events if e.event_type == "press"), key=lambda e: e.timestamp)
        keystroke_count = len(key_presses)
        word_count = keystroke_count / 5.0

        if len(key_presses) >= 2:
            active_duration_seconds = max(1.0, key_presses[-1].timestamp - key_presses[0].timestamp)
        else:
            active_duration_seconds = float(max(1, window_seconds))
        active_duration_minutes = active_duration_seconds / 60.0
        wpm = word_count / max(active_duration_minutes, 0.016)

        wpm_deviation = ((wpm - buffer.user_baseline_wpm) / max(buffer.user_baseline_wpm, 1.0)) * 100.0

        return {
            "wpm": round(wpm, 2),
            "wpm_baseline": buffer.user_baseline_wpm,
            "wpm_deviation": round(wpm_deviation, 2),
            "keystroke_count": keystroke_count,
        }

    @staticmethod
    def compute_key_pressure_duration(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Compute key hold duration statistics."""
        events = buffer.get_events_in_window(window_seconds)
        release_events = [e for e in events if e.event_type == "release" and e.duration > 0]

        if not release_events:
            return {
                "avg_key_hold_ms": 0.0,
                "max_key_hold_ms": 0.0,
                "std_dev_key_hold_ms": 0.0,
                "sample_count": 0,
            }

        durations_ms = [e.duration * 1000.0 for e in release_events]
        avg_hold = sum(durations_ms) / len(durations_ms)
        max_hold = max(durations_ms)
        variance = sum((d - avg_hold) ** 2 for d in durations_ms) / len(durations_ms)
        std_dev = variance ** 0.5

        return {
            "avg_key_hold_ms": round(avg_hold, 2),
            "max_key_hold_ms": round(max_hold, 2),
            "std_dev_key_hold_ms": round(std_dev, 2),
            "sample_count": len(durations_ms),
        }

    @staticmethod
    def compute_pause_patterns(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Detect hesitation and pause patterns between keystrokes."""
        events = buffer.get_events_in_window(window_seconds)

        if len(events) < 2:
            return {
                "avg_pause_ms": 0.0,
                "max_pause_ms": 0.0,
                "min_pause_ms": 0.0,
                "pause_count": 0,
                "pause_frequency": 0.0,
            }

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        pauses_ms: List[float] = []

        for i in range(1, len(sorted_events)):
            time_diff = (sorted_events[i].timestamp - sorted_events[i - 1].timestamp) * 1000.0
            if time_diff > 100.0:
                pauses_ms.append(time_diff)

        if not pauses_ms:
            return {
                "avg_pause_ms": 0.0,
                "max_pause_ms": 0.0,
                "min_pause_ms": 0.0,
                "pause_count": 0,
                "pause_frequency": 0.0,
            }

        pause_frequency = len(pauses_ms) / max(len(sorted_events), 1)

        return {
            "avg_pause_ms": round(sum(pauses_ms) / len(pauses_ms), 2),
            "max_pause_ms": round(max(pauses_ms), 2),
            "min_pause_ms": round(min(pauses_ms), 2),
            "pause_count": len(pauses_ms),
            "pause_frequency": round(pause_frequency, 4),
        }

    @staticmethod
    def compute_timing_variability(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Capture irregular timing and burst behavior.

        These features improve discrimination between normal, rushed, and stressed typing.
        """
        events = buffer.get_events_in_window(window_seconds)
        press_events = sorted((e for e in events if e.event_type == "press"), key=lambda e: e.timestamp)

        if len(press_events) < 2:
            return {
                "latency_std_ms": 0.0,
                "burst_ratio": 0.0,
                "key_variation": 0.0,
            }

        inter_key_ms = []
        for i in range(1, len(press_events)):
            gap_ms = (press_events[i].timestamp - press_events[i - 1].timestamp) * 1000.0
            if gap_ms >= 0.0:
                inter_key_ms.append(gap_ms)

        if not inter_key_ms:
            return {
                "latency_std_ms": 0.0,
                "burst_ratio": 0.0,
                "key_variation": 0.0,
            }

        mean_gap = sum(inter_key_ms) / len(inter_key_ms)
        latency_std_ms = math.sqrt(sum((gap - mean_gap) ** 2 for gap in inter_key_ms) / len(inter_key_ms))

        burst_count = sum(1 for gap in inter_key_ms if gap < 200.0)
        burst_ratio = burst_count / len(inter_key_ms)

        unique_keys = len({e.key for e in press_events})
        key_variation = unique_keys / len(press_events)

        return {
            "latency_std_ms": round(latency_std_ms, 2),
            "burst_ratio": round(burst_ratio, 4),
            "key_variation": round(key_variation, 4),
        }

    @staticmethod
    def compute_correction_and_error_features(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Measure correction activity, which often rises under frustration or stress."""
        events = buffer.get_events_in_window(window_seconds)
        if not events:
            return {
                "backspace_count": 0,
                "backspace_rate": 0.0,
            }

        backspace_count = sum(1 for e in events if e.event_type == "press" and e.key in BACKSPACE_KEYS)
        window_minutes = max(window_seconds / 60.0, 1.0 / 60.0)
        backspace_rate = backspace_count / window_minutes

        return {
            "backspace_count": int(backspace_count),
            "backspace_rate": round(backspace_rate, 4),
        }

    @staticmethod
    def extract_all_features(buffer: KeystrokeBuffer, window_seconds: int = 60) -> Dict[str, float]:
        """Extract all keystroke features at once."""
        wpm_data = KeystrokeFeatureExtractor.compute_wpm_data(buffer, window_seconds)
        pressure_data = KeystrokeFeatureExtractor.compute_key_pressure_duration(buffer, window_seconds)
        pause_data = KeystrokeFeatureExtractor.compute_pause_patterns(buffer, window_seconds)
        timing_data = KeystrokeFeatureExtractor.compute_timing_variability(buffer, window_seconds)
        correction_data = KeystrokeFeatureExtractor.compute_correction_and_error_features(buffer, window_seconds)

        all_features = {}
        all_features.update(wpm_data)
        all_features.update(pressure_data)
        all_features.update(pause_data)
        all_features.update(timing_data)
        all_features.update(correction_data)

        return all_features
