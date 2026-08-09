"""Real-time keystroke modality for stress scoring.

This module provides:
- Optional live keyboard capture via `pynput`
- Thread-safe event queue with 100ms polling
- Feature extraction over sliding windows
- Baseline z-score normalization
- Model-based or heuristic stress scoring
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from pynput import keyboard
except Exception:  # pragma: no cover - optional dependency at runtime
    keyboard = None


EPS = 1e-6
BASELINE_FEATURE_KEYS = [
    "velocity",
    "dwell_mean_ms",
    "dwell_std_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "pause_freq",
    "error_count",
    "backspace_rate",
    "burst_ratio",
    "key_variation",
]
DEFAULT_FEATURE_ORDER = [
    "velocity_zscore",
    "dwell_mean_zscore",
    "dwell_std_zscore",
    "latency_mean_zscore",
    "latency_std_zscore",
    "pause_freq_zscore",
    "error_count_zscore",
    "backspace_rate_zscore",
    "burst_ratio_zscore",
    "key_variation_zscore",
]


def _safe_z(value: float, mean: float, std: float) -> float:
    return (float(value) - float(mean)) / max(EPS, float(std))


@dataclass
class KeystrokeRecord:
    """Represents one key press-release pair."""

    key_name: str
    press_ts: float
    release_ts: float

    @property
    def dwell_ms(self) -> float:
        return max(0.0, (self.release_ts - self.press_ts) * 1000.0)


class KeystrokeTracker:
    """Capture keyboard events and emit stress-ready feature vectors."""

    def __init__(
        self,
        poll_interval_s: float = 0.1,
        baseline_path: str | Path | None = None,
        auto_load_baseline: bool = True,
        auto_update_baseline: bool = True,
    ) -> None:
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        self._queue: queue.Queue[Tuple[str, str, float]] = queue.Queue()
        self._press_times: Dict[str, float] = {}
        self._records: List[KeystrokeRecord] = []
        self._records_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._listener = None
        self.auto_update_baseline = bool(auto_update_baseline)
        self.baseline_path = Path(baseline_path) if baseline_path else (Path("data") / "keystroke" / "baseline_auto.json")
        self._baseline_loaded_from_disk = False
        self._baseline_update_count = 0

        # Default baseline values. Replace via set_baseline_stats after enrollment.
        self._baseline_means = {
            "velocity": 60.0,
            "dwell_mean_ms": 95.0,
            "dwell_std_ms": 30.0,
            "latency_mean_ms": 140.0,
            "latency_std_ms": 80.0,
            "pause_freq": 2.0,
            "error_count": 3.0,
            "backspace_rate": 1.5,
            "burst_ratio": 0.45,
            "key_variation": 0.7,
        }
        self._baseline_stds = {
            "velocity": 12.0,
            "dwell_mean_ms": 20.0,
            "dwell_std_ms": 10.0,
            "latency_mean_ms": 50.0,
            "latency_std_ms": 35.0,
            "pause_freq": 1.0,
            "error_count": 2.0,
            "backspace_rate": 1.2,
            "burst_ratio": 0.18,
            "key_variation": 0.2,
        }

        if auto_load_baseline and self.baseline_path.exists():
            try:
                self.load_baseline(self.baseline_path)
                self._baseline_loaded_from_disk = True
            except Exception:
                self._baseline_loaded_from_disk = False

    def _key_to_name(self, key: object) -> str:
        if hasattr(key, "char") and getattr(key, "char") is not None:
            return str(getattr(key, "char"))
        return str(key)

    def _on_press(self, key: object) -> None:
        self._queue.put(("press", self._key_to_name(key), time.perf_counter()))

    def _on_release(self, key: object) -> None:
        self._queue.put(("release", self._key_to_name(key), time.perf_counter()))

    def _drain_queue_once(self) -> None:
        while True:
            try:
                event_type, key_name, ts = self._queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "press":
                self._press_times[key_name] = ts
                continue

            press_ts = self._press_times.pop(key_name, None)
            if press_ts is None:
                continue

            record = KeystrokeRecord(key_name=key_name, press_ts=press_ts, release_ts=ts)
            with self._records_lock:
                self._records.append(record)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_queue_once()
            time.sleep(self.poll_interval_s)

        self._drain_queue_once()

    def start_listener(self) -> None:
        """Start background listener + queue worker.

        Raises RuntimeError if pynput is unavailable.
        """
        if keyboard is None:
            raise RuntimeError("pynput is not installed. Run: pip install pynput")

        if self._worker and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop_listener(self) -> None:
        self._stop_event.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    def inject_event(self, key_name: str, press_ts: float, release_ts: float) -> None:
        """Testing utility to add synthetic events without OS hooks."""
        with self._records_lock:
            self._records.append(KeystrokeRecord(key_name=key_name, press_ts=press_ts, release_ts=release_ts))

    def _window_records(self, window_seconds: float) -> List[KeystrokeRecord]:
        now = time.perf_counter()
        cutoff = now - max(1.0, float(window_seconds))
        with self._records_lock:
            records = [r for r in self._records if r.release_ts >= cutoff]
        return sorted(records, key=lambda r: r.press_ts)

    def clear_old_records(self, keep_seconds: float = 600.0) -> None:
        now = time.perf_counter()
        cutoff = now - max(1.0, float(keep_seconds))
        with self._records_lock:
            self._records = [r for r in self._records if r.release_ts >= cutoff]

    def raw_keystroke_tuples(self, window_seconds: float = 30.0) -> List[Tuple[str, float, float]]:
        """Return raw tuples: (key_name, press_timestamp, release_timestamp)."""
        records = self._window_records(window_seconds)
        return [(r.key_name, float(r.press_ts), float(r.release_ts)) for r in records]

    def compute_features(self, window_seconds: float = 30.0, pause_threshold_ms: float = 500.0) -> Dict[str, float]:
        """Compute rich keystroke features from recent records."""
        records = self._window_records(window_seconds)
        if len(records) == 0:
            return {
                "velocity": 0.0,
                "dwell_mean_ms": 0.0,
                "dwell_std_ms": 0.0,
                "latency_mean_ms": 0.0,
                "latency_std_ms": 0.0,
                "pause_freq": 0.0,
                "error_count": 0.0,
                "error_count_raw": 0.0,
                "backspace_rate": 0.0,
                "burst_ratio": 0.0,
                "key_variation": 0.0,
                "pattern_variance": 0.0,
                "sample_count": 0.0,
            }

        first_press = records[0].press_ts
        last_release = records[-1].release_ts
        elapsed_s = max(EPS, last_release - first_press)

        key_count = float(len(records))
        velocity_wpm = (key_count / elapsed_s) * 60.0

        dwells = np.array([r.dwell_ms for r in records], dtype=float)
        dwell_mean = float(np.mean(dwells))
        dwell_std = float(np.std(dwells))

        latencies_ms: List[float] = []
        for i in range(1, len(records)):
            latency = (records[i].press_ts - records[i - 1].release_ts) * 1000.0
            if latency >= 0.0:
                latencies_ms.append(latency)

        latency_mean = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        latency_std = float(np.std(latencies_ms)) if latencies_ms else 0.0

        pause_count = sum(1 for v in latencies_ms if v > pause_threshold_ms)
        pause_freq = float(pause_count / (elapsed_s / 60.0))

        error_count_raw = float(
            sum(1 for r in records if r.key_name in {"\b", "Key.backspace", "backspace", "Key.delete", "delete"})
        )
        error_count = float(error_count_raw / (elapsed_s / 60.0))

        backspace_rate = float(error_count_raw / max(elapsed_s / 60.0, EPS))
        burst_count = sum(
            1
            for i in range(1, len(records))
            if (records[i].press_ts - records[i - 1].press_ts) * 1000.0 < 200.0
        )
        burst_ratio = float(burst_count / max(len(records) - 1, 1))
        key_variation = float(len({r.key_name for r in records}) / max(len(records), 1))

        pattern_variance = float(np.std(np.array([*dwells.tolist(), *latencies_ms], dtype=float))) if latencies_ms else dwell_std

        return {
            "velocity": round(velocity_wpm, 4),
            "dwell_mean_ms": round(dwell_mean, 4),
            "dwell_std_ms": round(dwell_std, 4),
            "latency_mean_ms": round(latency_mean, 4),
            "latency_std_ms": round(latency_std, 4),
            "pause_freq": round(pause_freq, 4),
            "error_count": round(error_count, 4),
            "error_count_raw": round(error_count_raw, 4),
            "backspace_rate": round(backspace_rate, 4),
            "burst_ratio": round(burst_ratio, 4),
            "key_variation": round(key_variation, 4),
            "pattern_variance": round(pattern_variance, 4),
            "sample_count": key_count,
        }

    def set_baseline_stats(self, means: Dict[str, float], stds: Dict[str, float]) -> None:
        self._baseline_means.update({k: float(v) for k, v in means.items()})
        self._baseline_stds.update({k: max(EPS, float(v)) for k, v in stds.items()})

    def get_baseline_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current baseline means/stds."""
        return {
            "means": {k: float(v) for k, v in self._baseline_means.items()},
            "stds": {k: float(v) for k, v in self._baseline_stds.items()},
        }

    def save_baseline(self, path: str | Path) -> None:
        """Persist baseline means/stds to JSON."""
        payload = self.get_baseline_stats()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_baseline(self, path: str | Path) -> None:
        """Load baseline means/stds from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        means = data.get("means", {})
        stds = data.get("stds", {})
        self.set_baseline_stats(means=means, stds=stds)

    def _blend_baseline_from_features(self, features: Dict[str, float], blend: float = 0.08) -> None:
        """Update baseline via EMA using stable windows.

        This keeps history while adapting slowly to the user's normal typing drift.
        """
        alpha = max(0.01, min(0.25, float(blend)))
        for key in BASELINE_FEATURE_KEYS:
            value = float(features.get(key, 0.0))
            old_mean = float(self._baseline_means[key])
            new_mean = ((1.0 - alpha) * old_mean) + (alpha * value)

            # Approximate updated spread from absolute deviation to keep std stable.
            deviation = abs(value - new_mean)
            old_std = float(self._baseline_stds[key])
            new_std = ((1.0 - alpha) * old_std) + (alpha * max(EPS, deviation))

            self._baseline_means[key] = float(new_mean)
            self._baseline_stds[key] = float(max(EPS, new_std))

    def _maybe_auto_update_baseline(self, features: Dict[str, float], stress_score: float) -> bool:
        """Persist baseline history from likely-neutral windows only."""
        if not self.auto_update_baseline:
            return False
        if float(features.get("sample_count", 0.0)) < 25.0:
            return False
        if float(stress_score) > 0.45:
            return False

        self._blend_baseline_from_features(features)
        self._baseline_update_count += 1

        try:
            self.save_baseline(self.baseline_path)
        except Exception:
            return False
        return True

    def zscore_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply baseline z-score normalization to the expanded feature set."""
        out = {
            "velocity_zscore": _safe_z(features["velocity"], self._baseline_means["velocity"], self._baseline_stds["velocity"]),
            "dwell_mean_zscore": _safe_z(features["dwell_mean_ms"], self._baseline_means["dwell_mean_ms"], self._baseline_stds["dwell_mean_ms"]),
            "dwell_std_zscore": _safe_z(features["dwell_std_ms"], self._baseline_means["dwell_std_ms"], self._baseline_stds["dwell_std_ms"]),
            "latency_mean_zscore": _safe_z(features["latency_mean_ms"], self._baseline_means["latency_mean_ms"], self._baseline_stds["latency_mean_ms"]),
            "latency_std_zscore": _safe_z(features["latency_std_ms"], self._baseline_means["latency_std_ms"], self._baseline_stds["latency_std_ms"]),
            "pause_freq_zscore": _safe_z(features["pause_freq"], self._baseline_means["pause_freq"], self._baseline_stds["pause_freq"]),
            "error_count_zscore": _safe_z(features["error_count"], self._baseline_means["error_count"], self._baseline_stds["error_count"]),
            "backspace_rate_zscore": _safe_z(features["backspace_rate"], self._baseline_means["backspace_rate"], self._baseline_stds["backspace_rate"]),
            "burst_ratio_zscore": _safe_z(features["burst_ratio"], self._baseline_means["burst_ratio"], self._baseline_stds["burst_ratio"]),
            "key_variation_zscore": _safe_z(features["key_variation"], self._baseline_means["key_variation"], self._baseline_stds["key_variation"]),
        }
        return {k: float(round(v, 4)) for k, v in out.items()}

    def feature_vector(self, zscores: Dict[str, float]) -> np.ndarray:
        """Return ordered feature vector used by the model."""
        return np.array([[float(zscores[k]) for k in DEFAULT_FEATURE_ORDER]], dtype=float)

    def compare_to_baseline(self, features: Dict[str, float]) -> Dict[str, object]:
        """Compare a current feature window to the user-specific baseline.

        This is the base personalized anomaly model: if the user deviates strongly
        from their own normal typing pattern, the system marks it as stressed.
        """
        zscores = self.zscore_features(features)
        abs_zscores = [abs(float(value)) for value in zscores.values()]
        magnitude = float(np.mean(abs_zscores)) if abs_zscores else 0.0

        # A moderate deviation from baseline corresponds to the user's normal range.
        # Strong deviations push the score toward stressed territory.
        overall_score = 1.0 / (1.0 + math.exp(-(magnitude - 1.0)))
        overall_score = float(round(max(0.0, min(1.0, overall_score)), 4))

        if overall_score >= 0.6:
            status = "stressed"
        elif overall_score >= 0.35:
            status = "watch"
        else:
            status = "normal"

        return {
            "overall_score": overall_score,
            "status": status,
            "deviation_magnitude": round(magnitude, 4),
            "zscore_features": zscores,
            "feature_deltas": {
                key: round(float(value), 4) for key, value in zscores.items()
            },
            "baseline_loaded_from_disk": self._baseline_loaded_from_disk,
            "baseline_path": str(self.baseline_path),
        }

    def keystroke_stress_score(
        self,
        window_seconds: float = 30.0,
        model_path: str | Path | None = None,
    ) -> Dict[str, object]:
        """Return fusion-ready keystroke stress score every 10-30 seconds."""
        features = self.compute_features(window_seconds=window_seconds)

        # Avoid overconfident outputs when there is too little typing evidence.
        if float(features.get("sample_count", 0.0)) < 8.0:
            return {
                "modality": "keystroke",
                "keystroke_score": 0.5,
                "confidence": 0.2,
                "window_seconds": int(window_seconds),
                "features": features,
                "zscore_features": {},
                "feature_vector": [],
                "model_used": "insufficient_data",
                "baseline_loaded_from_disk": self._baseline_loaded_from_disk,
                "baseline_path": str(self.baseline_path),
                "baseline_update_count": int(self._baseline_update_count),
            }

        zscores = self.zscore_features(features)
        vector = self.feature_vector(zscores)

        stress_score = None
        model_used = "heuristic"

        if model_path:
            try:
                model = joblib.load(str(model_path))
                if hasattr(model, "predict_proba"):
                    stress_score = float(model.predict_proba(vector)[0][1])
                    model_used = "trained_model"
            except Exception:
                stress_score = None

        if stress_score is None:
            # Heuristic fallback from magnitude of z-score deviation.
            magnitude = float(np.mean(np.abs(vector[0])))
            stress_score = 1.0 / (1.0 + math.exp(-(magnitude - 1.0)))

        baseline_updated = self._maybe_auto_update_baseline(features=features, stress_score=float(stress_score))

        confidence = min(1.0, max(0.2, features["sample_count"] / 120.0))

        return {
            "modality": "keystroke",
            "keystroke_score": float(round(stress_score, 4)),
            "confidence": float(round(confidence, 4)),
            "window_seconds": int(window_seconds),
            "features": features,
            "zscore_features": zscores,
            "feature_vector": vector[0].tolist(),
            "model_used": model_used,
            "baseline_loaded_from_disk": self._baseline_loaded_from_disk,
            "baseline_path": str(self.baseline_path),
            "baseline_updated": bool(baseline_updated),
            "baseline_update_count": int(self._baseline_update_count),
        }


def keystroke_stress_score(
    features: Dict[str, float],
    baseline_means: Dict[str, float],
    baseline_stds: Dict[str, float],
    model_path: str | Path | None = None,
) -> Dict[str, object]:
    """Standalone score helper used by inference/pipeline code.

    Expected feature keys:
    velocity, dwell_mean_ms, dwell_std_ms, latency_mean_ms, pause_freq, error_count
    """
    zscores = {
        "velocity_zscore": _safe_z(features["velocity"], baseline_means["velocity"], baseline_stds["velocity"]),
        "dwell_mean_zscore": _safe_z(features["dwell_mean_ms"], baseline_means["dwell_mean_ms"], baseline_stds["dwell_mean_ms"]),
        "dwell_std_zscore": _safe_z(features["dwell_std_ms"], baseline_means["dwell_std_ms"], baseline_stds["dwell_std_ms"]),
        "latency_mean_zscore": _safe_z(features["latency_mean_ms"], baseline_means["latency_mean_ms"], baseline_stds["latency_mean_ms"]),
        "latency_std_zscore": _safe_z(features.get("latency_std_ms", 0.0), baseline_means.get("latency_std_ms", 0.0), max(EPS, baseline_stds.get("latency_std_ms", 1.0))),
        "pause_freq_zscore": _safe_z(features["pause_freq"], baseline_means["pause_freq"], baseline_stds["pause_freq"]),
        "error_count_zscore": _safe_z(features["error_count"], baseline_means["error_count"], baseline_stds["error_count"]),
        "backspace_rate_zscore": _safe_z(features.get("backspace_rate", 0.0), baseline_means.get("backspace_rate", 0.0), max(EPS, baseline_stds.get("backspace_rate", 1.0))),
        "burst_ratio_zscore": _safe_z(features.get("burst_ratio", 0.0), baseline_means.get("burst_ratio", 0.0), max(EPS, baseline_stds.get("burst_ratio", 1.0))),
        "key_variation_zscore": _safe_z(features.get("key_variation", 0.0), baseline_means.get("key_variation", 0.0), max(EPS, baseline_stds.get("key_variation", 1.0))),
    }
    zscores = {k: float(round(v, 4)) for k, v in zscores.items()}
    vector = np.array([[float(zscores[k]) for k in DEFAULT_FEATURE_ORDER]], dtype=float)

    stress_score = None
    model_used = "heuristic"
    if model_path:
        try:
            model = joblib.load(str(model_path))
            if hasattr(model, "predict_proba"):
                stress_score = float(model.predict_proba(vector)[0][1])
                model_used = "trained_model"
        except Exception:
            stress_score = None

    if stress_score is None:
        magnitude = float(np.mean(np.abs(vector[0])))
        stress_score = 1.0 / (1.0 + math.exp(-(magnitude - 1.0)))

    return {
        "modality": "keystroke",
        "keystroke_score": float(round(stress_score, 4)),
        "zscore_features": zscores,
        "feature_vector": vector[0].tolist(),
        "model_used": model_used,
    }


def train_keystroke_model(
    X: np.ndarray,
    y: np.ndarray,
    model_out: str | Path = "keystroke_model.pkl",
    random_state: int = 42,
) -> Dict[str, float]:
    """Train RandomForest and save model.

    Expects:
    - X: shape (n_samples, 6)
    - y: shape (n_samples,) with 0=normal, 1=stressed
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(model, str(model_out))

    return {
        "f1_score": round(f1, 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "mean_stress_probability": float(round(np.mean(y_prob), 4)),
        "model_path": str(model_out),
    }
