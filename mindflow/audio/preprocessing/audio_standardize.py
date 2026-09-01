"""
Phase 1, step 2 — Standardize audio.

Every file becomes: 16 kHz, mono, WAV, normalized, silence-trimmed,
voice-activity gated.

Requires: librosa, soundfile, numpy, webrtcvad
    pip install librosa soundfile numpy webrtcvad
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import webrtcvad

from config.settings import (
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    TRIM_TOP_DB,
    VAD_AGGRESSIVENESS,
)

logger = logging.getLogger("mindflow.audio_standardize")


def load_and_resample(path: Path) -> np.ndarray:
    """Load audio, force mono, resample to TARGET_SAMPLE_RATE."""
    y, _sr = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=(TARGET_CHANNELS == 1))
    return y.astype(np.float32)


def normalize_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize waveform to avoid clipping while maximizing dynamic range."""
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak < 1e-8:
        return y  # silent clip — leave as-is, caller may choose to drop it
    return y * (target_peak / peak)


def trim_silence(y: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    """Trim leading/trailing silence based on a dB threshold."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size else y


def apply_vad(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE,
              frame_ms: int = 30, aggressiveness: int = VAD_AGGRESSIVENESS) -> np.ndarray:
    """
    Voice-activity gate: drop frames webrtcvad classifies as non-speech.
    webrtcvad requires 16-bit PCM mono at 8k/16k/32k/48k and frames of
    10/20/30 ms.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)

    # Convert float32 [-1, 1] -> int16 PCM bytes
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

    voiced_frames = []
    n_frames = len(pcm16) // frame_len
    for i in range(n_frames):
        frame = pcm16[i * frame_len:(i + 1) * frame_len]
        frame_bytes = frame.tobytes()
        if len(frame_bytes) < frame_len * 2:
            continue
        try:
            if vad.is_speech(frame_bytes, sample_rate=sr):
                voiced_frames.append(frame)
        except Exception:
            # If VAD chokes on a malformed frame, keep it rather than lose audio
            voiced_frames.append(frame)

    if not voiced_frames:
        # Nothing detected as speech (common on acted, whispery, or noisy
        # clips) — fall back to the original signal rather than emit silence.
        return y

    voiced = np.concatenate(voiced_frames).astype(np.float32) / 32767.0
    return voiced


def standardize_audio_file(src_path: Path, dst_path: Path, use_vad: bool = True) -> bool:
    """
    Run the full standardization chain on one file and write the result.
    Returns True on success, False if the file was skipped (e.g. empty/corrupt).
    """
    try:
        y = load_and_resample(src_path)
        y = trim_silence(y)
        if use_vad:
            y = apply_vad(y)
        y = normalize_peak(y)

        if y.size == 0:
            logger.warning("Skipping %s — empty after processing", src_path)
            return False

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), y, TARGET_SAMPLE_RATE, subtype="PCM_16")
        return True

    except Exception as e:
        logger.error("Failed to standardize %s: %s", src_path, e)
        return False


def get_duration_seconds(path: Path) -> float:
    """Duration of a (already standardized or raw) audio file, in seconds."""
    return float(librosa.get_duration(path=str(path)))
