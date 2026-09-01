"""
MindFlow — User Profile Calibration Module.

Why Profile Calibration is Essential:
    Generic global audio models suffer from speaker idiosyncrasy:
    - Soft-spoken or monotone speakers can be misclassified as sad or high stress.
    - Highly animated speakers can be misclassified as angry or surprised.
    - In clinical psychiatry, mental health changes manifest as DEVIATIONS from a
      patient's personal acoustic baseline (flat affect, speech slowing, pause lengthening).

This module enables:
    1. Baseline registration from a neutral onboarding audio clip (~30s).
    2. Multi-feature acoustic extraction (Pitch/F0, RMS energy, pause ratio, speech dynamics).
    3. Relative delta inference (evaluating deviation from baseline rather than absolute thresholds).
    4. Profile persistence (JSON serialization per user).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000


@dataclass
class UserProfile:
    user_id: str
    base_pitch_mean: float
    base_pitch_std: float
    base_energy_rms: float
    base_pause_ratio: float
    base_embedding: List[float]
    base_emotion_probs: Dict[str, float]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, file_path: str | Path) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> 'UserProfile':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class UserProfileCalibrator:
    """
    Handles recording baseline audio profiles and calibrating live inferences
    relative to a user's unique baseline acoustic fingerprint.
    """

    def __init__(self, profiles_dir: str | Path = 'profiles'):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_acoustic_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
        """Extracts pitch (F0), RMS energy, and pause dynamics from raw waveform."""
        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(y ** 2)))

        # 2. Pitch / Fundamental Frequency (F0) via Yin algorithm
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                pitch_mean = float(np.mean(voiced_f0))
                pitch_std = float(np.std(voiced_f0))
            else:
                pitch_mean = 150.0
                pitch_std = 20.0
        except Exception:
            pitch_mean = 150.0
            pitch_std = 20.0

        # 3. Voice Activity & Pause Ratio
        intervals = librosa.effects.split(y, top_db=30)
        active_samples = sum(end - start for start, end in intervals) if len(intervals) > 0 else len(y)
        pause_ratio = float(max(0.0, 1.0 - (active_samples / max(len(y), 1))))

        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_rms': rms,
            'pause_ratio': pause_ratio,
        }

    def register_user_profile(
        self,
        audio_path: str | Path,
        user_id: str,
        audio_interface: Any,
    ) -> UserProfile:
        """
        Processes a baseline neutral calibration audio clip (~30s) and saves
        the user's acoustic and neural baseline profile.
        """
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        acoustics = self.extract_acoustic_features(y, sr)

        # Run through AudioModel for base embedding & emotion distribution
        base_prediction = audio_interface.predict(str(audio_path))

        profile = UserProfile(
            user_id=user_id,
            base_pitch_mean=acoustics['pitch_mean'],
            base_pitch_std=acoustics['pitch_std'],
            base_energy_rms=acoustics['energy_rms'],
            base_pause_ratio=acoustics['pause_ratio'],
            base_embedding=base_prediction['embedding'],
            base_emotion_probs=base_prediction['emotion_probs'],
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        )

        profile_path = self.profiles_dir / f"{user_id}_profile.json"
        profile.save(profile_path)
        print(f"Registered and saved baseline profile for user '{user_id}' -> {profile_path}")
        return profile

    def calibrate_inference(
        self,
        raw_result: Dict[str, Any],
        current_audio_path: str | Path,
        profile: UserProfile,
    ) -> Dict[str, Any]:
        """
        Adjusts raw predictions by comparing current acoustic and embedding
        markers against the user's personal baseline.
        """
        y, sr = librosa.load(current_audio_path, sr=SAMPLE_RATE, mono=True)
        curr_acoustics = self.extract_acoustic_features(y, sr)

        # 1. Pitch Delta (Percentage shift relative to baseline)
        pitch_delta_pct = (curr_acoustics['pitch_mean'] - profile.base_pitch_mean) / max(profile.base_pitch_mean, 1e-3)

        # 2. Energy / Loudness Delta (in dB)
        curr_rms = max(curr_acoustics['energy_rms'], 1e-6)
        base_rms = max(profile.base_energy_rms, 1e-6)
        energy_delta_db = 20.0 * np.log10(curr_rms / base_rms)

        # 3. Pause Ratio Delta
        pause_delta = curr_acoustics['pause_ratio'] - profile.base_pause_ratio

        # 4. Neural Embedding Cosine Similarity
        e_curr = np.array(raw_result['embedding'])
        e_base = np.array(profile.base_embedding)
        cosine_sim = float(np.dot(e_curr, e_base) / (np.linalg.norm(e_curr) * np.linalg.norm(e_base) + 1e-8))

        # 5. Clinical Biomarker Checks:
        clinical_markers = []
        if pitch_delta_pct < -0.15 and pause_delta > 0.10:
            clinical_markers.append('PROSODIC_FLATTENING_DETECTED')
        if energy_delta_db > 6.0 and pitch_delta_pct > 0.20:
            clinical_markers.append('VOCAL_AGITATION_DETECTED')
        if abs(energy_delta_db) < 2.0 and abs(pitch_delta_pct) < 0.05 and cosine_sim > 0.88:
            clinical_markers.append('CONSISTENT_WITH_BASELINE')

        # 6. Calibrated Stress Calculation
        raw_stress = raw_result['stress']
        stress_adjustment = 0.0
        if 'PROSODIC_FLATTENING_DETECTED' in clinical_markers:
            stress_adjustment += 0.12
        if 'VOCAL_AGITATION_DETECTED' in clinical_markers:
            stress_adjustment += 0.15
        if 'CONSISTENT_WITH_BASELINE' in clinical_markers:
            stress_adjustment -= 0.05

        calibrated_stress = float(np.clip(raw_stress + stress_adjustment, 0.0, 1.0))

        # 7. Calibrated Emotion Probability Alignment
        calibrated_probs = dict(raw_result['emotion_probs'])
        if abs(energy_delta_db) < 3.0 and profile.base_energy_rms < 0.02:
            if 'sad' in calibrated_probs and 'neutral' in calibrated_probs:
                transfer = calibrated_probs['sad'] * 0.25
                calibrated_probs['sad'] -= transfer
                calibrated_probs['neutral'] += transfer

        prob_sum = sum(calibrated_probs.values())
        calibrated_probs = {k: float(v / prob_sum) for k, v in calibrated_probs.items()}
        calibrated_emotion = max(calibrated_probs, key=calibrated_probs.get)

        calibrated_output = dict(raw_result)
        calibrated_output['calibrated_emotion'] = calibrated_emotion
        calibrated_output['calibrated_emotion_probs'] = calibrated_probs
        calibrated_output['calibrated_stress'] = calibrated_stress
        calibrated_output['calibration_metadata'] = {
            'user_id': profile.user_id,
            'cosine_similarity_to_base': round(cosine_sim, 3),
            'pitch_delta_pct': round(pitch_delta_pct * 100, 1),
            'energy_delta_db': round(energy_delta_db, 2),
            'pause_ratio_delta': round(pause_delta, 3),
            'clinical_markers': clinical_markers,
        }

        return calibrated_output