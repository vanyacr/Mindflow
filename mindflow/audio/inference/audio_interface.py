"""
MindFlow — Audio Branch Fusion Interface.

This is what the fusion team (and video/text teammates) actually call.
Give it a path to any audio file; get back the 768-dim embedding plus
emotion/stress/confidence predictions.

Usage (as a library):
    from inference.audio_interface import AudioInference
    audio_model = AudioInference()
    result = audio_model.predict("path/to/clip.wav")
    result["embedding"]        -> list[float], length 768 (THIS is what fusion consumes)
    result["emotion"]          -> str, e.g. "happy"
    result["emotion_probs"]    -> dict[str, float], all 7 class probabilities
    result["stress"]           -> float in [0, 1]
    result["confidence"]       -> float in [0, 1] (untrained placeholder -- see note below)

Usage (command line, for quick manual testing):
    python inference/audio_interface.py path/to/clip.wav

IMPORTANT NOTE ON CONFIDENCE:
    The confidence head has no labeled training data yet (flagged as a known
    gap in Phase 2). Its output right now is NOT meaningful -- it's an
    untrained head producing near-random values. Fusion should either ignore
    this field for now, or treat it as a placeholder until a labeled
    confidence dataset/proxy is identified and Stage 3 trains it properly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import UNIFIED_EMOTIONS
from audio_model.audio_model import AudioModel
from inference.user_calibration import UserProfile, UserProfileCalibrator

SAMPLE_RATE = 16000
DEFAULT_CROP_SECONDS = 6  # matches Stage 1 training clip length


class AudioInference:
    def __init__(
        self,
        stage1_checkpoint: str = "checkpoints/stage1_best.pt",
        stage2_checkpoint: str = "checkpoints/stage2_stress_best.pt",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioModel(num_emotions=len(UNIFIED_EMOTIONS)).to(self.device)

        # Load Stage 1 weights first (emotion head + backbone + pooling + projection),
        # then layer Stage 2's stress head on top if it exists -- this way we always
        # get the most-trained version of every component currently available.
        stage1_path = Path(stage1_checkpoint)
        if not stage1_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_path}")
        stage1_state = torch.load(stage1_path, map_location=self.device, weights_only=True)
        # Stage 1's checkpoint was saved back when stress_head was a single Linear
        # layer; audio_model.py's stress_head has since grown into a 4-layer MLP
        # (see train_stage2_stress_v2.py's docstring). Those old stress_head keys
        # no longer match the current architecture's shapes, so skip them here --
        # the real stress_head weights get loaded fresh from stage2 right below.
        stage1_state = {k: v for k, v in stage1_state.items() if not k.startswith("stress_head")}
        self.model.load_state_dict(stage1_state, strict=False)

        stage2_path = Path(stage2_checkpoint)
        if stage2_path.exists():
            stage2_state = torch.load(stage2_path, map_location=self.device, weights_only=True)
            # Only pull in the stress_head weights from the stage2 checkpoint;
            # everything else in that file is identical to stage1 anyway.
            stress_head_keys = {k: v for k, v in stage2_state.items() if k.startswith("stress_head.")}
            self.model.load_state_dict(stress_head_keys, strict=False)
            print(f"Loaded stress head from {stage2_path}")
        else:
            print(f"WARNING: {stage2_path} not found -- stress predictions will be untrained/meaningless.")

        self.model.eval()
        self.calibrator = UserProfileCalibrator()

    def _load_audio(self, audio_path: str, crop_seconds: float = DEFAULT_CROP_SECONDS) -> torch.Tensor:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # 1. Trim leading and trailing silence
        trimmed_y, _ = librosa.effects.trim(y, top_db=30)
        if len(trimmed_y) > int(0.5 * SAMPLE_RATE):
            y = trimmed_y

        # 2. RMS / Peak Normalization to standardize energy profile across mics & environments
        peak = np.max(np.abs(y))
        if peak > 1e-6:
            y = y / peak * 0.95

        crop_len = int(crop_seconds * SAMPLE_RATE)
        if len(y) > crop_len:
            # Center crop for inference
            start = (len(y) - crop_len) // 2
            y = y[start:start + crop_len]
        elif len(y) < crop_len:
            # Tile/repeat active speech instead of zero padding to prevent low-energy / sad bias
            n_repeats = int(np.ceil(crop_len / max(len(y), 1)))
            y = np.tile(y, n_repeats)[:crop_len]

        return torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, samples)

    def register_user_baseline(self, audio_path: str | Path, user_id: str) -> UserProfile:
        """Records baseline acoustic profile for a user from neutral speech."""
        return self.calibrator.register_user_profile(audio_path, user_id, self)

    @torch.no_grad()
    def predict(
        self,
        audio_path: str | Path,
        crop_seconds: float = DEFAULT_CROP_SECONDS,
        temperature: float = 1.0,
        user_profile: UserProfile | None = None,
    ) -> dict:
        waveform = self._load_audio(str(audio_path), crop_seconds).to(self.device)
        output = self.model(waveform)

        logits = output["emotion_logits"] / max(temperature, 1e-3)
        emotion_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        emotion_idx = int(emotion_probs.argmax())

        raw_result = {
            "embedding": output["embedding"].squeeze(0).cpu().tolist(),  # length 768
            "emotion": UNIFIED_EMOTIONS[emotion_idx],
            "emotion_probs": {emo: float(p) for emo, p in zip(UNIFIED_EMOTIONS, emotion_probs)},
            "stress": float(output["stress"].item()),
            "confidence": float(output["confidence"].item()),  # NOTE: untrained, see module docstring
        }

        if user_profile is not None:
            return self.calibrator.calibrate_inference(raw_result, audio_path, user_profile)

        return raw_result


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python inference/audio_interface.py path/to/audio.wav [profile.json]")
        sys.exit(1)

    audio_model = AudioInference()
    profile = None
    if len(sys.argv) == 3:
        profile = UserProfile.load(sys.argv[2])
        print(f"Loaded User Profile for: {profile.user_id}")

    result = audio_model.predict(sys.argv[1], user_profile=profile)

    print(f"\nFile: {sys.argv[1]}")
    print(f"Emotion: {result.get('calibrated_emotion', result['emotion'])}")
    print("Emotion probabilities:")
    probs = result.get("calibrated_emotion_probs", result["emotion_probs"])
    for emo, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:10s} {prob:.3f}")
    stress_val = result.get("calibrated_stress", result["stress"])
    print(f"Stress (PHQ-8 Continuous [0-1]): {stress_val:.3f}")
    print(f"Confidence: {result['confidence']:.3f} (untrained placeholder)")
    if "calibration_metadata" in result:
        print("\nPersonalized Calibration Metadata:")
        for k, v in result["calibration_metadata"].items():
            print(f"  {k}: {v}")
    print(f"\nEmbedding: length {len(result['embedding'])}, first 5 values: {result['embedding'][:5]}")
