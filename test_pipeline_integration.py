"""
test_pipeline_integration.py — Integration test for consolidated inference & user calibration.
Validates:
1. User profile creation, saving, loading, and delta calculation.
2. Loading the EmotionModel on RTX 4090/CUDA.
3. Running predict_frame with user profile calibration.
4. Verifying exact compliance with the JSON output contract for Srujana's fusion layer.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import json
import numpy as np
import torch
from pathlib import Path

import config
from model import load_model
from user_profile import UserProfile, FaceBaseline, PostureBaseline
import inference as inf


def test_integration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Test UserProfile persistence and deltas ──
    print("\n[1/4] Testing UserProfile serialization and deltas...")
    test_user = "test_student_01"
    profile = UserProfile(
        user_id=test_user,
        face=FaceBaseline(ear_mean=0.32, blink_rate_bpm=16.0, head_tilt_deg=1.5),
        posture=PostureBaseline(shoulder_raise_neutral=0.01, forward_lean_neutral=-0.02)
    )
    profile.save()
    loaded = UserProfile.load(test_user)
    assert loaded is not None, "Failed to reload profile!"
    assert abs(loaded.face.ear_mean - 0.32) < 1e-4
    print("  ✓ UserProfile saved and reloaded successfully")

    deltas = loaded.compute_deltas(ear=0.30, bpm=18.0, tilt=2.0, shoulder_raise=0.03, forward_lean=-0.01)
    print(f"  ✓ Computed personal deltas: {deltas}")
    assert "ear_delta" in deltas and "blink_delta_bpm" in deltas

    # ── 2. Test Personal Head Fine-Tuning ──
    print("\n[2/5] Testing personal head fine-tuning...")
    from user_profile import finetune_user_head
    dummy_crops = [np.full((128, 128, 3), 120 + i, dtype=np.uint8) for i in range(10)]
    head_path = finetune_user_head(user_id=test_user, face_crops=dummy_crops, device=device)
    assert head_path is not None and head_path.exists(), "Head fine-tuning failed to save checkpoint!"
    print("  ✓ Personal head fine-tuned and verified")

    # ── 3. Test Model Loading & Personal Head Weights ──
    print("\n[3/5] Loading emotion classifier model and applying personal head...")
    model, device = load_model(config.CKPT_PATH, device)
    head_state = torch.load(str(head_path), map_location=device)
    model.head.load_state_dict(head_state)
    model.eval()
    print("  ✓ EmotionModel loaded with personalized head weights")

    # ── 4. Test Tracker Instantiation with Profile ──
    print("\n[4/5] Initializing trackers with personal profile...")
    bt, st, sm, sp, dm, es, tl, ss, cal, gz = inf._make_trackers(fps=30, user_profile=loaded)
    assert cal.is_done, "EarCalibrator should be marked done immediately with user profile!"
    print(f"  ✓ Calibration threshold set immediately to: {cal.threshold:.4f}")

    # ── 5. Test Single Frame Inference & Output Contract ──
    print("\n[5/5] Testing predict_frame and JSON output contract...")
    # Create synthetic test frame
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Draw simple circle to simulate face region
    cv2.circle(dummy_frame, (320, 240), 100, (200, 180, 160), -1)

    result = inf.predict_frame(
        model=model,
        frame_bgr=dummy_frame,
        timestamp=0.5,
        frame_idx=15,
        blink_tracker=bt,
        subtle_tracker=st,
        smoother=sm,
        suppressor=sp,
        drowsiness_monitor=dm,
        engagement_scorer=es,
        gaze_tracker=gz,
        window_ms=500,
        save_landmarks=False,
        use_tta=False,
        user_profile=loaded,
    )

    # Verify JSON contract keys
    required_keys = [
        "timestamp", "modality", "emotion", "confidence", "all_scores",
        "window_ms", "drowsiness", "engagement", "features", "error"
    ]
    for k in required_keys:
        assert k in result, f"Missing required JSON contract key: {k}"

    feature_keys = [
        "landmarks", "action_units", "frame_idx", "subtle_expr",
        "blink", "head_pose", "gaze", "posture", "personal_deltas"
    ]
    for fk in feature_keys:
        assert fk in result["features"], f"Missing feature key: {fk}"

    print("  ✓ JSON Contract output verified:")
    print(json.dumps({
        "timestamp": result["timestamp"],
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "engagement": result["engagement"],
        "features": {k: result["features"][k] for k in ["blink", "head_pose", "gaze", "personal_deltas"]},
    }, indent=2))

    # Clean up test profile
    loaded.delete()
    print("\nAll integration checks passed cleanly!")


if __name__ == "__main__":
    test_integration()
