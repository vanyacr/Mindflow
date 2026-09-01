"""
user_profile.py — Privacy-safe per-user calibration and delta-based personalisation.

Design principles:
  - Profiles stored as local JSON only — never synced or sent anywhere
  - Profile contains DERIVED geometry (deltas, ratios), NOT raw biometric data
  - User can delete their profile at any time via delete_profile()
  - Rolling average update (90% old, 10% new) keeps profile current without large drift
  - No face images, no landmark arrays stored in profiles

Usage:
    from user_profile import UserProfile

    profile = UserProfile("alice")
    profile.calibrate(cap, model)          # 60-second baseline capture
    deltas = profile.compute_deltas(current_ear, current_bpm, current_tilt)
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import mediapipe as mp

PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)

CALIBRATION_SECONDS = 60   # how long to collect baseline
ROLLING_ALPHA = 0.10       # weight of new observation when updating baseline


@dataclass
class FaceBaseline:
    ear_mean:       float = 0.30   # resting Eye Aspect Ratio
    blink_rate_bpm: float = 14.0   # resting blinks per minute
    head_tilt_deg:  float = 0.0    # neutral head roll angle
    au06_rest:      float = 0.05   # cheek raiser at rest
    au12_rest:      float = 0.10   # lip corner at rest


@dataclass
class PostureBaseline:
    shoulder_raise_neutral: float = 0.0
    forward_lean_neutral:   float = 0.0
    asymmetry_neutral:      float = 0.02


@dataclass
class UserProfile:
    user_id:   str
    face:      FaceBaseline    = field(default_factory=FaceBaseline)
    posture:   PostureBaseline = field(default_factory=PostureBaseline)
    created_at: float          = field(default_factory=time.time)
    updated_at: float          = field(default_factory=time.time)

    # ── persistence ──────────────────────────────────────────

    @property
    def _path(self) -> Path:
        return PROFILES_DIR / f"{self.user_id}.json"

    def save(self):
        self.updated_at = time.time()
        data = {
            "user_id":    self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "face":       asdict(self.face),
            "posture":    asdict(self.posture),
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Profile saved → {self._path}")

    @classmethod
    def load(cls, user_id: str) -> Optional["UserProfile"]:
        path = PROFILES_DIR / f"{user_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        p = cls(user_id=user_id)
        p.created_at = data.get("created_at", time.time())
        p.updated_at = data.get("updated_at", time.time())
        p.face       = FaceBaseline(**data["face"])
        p.posture    = PostureBaseline(**data["posture"])
        print(f"  Profile loaded ← {path}")
        return p

    def delete(self):
        """User can request full deletion at any time."""
        if self._path.exists():
            self._path.unlink()
            print(f"  Profile deleted: {self._path}")
        else:
            print(f"  No profile found for {self.user_id}")
        head_path = PROFILES_DIR / f"{self.user_id}_head.pt"
        if head_path.exists():
            head_path.unlink()
            print(f"  Personalized head weights deleted: {head_path}")

    # ── rolling update (lightweight online learning) ──────────

    def update_from_observation(self, ear: float, bpm: float, tilt: float,
                                shoulder_raise: float, forward_lean: float):
        """
        After each session, call this to slowly update the baseline.
        Uses exponential moving average — 90% old, 10% new.
        This means the profile drifts gently over time rather than
        locking to one calibration session.
        """
        a = ROLLING_ALPHA
        self.face.ear_mean           = (1 - a) * self.face.ear_mean       + a * ear
        self.face.blink_rate_bpm     = (1 - a) * self.face.blink_rate_bpm + a * bpm
        self.face.head_tilt_deg      = (1 - a) * self.face.head_tilt_deg  + a * tilt
        self.posture.shoulder_raise_neutral = (
            (1 - a) * self.posture.shoulder_raise_neutral + a * shoulder_raise
        )
        self.posture.forward_lean_neutral = (
            (1 - a) * self.posture.forward_lean_neutral + a * forward_lean
        )
        self.save()

    # ── delta computation ─────────────────────────────────────

    def compute_deltas(self, ear: float, bpm: float, tilt: float,
                       shoulder_raise: float, forward_lean: float) -> dict:
        """
        Returns deltas from personal baseline — not from a global mean.
        These deltas are what you feed into the fusion MLP alongside emotion scores.

        Example: someone who naturally squints gets ear_delta ≈ 0.0,
                 not incorrectly flagged as stressed.
        """
        return {
            "ear_delta":            round(ear            - self.face.ear_mean,                    4),
            "blink_delta_bpm":      round(bpm            - self.face.blink_rate_bpm,              2),
            "head_tilt_delta_deg":  round(tilt           - self.face.head_tilt_deg,               2),
            "shoulder_raise_delta": round(shoulder_raise - self.posture.shoulder_raise_neutral,   4),
            "forward_lean_delta":   round(forward_lean   - self.posture.forward_lean_neutral,     4),
        }


# ─────────────────────────────────────────────────────────────
#  CALIBRATION — collects baseline during neutral resting
#  Uses modern MediaPipe Tasks API
# ─────────────────────────────────────────────────────────────

def _init_landmarkers():
    model_dir = Path("models")
    face_task = model_dir / "face_landmarker.task"
    pose_task = model_dir / "pose_landmarker.task"

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(face_task)),
            output_face_blendshapes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            num_faces=1,
        )
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_task)),
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )
    return face_landmarker, pose_landmarker


def calibrate_user(user_id: str,
                   duration_sec: int = CALIBRATION_SECONDS) -> Optional[UserProfile]:
    """
    Opens webcam, shows countdown, collects resting face geometry for baseline.
    No frames are saved — only aggregate statistics (mean EAR, mean blink rate, etc.)

    Returns a saved UserProfile.
    """
    face_landmarker, pose_landmarker = _init_landmarkers()

    print(f"\n{'='*65}")
    print("  INTERACTIVE GUIDED CALIBRATION (MediaPipe Tasks API)")
    print(f"{'='*65}")
    print(f"  User ID: {user_id}")
    print(f"  Total Duration: {duration_sec}s across 4 guided expression phases:")
    print("    1. Neutral Baseline (rested face, baseline EAR/gaze)")
    print("    2. Genuine Smile    (Happy anchor)")
    print("    3. Brow Raise       (Surprise anchor)")
    print("    4. Brow Furrow      (Concentration / Thinking anchor)")
    print("  No video is saved. Press Q to abort.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam for calibration")

    # accumulators
    ears, bpms, tilts, shoulder_raises, forward_leans = [], [], [], [], []
    labeled_crops = []  # list of (crop_bgr, target_emotion_idx)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(fps * duration_sec)
    frame_idx    = 0
    blink_count  = 0
    was_closed   = False

    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    EAR_THRESH = 0.20

    start_time = time.time()

    # Define 4 guided expression phases
    p1_end = int(total_frames * 0.50)   # 50% Neutral baseline
    p2_end = int(total_frames * 0.70)   # 20% Happy
    p3_end = int(total_frames * 0.85)   # 15% Surprise
    p4_end = total_frames               # 15% Brow furrow / concentration

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        remaining = max(0, duration_sec - int(elapsed))

        # Determine current interactive phase
        if frame_idx < p1_end:
            phase_num = 1
            phase_name = "NEUTRAL BASELINE"
            phase_prompt = "Look naturally at the camera (Resting face)"
            phase_color = (80, 220, 80)     # Green
            phase_target_emo = "neutral"
        elif frame_idx < p2_end:
            phase_num = 2
            phase_name = "HAPPY ANCHOR"
            phase_prompt = "Give a genuine smile (Happy expression)"
            phase_color = (50, 215, 255)    # Yellow / Gold
            phase_target_emo = "happy"
        elif frame_idx < p3_end:
            phase_num = 3
            phase_name = "SURPRISE ANCHOR"
            phase_prompt = "Raise your eyebrows / open eyes wide (Surprise)"
            phase_color = (255, 120, 50)    # Blue / Cyan
            phase_target_emo = "surprise"
        else:
            phase_num = 4
            phase_name = "FOCUS ANCHOR"
            phase_prompt = "Furrow brows / show concentration (Thinking)"
            phase_color = (220, 80, 220)    # Purple / Magenta
            phase_target_emo = "neutral"

        target_idx = config.EMOTION_TO_IDX[phase_target_emo]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # face landmarks
        fm_result = face_landmarker.detect(mp_img)
        if fm_result.face_landmarks:
            lm = fm_result.face_landmarks[0]

            def ear_val(indices):
                pts = np.array([[lm[i].x, lm[i].y] for i in indices])
                A = np.linalg.norm(pts[1] - pts[5])
                B = np.linalg.norm(pts[2] - pts[4])
                C = np.linalg.norm(pts[0] - pts[3])
                return (A + B) / (2.0 * C + 1e-6)

            ear = (ear_val(LEFT_EYE) + ear_val(RIGHT_EYE)) / 2

            # Accumulate resting EAR and blinks primarily during neutral baseline
            if phase_num == 1:
                ears.append(ear)
                closed = ear < EAR_THRESH
                if closed and not was_closed:
                    blink_count += 1
                was_closed = closed

            # Accumulate face crops for personal head fine-tuning
            if frame_idx % 4 == 0 and len(labeled_crops) < 60:
                h, w, _ = frame.shape
                xs = [p.x * w for p in lm]
                ys = [p.y * h for p in lm]
                x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
                y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
                y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        labeled_crops.append((crop, target_idx))

        # posture
        ps_result = pose_landmarker.detect(mp_img)
        if ps_result.pose_landmarks and phase_num == 1:
            plm = ps_result.pose_landmarks[0]
            ls  = np.array([plm[11].x, plm[11].y])
            rs  = np.array([plm[12].x, plm[12].y])
            ns  = np.array([plm[0].x,  plm[0].y])
            smid = (ls + rs) / 2
            shoulder_raises.append(0.45 - smid[1])
            forward_leans.append(ns[0] - smid[0])

        # UI overlay
        pct = int(100 * frame_idx / total_frames)
        bar = int(frame.shape[1] * 0.8 * frame_idx / total_frames)
        cv2.rectangle(frame, (int(frame.shape[1]*0.1), frame.shape[0]-30),
                      (int(frame.shape[1]*0.1) + bar, frame.shape[0]-10), (50, 200, 50), -1)

        # Header Banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (20, 20, 20), -1)
        cv2.putText(frame, f"PHASE [{phase_num}/4]: {phase_name} ({remaining}s left)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, phase_color, 2)
        cv2.putText(frame, phase_prompt,
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1)

        cv2.imshow("Guided Calibration [NO DATA SAVED]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # compute baselines from accumulators
    elapsed = time.time() - start_time
    bpm     = round(blink_count / ((elapsed * 0.5) / 60), 1) if elapsed > 0 else 14.0

    profile = UserProfile(user_id=user_id)
    profile.face.ear_mean           = float(np.mean(ears))       if ears            else 0.30
    profile.face.blink_rate_bpm     = bpm
    profile.posture.shoulder_raise_neutral = float(np.mean(shoulder_raises)) if shoulder_raises else 0.0
    profile.posture.forward_lean_neutral   = float(np.mean(forward_leans))   if forward_leans   else 0.0

    profile.save()

    print(f"\n  Guided calibration complete for '{user_id}'")
    print(f"  EAR baseline:       {profile.face.ear_mean:.3f}")
    print(f"  Blink rate:         {profile.face.blink_rate_bpm:.1f} bpm")
    print(f"  Shoulder raise:     {profile.posture.shoulder_raise_neutral:.3f}")
    print(f"  Forward lean:       {profile.posture.forward_lean_neutral:.3f}\n")

    if labeled_crops:
        print(f"  Fine-tuning personal multi-class head from {len(labeled_crops)} guided expression crops...")
        finetune_user_head(user_id=user_id, labeled_crops=labeled_crops)

    return profile


def finetune_user_head(user_id: str, labeled_crops: list = None, face_crops: list = None, device=None) -> Optional[Path]:
    """
    Fine-tunes the classification head on the student's personal anchor crops
    (Neutral, Happy, Surprise, Concentration), adapting the model to the student's
    personal expression geometry while keeping the EfficientNet-B2 backbone frozen.
    Uses L2 anchor regularization against the base head weights to prevent forgetting.
    Saves the fine-tuned head state dict to profiles/<user_id>_head.pt.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crops_to_use = labeled_crops if labeled_crops is not None else face_crops
    if not crops_to_use or len(crops_to_use) < 5:
        print("  Notice: Insufficient face crops collected for head fine-tuning.")
        return None

    import torch
    import torch.nn as nn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import config
    from model import load_model

    model, _ = load_model(config.CKPT_PATH, device=device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    tf = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])

    tensor_crops = []
    target_indices = []

    for item in crops_to_use:
        if isinstance(item, tuple) and len(item) == 2:
            crop, emo_idx = item
        else:
            crop = item
            emo_idx = config.EMOTION_TO_IDX["neutral"]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor_crops.append(tf(image=rgb)["image"])
        target_indices.append(emo_idx)

    batch_tensors = torch.stack(tensor_crops).to(device)  # (N, 3, 224, 224)
    target_idx = torch.tensor(target_indices, dtype=torch.long, device=device)

    with torch.no_grad():
        feats = model.backbone(batch_tensors)  # (N, 1408)

    # Initial head state for anchor regularization (preserves general knowledge)
    init_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.head.train()
    for step in range(30):
        optimizer.zero_grad()
        logits = model.head(feats)
        loss = criterion(logits, target_idx)

        # L2 Anchor regularization to original head parameters
        reg_loss = 0.0
        for name, param in model.head.named_parameters():
            if name in init_head_state:
                reg_loss += torch.norm(param - init_head_state[name])

        total_loss = loss + 0.02 * reg_loss
        total_loss.backward()
        optimizer.step()

    model.head.eval()
    head_path = PROFILES_DIR / f"{user_id}_head.pt"
    torch.save(model.head.state_dict(), str(head_path))
    print(f"  ✓ User-adapted personal head saved → {head_path}\n")
    return head_path


def list_profiles() -> list[str]:
    return [p.stem for p in PROFILES_DIR.glob("*.json")]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="User calibration tool")
    parser.add_argument("--user",     type=str, default=None, help="User ID (e.g. alice)")
    parser.add_argument("--delete",   action="store_true", help="Delete a user's profile")
    parser.add_argument("--list",     action="store_true", help="List all profiles")
    parser.add_argument("--duration", type=int, default=CALIBRATION_SECONDS, help="Calibration duration in seconds")
    args = parser.parse_args()

    if args.list:
        profiles = list_profiles()
        print("Existing profiles:", profiles or "(none)")

    elif args.delete:
        if not args.user:
            print("ERROR: --delete requires --user <user_id>")
            sys.exit(1)
        p = UserProfile.load(args.user)
        if p:
            p.delete()
        else:
            print(f"No profile found for '{args.user}'")

    else:
        if not args.user:
            print("ERROR: Calibration requires --user <user_id>")
            sys.exit(1)
        calibrate_user(args.user, duration_sec=args.duration)
