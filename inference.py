"""
inference.py — Privacy-first emotion + drowsiness inference.

Real-world robustness fixes in this version:
  1. CLAHE face preprocessing
       Adaptive histogram equalisation applied to the face crop before
       the model sees it. Fixes low-contrast / dim-room / bad-webcam images.
       The model was trained on well-lit dataset images — CLAHE bridges the gap.

  2. Gamma correction on full frame
       Brightens dark frames before YOLO face detection so the face detector
       doesn't miss faces in poor lighting. gamma=1.8 by default (adjustable).

  3. Glasses-aware EAR auto-calibration
       Glasses create reflections that make MediaPipe underestimate eye openness.
       The EarCalibrator collects EAR values for 5 seconds at startup, then sets
       the blink/drowsiness threshold at 65% of the median — adapts to YOUR face
       instead of using a fixed global value.

  4. Neutral suppression (emotion pressure detector)
       Old behaviour: if top emotion confidence < 0.45 → output neutral.
       New behaviour: if any non-neutral emotion has held >0.25 probability
       for 3+ consecutive frames, emit that emotion even at lower confidence.
       This stops the model being "stuck on neutral" when you're clearly
       showing something else.

  5. Colour overhaul — no more grey
       Every colour in the HUD is now high-visibility. Neutral is white.
       All emotion bars use saturated colours. Overlay text is brighter.

  6. Gaze Tracker (NEW)
       Iris landmark-based horizontal gaze estimation.
       Returns gaze_x in [-1, +1]: negative = looking left, positive = looking right.

  7. UI Resize Fix (NEW)
       All display windows are resized to a fixed width (960px) so the
       HUD overlay always matches the display resolution — fixes the
       dimension mismatch when processing portrait or high-res images.

Compatible with:
  Python    : 3.12
  MediaPipe : 0.10.33  (Tasks API)
  NumPy     : 1.26.4

Required model files (run download_models.py once):
  models/face_landmarker.task
  models/pose_landmarker.task

Run:
    python inference.py --webcam --consent
    python inference.py --webcam --consent --no_tta   <- faster on CPU
    python inference.py --image  face.jpg
    python inference.py --video  clip.mp4
"""

import argparse
import json
import math
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ultralytics import YOLO

import config
from model import load_model
from user_profile import UserProfile


# ─────────────────────────────────────────────────────────────
#  CONFIG — tuned for real-world (dim room, glasses, bad webcam)
# ─────────────────────────────────────────────────────────────

CONF_THRESHOLD        = 0.30   # was 0.45 — too aggressive, kept locking on neutral
SMOOTHING_WINDOW      = 5      # was 8 — shorter window lets genuine emotions through
PRESSURE_WINDOW       = 3      # frames a non-neutral must hold before overriding neutral
PRESSURE_MIN_PROB     = 0.25   # minimum probability to count as "pressing"
GAMMA                 = 1.8    # gamma < 1 darkens, > 1 brightens — raise if room is dark
CALIBRATION_SECS      = 5      # seconds to collect EAR baseline at startup
EAR_CALIBRATION_PCT   = 0.65   # threshold = 65% of median resting EAR
MODEL_DIR             = Path("models")
TIMELINE_SECS         = 30
DISPLAY_WIDTH         = 960    # fixed display width — fixes image/UI dimension mismatch


# ─────────────────────────────────────────────────────────────
#  COLOURS — high-visibility, no grey
#
#  BGR format (OpenCV is BGR not RGB).
#  Neutral is WHITE so it's never invisible.
#  All other emotions use saturated primaries.
# ─────────────────────────────────────────────────────────────

EMOTION_COLORS_BGR = {
    "happy":    (50,  220,  80),   # vivid green
    "neutral":  (255, 255, 255),   # WHITE — was grey, now clearly visible
    "sad":      (220,  80,  50),   # vivid blue
    "angry":    (50,   50, 230),   # vivid red
    "fear":     (50,  140, 230),   # orange-red
    "disgust":  (50,  200, 120),   # teal-green
    "surprise": (50,  220, 230),   # vivid yellow
}

# Text colours for labels on dark background — all bright
EMOTION_TEXT_BGR = {
    "happy":    (80,  255, 120),
    "neutral":  (255, 255, 255),
    "sad":      (120, 160, 255),
    "angry":    (80,   80, 255),
    "fear":     (80,  160, 255),
    "disgust":  (80,  230, 160),
    "surprise": (80,  240, 255),
}


# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_face = YOLO("yolov8n-face.pt")

_face_task = MODEL_DIR / "face_landmarker.task"
_pose_task = MODEL_DIR / "pose_landmarker.task"

if not _face_task.exists():
    raise FileNotFoundError(f"Missing: {_face_task}\nRun: python download_models.py")
if not _pose_task.exists():
    raise FileNotFoundError(f"Missing: {_pose_task}\nRun: python download_models.py")

face_mesh = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_face_task)),
        num_faces=1,
        min_face_detection_confidence=0.4,   # lowered — helps with glasses/dim light
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)

pose_est = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_pose_task)),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)

print(f"Device   : {device}")
print(f"FaceMesh : {_face_task.name} OK")
print(f"Pose     : {_pose_task.name} OK")


# ─────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────

_gamma_lut = np.array([
    min(255, int(((i / 255.0) ** (1.0 / GAMMA)) * 255))
    for i in range(256)
], dtype=np.uint8)

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def apply_gamma(frame_bgr: np.ndarray) -> np.ndarray:
    """Brighten dark frames using a precomputed lookup table (fast)."""
    return cv2.LUT(frame_bgr, _gamma_lut)


def apply_clahe(face_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to face crop to normalise contrast.
    Converts to LAB, equalises only the L (lightness) channel,
    then converts back. This preserves colour tone while fixing contrast.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
#  UI RESIZE — fixes image/HUD dimension mismatch
#
#  When an image has a different aspect ratio or resolution than
#  the webcam, the HUD overlay coordinates (bar positions, text
#  anchors) can end up out of bounds or squashed.
#  resize_for_display() normalises every frame to DISPLAY_WIDTH
#  before showing it, so the HUD always fits correctly.
# ─────────────────────────────────────────────────────────────

def resize_for_display(frame: np.ndarray, width: int = DISPLAY_WIDTH) -> np.ndarray:
    """Resize frame to a fixed display width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────
#  TTA TRANSFORMS & TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────

TEMPERATURE = 1.20   # Temperature scaling factor for well-calibrated soft probabilities

_norm  = A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
val_tf = A.Compose([_norm, ToTensorV2()])

TTA_TRANSFORMS = [
    # 1. Standard original crop
    A.Compose([_norm, ToTensorV2()]),
    # 2. Horizontal mirror reflection
    A.Compose([A.HorizontalFlip(p=1.0), _norm, ToTensorV2()]),
    # 3. Multi-scale center zoom (5% zoomed crop)
    A.Compose([
        A.CenterCrop(height=int(config.IMAGE_SIZE * 0.95), width=int(config.IMAGE_SIZE * 0.95)),
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        _norm, ToTensorV2()
    ]),
    # 4. Subtle lighting and contrast enhancement
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.15, 0.15), contrast_limit=(0.10, 0.10), p=1.0),
        _norm, ToTensorV2()
    ]),
]


def tta_predict(model, face_rgb: np.ndarray, use_tta: bool = True) -> dict:
    if not use_tta:
        t = val_tf(image=face_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)
            scaled_logits = logits / TEMPERATURE
            p = F.softmax(scaled_logits, dim=1)[0].cpu().tolist()
        return {config.IDX_TO_EMOTION[i]: round(p[i], 4) for i in range(config.NUM_CLASSES)}

    tensors = [tf(image=face_rgb)["image"] for tf in TTA_TRANSFORMS]
    batch   = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        scaled_logits = logits / TEMPERATURE
        avg_probs = F.softmax(scaled_logits, dim=1).mean(dim=0)
    return {config.IDX_TO_EMOTION[i]: round(avg_probs[i].item(), 4) for i in range(config.NUM_CLASSES)}


# ─────────────────────────────────────────────────────────────
#  FACE DETECTION + CROP
# ─────────────────────────────────────────────────────────────

def detect_face_bbox(frame_bgr):
    res = yolo_face(frame_bgr, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    box = res.boxes[res.boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    h, w = frame_bgr.shape[:2]
    pad  = 20
    return (max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad))


def crop_face(frame_bgr, bbox, size=config.IMAGE_SIZE):
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (size, size))
    return apply_clahe(resized)   # CLAHE applied here on the crop


# ─────────────────────────────────────────────────────────────
#  EAR CALIBRATOR — glasses-aware
# ─────────────────────────────────────────────────────────────

class EarCalibrator:
    def __init__(self, fps=30, secs=CALIBRATION_SECS):
        self._target   = int(fps * secs)
        self._samples  = []
        self._done     = False
        self.threshold = 0.20

    @property
    def is_done(self):
        return self._done

    def collect(self, ear: float):
        if self._done:
            return
        if ear > 0.10:
            self._samples.append(ear)
        if len(self._samples) >= self._target:
            median = float(np.median(self._samples))
            self.threshold = round(median * EAR_CALIBRATION_PCT, 3)
            self._done = True
            print(f"\n  EAR calibration done.")
            print(f"  Median resting EAR : {median:.3f}")
            print(f"  Blink threshold    : {self.threshold:.3f}")
            print(f"  (glasses compensation active)\n")

    def set_baseline(self, ear_baseline: float):
        """Instantly apply a pre-calibrated user profile baseline."""
        self.threshold = round(ear_baseline * EAR_CALIBRATION_PCT, 3)
        self._done = True

    def frames_remaining(self) -> int:
        return max(0, self._target - len(self._samples))


# ─────────────────────────────────────────────────────────────
#  FACE FEATURES
# ─────────────────────────────────────────────────────────────

def extract_face_features(face_bgr, save_landmarks=False):
    rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_mesh.detect(mp_img)

    if not result.face_landmarks:
        return [], {"AU04": 0.0, "AU06": 0.0, "AU12": 0.0}, None

    lm   = result.face_landmarks[0]
    h, w = face_bgr.shape[:2]

    au06 = round(abs(lm[116].y - lm[33].y), 4)
    au12 = round(abs(lm[291].x - lm[61].x), 4)
    au04 = round(abs(lm[70].y  - lm[33].y), 4)

    landmarks = []
    if save_landmarks:
        for i in range(min(68, len(lm))):
            landmarks.append([round(lm[i].x * w, 2), round(lm[i].y * h, 2)])

    return landmarks, {"AU04": au04, "AU06": au06, "AU12": au12}, lm


# ─────────────────────────────────────────────────────────────
#  EAR CALCULATION
# ─────────────────────────────────────────────────────────────

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def _ear(lm, indices):
    pts = np.array([[lm[i].x, lm[i].y] for i in indices])
    A   = np.linalg.norm(pts[1] - pts[5])
    B   = np.linalg.norm(pts[2] - pts[4])
    C   = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ─────────────────────────────────────────────────────────────
#  GAZE TRACKER
#
#  Uses MediaPipe iris landmarks (indices 468 and 473) to estimate
#  horizontal gaze direction.
#
#  For each eye:
#    gaze_ratio = (iris_x - inner_corner_x) / eye_width
#    0.0 = looking far left, 0.5 = centred, 1.0 = looking far right
#
#  The final gaze_x is normalised to [-1, +1]:
#    -1.0 = hard left gaze
#     0.0 = centred / forward
#    +1.0 = hard right gaze
#
#  Note: iris landmarks require face_landmarker.task built with
#  output_face_blendshapes=False and output_facial_transformation_matrixes=False
#  AND with iris detection enabled (the default task includes them).
# ─────────────────────────────────────────────────────────────

LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_CORNERS  = [33, 133]    # inner corner, outer corner
RIGHT_EYE_CORNERS = [362, 263]   # inner corner, outer corner


class GazeTracker:
    def __init__(self):
        self.history = deque(maxlen=30)

    def compute_gaze(self, lm) -> dict:
        """
        lm: list of MediaPipe face landmarks.
        Returns {"gaze_x": float} where -1=left, 0=centre, +1=right.
        Returns 0.0 if landmarks or iris indices are unavailable.
        """
        if lm is None:
            return {"gaze_x": 0.0}

        # Iris landmarks sit beyond index 467; check the list is long enough
        if len(lm) <= max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
            return {"gaze_x": 0.0}

        def _eye_gaze(iris_idx, corner_indices):
            iris  = np.array([lm[iris_idx].x,           lm[iris_idx].y])
            left  = np.array([lm[corner_indices[0]].x,  lm[corner_indices[0]].y])
            right = np.array([lm[corner_indices[1]].x,  lm[corner_indices[1]].y])
            eye_width = np.linalg.norm(right - left) + 1e-6
            return (iris[0] - left[0]) / eye_width

        gx_left  = _eye_gaze(LEFT_IRIS_CENTER,  LEFT_EYE_CORNERS)
        gx_right = _eye_gaze(RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)

        raw_gaze = (gx_left + gx_right) / 2.0
        self.history.append(raw_gaze)

        # Smooth over the history buffer to reduce jitter
        smoothed = float(np.mean(self.history))

        # Map from [0, 1] to [-1, +1]  (0.5 = centre)
        gaze_x = round((smoothed - 0.5) * 2.0, 3)

        return {"gaze_x": gaze_x}


# ─────────────────────────────────────────────────────────────
#  BLINK TRACKER
# ─────────────────────────────────────────────────────────────

class BlinkTracker:
    def __init__(self, fps=30, window_sec=60, ear_threshold=0.20):
        self.fps         = fps
        self.threshold   = ear_threshold
        self.window      = deque(maxlen=int(fps * window_sec))
        self.blink_count = 0
        self.was_closed  = False

    def set_threshold(self, t: float):
        self.threshold = t

    def update(self, lm) -> dict:
        if lm is None:
            return {"ear_left": 0.0, "ear_right": 0.0,
                    "blink_rate_bpm": 0.0, "ear_avg": 0.3}

        ear_l = round(_ear(lm, LEFT_EYE_IDX),  3)
        ear_r = round(_ear(lm, RIGHT_EYE_IDX), 3)
        avg   = (ear_l + ear_r) / 2
        self.window.append(avg)

        closed = avg < self.threshold
        if closed and not self.was_closed:
            self.blink_count += 1
        self.was_closed = closed

        seconds = len(self.window) / self.fps
        bpm     = round(self.blink_count / max(seconds / 60, 1e-3), 1)
        return {"ear_left": ear_l, "ear_right": ear_r,
                "blink_rate_bpm": bpm, "ear_avg": round(avg, 3)}


# ─────────────────────────────────────────────────────────────
#  DROWSINESS MONITOR  (PERCLOS)
# ─────────────────────────────────────────────────────────────

class DrowsinessMonitor:
    LEVELS = ["ALERT", "MILD", "DROWSY", "CRITICAL"]
    _THRESHOLDS = [(0.0, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 1.0)]

    LEVEL_COLORS = {
        "ALERT":    (80,  220,  80),
        "MILD":     (50,  210, 210),
        "DROWSY":   (50,  140, 255),
        "CRITICAL": (50,   50, 240),
    }

    def __init__(self, fps=30, window_sec=60, ear_threshold=0.20):
        self.fps          = fps
        self.threshold    = ear_threshold
        self._window      = deque(maxlen=int(fps * window_sec))
        self._alert_count = 0

    def set_threshold(self, t: float):
        self.threshold = t

    def update(self, ear_avg: float) -> dict:
        closed = 1 if ear_avg < self.threshold else 0
        self._window.append(closed)
        perclos = sum(self._window) / max(len(self._window), 1)

        level = "ALERT"
        for i, (lo, hi) in enumerate(self._THRESHOLDS):
            if lo <= perclos < hi:
                level = self.LEVELS[i]
                break

        if level in ("DROWSY", "CRITICAL"):
            self._alert_count += 1

        return {
            "perclos":     round(perclos, 3),
            "level":       level,
            "alert_count": self._alert_count,
        }


# ─────────────────────────────────────────────────────────────
#  NEUTRAL SUPPRESSOR  (emotion pressure detector)
# ─────────────────────────────────────────────────────────────

class NeutralSuppressor:
    def __init__(self, window=PRESSURE_WINDOW, min_prob=PRESSURE_MIN_PROB):
        self.window   = window
        self.min_prob = min_prob
        self._recent  = deque(maxlen=window)

    def apply(self, top_emotion: str, confidence: float,
              raw_scores: dict) -> tuple[str, float]:
        self._recent.append(raw_scores)

        if top_emotion != "neutral":
            return top_emotion, confidence

        if len(self._recent) < self.window:
            return top_emotion, confidence

        candidates = {}
        for emo in config.EMOTIONS:
            if emo == "neutral":
                continue
            scores_for_emo = [frame.get(emo, 0.0) for frame in self._recent]
            if all(s >= self.min_prob for s in scores_for_emo):
                avg_score = float(np.mean(scores_for_emo))
                candidates[emo] = avg_score

        if not candidates:
            return top_emotion, confidence

        best_emo   = max(candidates, key=candidates.get)
        best_score = candidates[best_emo]
        return best_emo, round(best_score, 4)


# ─────────────────────────────────────────────────────────────
#  ROLLING SOFTMAX SMOOTHER
# ─────────────────────────────────────────────────────────────

class SoftmaxSmoother:
    def __init__(self, window=SMOOTHING_WINDOW, threshold=CONF_THRESHOLD):
        self.window    = deque(maxlen=window)
        self.threshold = threshold

    def update(self, probs: dict):
        self.window.append(probs)
        avg         = {e: float(np.mean([w[e] for w in self.window])) for e in probs}
        top_emotion = max(avg, key=avg.get)
        confidence  = round(avg[top_emotion], 4)
        if confidence < self.threshold:
            top_emotion = "neutral"
        return top_emotion, confidence, {k: round(v, 4) for k, v in avg.items()}


# ─────────────────────────────────────────────────────────────
#  ENGAGEMENT SCORER
# ─────────────────────────────────────────────────────────────

class EngagementScorer:
    def __init__(self):
        self._history = deque(maxlen=90)

    def compute(self, emotion_scores: dict, blink_rate: float,
                yaw: float, drowsy_level: str) -> int:
        score    = 50
        positive = emotion_scores.get("happy", 0) + emotion_scores.get("surprise", 0)
        negative = (emotion_scores.get("sad",     0) * 0.8 +
                    emotion_scores.get("fear",    0) * 0.5 +
                    emotion_scores.get("neutral", 0) * 0.3)
        score += int(positive * 20)
        score -= int(negative * 15)
        if 10 <= blink_rate <= 18:
            score += 8
        elif blink_rate < 5 or blink_rate > 25:
            score -= 8
        if abs(yaw) > 30:
            score -= 15
        elif abs(yaw) > 20:
            score -= 8
        penalties = {"ALERT": 0, "MILD": -5, "DROWSY": -20, "CRITICAL": -40}
        score    += penalties.get(drowsy_level, 0)
        final     = max(0, min(100, score))
        self._history.append(final)
        return final

    @property
    def smoothed(self) -> int:
        return int(np.mean(self._history)) if self._history else 50


# ─────────────────────────────────────────────────────────────
#  EMOTION TIMELINE
# ─────────────────────────────────────────────────────────────

class EmotionTimeline:
    def __init__(self, window_sec=TIMELINE_SECS, fps=30):
        self.fps        = fps
        self.window_sec = window_sec
        self._buf       = deque(maxlen=fps)
        self._history   = deque(maxlen=window_sec)
        self._cnt       = 0

    def update(self, emotion: str):
        self._buf.append(emotion)
        self._cnt += 1
        if self._cnt % self.fps == 0:
            dominant = Counter(self._buf).most_common(1)[0][0]
            self._history.append(dominant)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self._history:
            return frame
        h, w      = frame.shape[:2]
        strip_h   = 20
        strip_y   = h - strip_h
        bar_w     = max(1, w // self.window_sec)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, strip_y), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, emo in enumerate(self._history):
            x1    = i * bar_w
            x2    = x1 + bar_w
            color = EMOTION_COLORS_BGR.get(emo, (180, 180, 180))
            cv2.rectangle(frame, (x1, strip_y + 2), (x2, h - 2), color, -1)

        cv2.putText(frame, f"last {self.window_sec}s", (4, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        return frame


# ─────────────────────────────────────────────────────────────
#  SESSION STATS
# ─────────────────────────────────────────────────────────────

class SessionStats:
    def __init__(self):
        self._emotions      = []
        self._engagement    = []
        self._drowsy_counts = Counter()
        self._start_time    = time.time()

    def update(self, emotion: str, engagement: int, drowsy_level: str):
        self._emotions.append(emotion)
        self._engagement.append(engagement)
        self._drowsy_counts[drowsy_level] += 1

    def print_report(self):
        duration = time.time() - self._start_time
        n        = max(len(self._emotions), 1)
        top3     = Counter(self._emotions).most_common(3)
        avg_eng  = int(np.mean(self._engagement)) if self._engagement else 0

        print("\n" + "=" * 52)
        print("  SESSION REPORT")
        print("=" * 52)
        print(f"  Duration         : {duration:.0f}s")
        print(f"  Frames analysed  : {n}")
        print(f"\n  Dominant emotions:")
        for emo, cnt in top3:
            pct = 100 * cnt / n
            bar = chr(9608) * int(pct / 4)
            print(f"    {emo:<10} {pct:5.1f}%  {bar}")
        print(f"\n  Avg engagement   : {avg_eng}/100")
        print(f"\n  Drowsiness breakdown:")
        for lvl in DrowsinessMonitor.LEVELS:
            cnt = self._drowsy_counts.get(lvl, 0)
            pct = 100 * cnt / n
            print(f"    {lvl:<10} {pct:5.1f}%")
        print("=" * 52 + "\n")


# ─────────────────────────────────────────────────────────────
#  AUDIO ALERT
# ─────────────────────────────────────────────────────────────

def _beep():
    try:
        import winsound
        winsound.Beep(880, 350)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
#  HEAD POSE
# ─────────────────────────────────────────────────────────────

_MODEL_PTS = np.array([
    [0.,    0.,    0.   ],
    [0.,   -330., -65.  ],
    [-225., 170., -135. ],
    [225.,  170., -135. ],
    [-150.,-150., -125. ],
    [150., -150., -125. ],
], dtype=np.float64)
_LM_IDX = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(lm, fw, fh) -> dict:
    if lm is None:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    pts = np.array([[lm[i].x * fw, lm[i].y * fh] for i in _LM_IDX], dtype=np.float64)
    cam = np.array([[fw,0,fw/2],[0,fw,fh/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_PTS, pts, cam, np.zeros((4,1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    return {
        "pitch": round(math.degrees(math.atan2(-rmat[2,0], sy)),          2),
        "yaw":   round(math.degrees(math.atan2( rmat[1,0], rmat[0,0])),  2),
        "roll":  round(math.degrees(math.atan2( rmat[2,1], rmat[2,2])),  2),
    }


# ─────────────────────────────────────────────────────────────
#  SUBTLE EXPRESSION
# ─────────────────────────────────────────────────────────────

class SubtleExprTracker:
    def __init__(self):
        self.prev_gray = None
        self.flow_hist = deque(maxlen=15)

    def update(self, face_bgr) -> dict:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return {"optical_flow_magnitude": 0.0, "expression_change_rate": 0.0}
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = round(float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))), 4)
        self.flow_hist.append(mag)
        self.prev_gray = gray
        change = round(float(np.std(self.flow_hist)), 4) if len(self.flow_hist) > 2 else 0.0
        return {"optical_flow_magnitude": mag, "expression_change_rate": change}


# ─────────────────────────────────────────────────────────────
#  POSTURE
# ─────────────────────────────────────────────────────────────

def extract_posture(frame_bgr) -> dict:
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_est.detect(mp_img)
    if not result.pose_landmarks:
        return {"shoulder_raise": 0.0, "forward_lean": 0.0, "asymmetry": 0.0}
    lm  = result.pose_landmarks[0]
    ls  = np.array([lm[11].x, lm[11].y])
    rs  = np.array([lm[12].x, lm[12].y])
    ns  = np.array([lm[0].x,  lm[0].y])
    mid = (ls + rs) / 2
    return {
        "shoulder_raise": round(float(0.45 - mid[1]),      3),
        "forward_lean":   round(float(ns[0] - mid[0]),     3),
        "asymmetry":      round(float(abs(ls[1] - rs[1])), 3),
    }


# ─────────────────────────────────────────────────────────────
#  DRAW OVERLAY — polished HUD, gaze row added
#
#  Layout:
#   LEFT PANEL  (0..260): title, emotion, confidence, TTA,
#                          blink, EAR, pitch/yaw, gaze, shoulder, engagement bar
#   BOTTOM-LEFT badge  : drowsiness
#   TOP-CENTER banner  : alert when drowsy/critical
#   RIGHT PANEL        : emotion probability bars (w-220 onwards)
#   BOTTOM-RIGHT       : LOCAL ONLY privacy tag
#   BOTTOM STRIP       : emotion timeline
# ─────────────────────────────────────────────────────────────

def _eng_color(score: int):
    if score >= 70:
        return (50, 230,  80)   # bright green
    elif score >= 40:
        return (50, 210, 220)   # cyan
    else:
        return (50,  80, 230)   # bright red


def _gaze_label(gaze_x: float) -> str:
    if gaze_x < -0.25:
        return "LEFT"
    elif gaze_x > 0.25:
        return "RIGHT"
    else:
        return "CTR"


def draw_overlay(frame: np.ndarray, result: dict,
                 calibrator: EarCalibrator = None,
                 timeline: EmotionTimeline = None) -> np.ndarray:

    h, w = frame.shape[:2]

    # ── calibration banner (first 5 seconds) ──
    if calibrator and not calibrator.is_done:
        rem = calibrator.frames_remaining()
        cv2.rectangle(frame, (0, 0), (w, 55), (18, 18, 18), -1)
        cv2.putText(frame, "AI EMOTION SYSTEM",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
        cv2.putText(frame,
                    f"Calibrating EAR... sit normally  ({rem} frames left)",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 230, 230), 2)
        if timeline:
            timeline.draw(frame)
        return frame

    if result["error"]:
        cv2.rectangle(frame, (0, 0), (w, 46), (18, 18, 18), -1)
        cv2.putText(frame, "AI EMOTION SYSTEM",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
        cv2.putText(frame, result["error"], (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)
        if timeline:
            timeline.draw(frame)
        return frame

    emo       = result["emotion"]
    conf      = result["confidence"]
    tta       = result.get("tta_enabled", False)
    drowsy    = result.get("drowsiness", {})
    engage    = result.get("engagement",  50)
    feat      = result["features"]
    hp        = feat["head_pose"]
    ps        = feat["posture"]
    blink     = feat["blink"]
    gaze      = feat.get("gaze", {"gaze_x": 0.0})
    dlevel    = drowsy.get("level",   "ALERT")
    perclos   = drowsy.get("perclos", 0.0)

    emo_col   = EMOTION_TEXT_BGR.get(emo, (255, 255, 255))
    drowsy_c  = DrowsinessMonitor.LEVEL_COLORS.get(dlevel, (200, 200, 200))

    # ── left sidebar background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 270), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # ── title bar ──
    cv2.rectangle(frame, (0, 0), (260, 22), (25, 25, 40), -1)
    cv2.putText(frame, "AI EMOTION SYSTEM",
                (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 200, 255), 1)

    # ── TTA pill ──
    tta_color = (60, 180, 60) if tta else (60, 60, 180)
    cv2.rectangle(frame, (180, 3), (257, 19), tta_color, -1)
    cv2.putText(frame, "TTA ON" if tta else "TTA OFF",
                (184, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1)

    # ── emotion name (large) ──
    cv2.putText(frame, f"{emo.upper()}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, emo_col, 3)

    # ── confidence ──
    cv2.putText(frame, f"conf: {conf:.2f}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, emo_col, 1)

    # ── divider ──
    cv2.line(frame, (8, 80), (252, 80), (50, 50, 60), 1)

    # ── physiology rows ──
    y = 96
    def _row(label, value, color=(190, 215, 255)):
        nonlocal y
        cv2.putText(frame, f"{label:<7} {value}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 19

    _row("blink", f"{blink['blink_rate_bpm']:.0f} bpm")
    _row("EAR",   f"{blink['ear_avg']:.3f}")
    _row("pitch",  f"{hp['pitch']:+.1f}")
    _row("yaw",    f"{hp['yaw']:+.1f}")

    # ── gaze row — NEW ──
    gaze_x     = gaze.get("gaze_x", 0.0)
    gaze_dir   = _gaze_label(gaze_x)
    gaze_color = (80, 255, 200) if gaze_dir == "CTR" else (80, 200, 255)
    _row("gaze", f"{gaze_x:+.2f}  {gaze_dir}", color=gaze_color)

    _row("shoul", f"{ps['shoulder_raise']:+.2f}")

    # ── divider ──
    cv2.line(frame, (8, y), (252, y), (50, 50, 60), 1)
    y += 6

    # ── engagement bar ──
    eng_c    = _eng_color(engage)
    bar_max  = 200
    bar_fill = int(engage / 100 * bar_max)
    cv2.rectangle(frame, (10, y),            (10 + bar_max, y + 13), (50, 50, 55), -1)
    cv2.rectangle(frame, (10, y),            (10 + bar_fill, y + 13), eng_c,       -1)
    cv2.putText(frame, f"engage {engage:3d}%",
                (10, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, eng_c, 1)

    # ── drowsiness badge (bottom-left) ──
    badge_y = h - 52
    cv2.rectangle(frame, (8, badge_y), (210, badge_y + 32), (12, 12, 12), -1)
    cv2.rectangle(frame, (8, badge_y), (210, badge_y + 32), drowsy_c, 1)
    cv2.putText(frame, f"drowsy: {dlevel}  {perclos:.0%}",
                (13, badge_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, drowsy_c, 1)

    # ── DROWSY / CRITICAL alert banner ──
    if dlevel in ("DROWSY", "CRITICAL"):
        msg   = "WAKE UP!" if dlevel == "CRITICAL" else "DROWSY DETECTED"
        color = (50,  50, 255) if dlevel == "CRITICAL" else (50, 150, 255)
        cx    = w // 2
        cv2.rectangle(frame, (cx - 148, 26), (cx + 148, 56), (12, 12, 12), -1)
        cv2.putText(frame, msg, (cx - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)

    # ── emotion probability bars (right panel) ──
    bar_x = w - 220
    by    = 50
    cv2.putText(frame, "SCORES", (bar_x + 155, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 180, 210), 1)
    for emo_name, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        bw  = int(score * 155)
        col = EMOTION_COLORS_BGR.get(emo_name, (180, 180, 180))
        cv2.rectangle(frame, (bar_x, by), (bar_x + bw, by + 14), col, -1)
        txt_col = EMOTION_TEXT_BGR.get(emo_name, (240, 240, 240))
        cv2.putText(frame, f"{emo_name[:3]} {score:.2f}",
                    (bar_x + 160, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1)
        by += 18

    # ── privacy tag ──
    cv2.putText(frame, "LOCAL ONLY", (w - 112, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 200, 80), 1)

    # ── timeline ──
    if timeline:
        timeline.draw(frame)

    return frame


# ─────────────────────────────────────────────────────────────
#  CORE PREDICT
# ─────────────────────────────────────────────────────────────

def predict_frame(
    model,
    frame_bgr: np.ndarray,
    timestamp: float,
    frame_idx: int,
    blink_tracker: BlinkTracker,
    subtle_tracker: SubtleExprTracker,
    smoother: SoftmaxSmoother,
    suppressor: NeutralSuppressor,
    drowsiness_monitor: DrowsinessMonitor,
    engagement_scorer: EngagementScorer,
    gaze_tracker: GazeTracker,          # NEW
    window_ms: int = 500,
    save_landmarks: bool = False,
    use_tta: bool = True,
    user_profile: UserProfile = None,
) -> dict:

    _empty = {
        "landmarks": [], "action_units": {}, "frame_idx": frame_idx,
        "subtle_expr": {"optical_flow_magnitude": 0.0, "expression_change_rate": 0.0},
        "blink":       {"ear_left": 0.0, "ear_right": 0.0,
                        "blink_rate_bpm": 0.0, "ear_avg": 0.3},
        "head_pose":   {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "posture":     {"shoulder_raise": 0.0, "forward_lean": 0.0, "asymmetry": 0.0},
        "gaze":        {"gaze_x": 0.0},
        "personal_deltas": None,
    }

    bbox = detect_face_bbox(frame_bgr)
    if bbox is None:
        drowsy_data = drowsiness_monitor.update(0.3)
        return {
            "timestamp": timestamp, "modality": "video",
            "emotion": "neutral", "confidence": 0.0,
            "all_scores": {e: 0.0 for e in config.EMOTIONS},
            "window_ms": window_ms, "tta_enabled": use_tta,
            "drowsiness": drowsy_data, "engagement": 50,
            "features": _empty, "error": "no face detected",
        }

    face = crop_face(frame_bgr, bbox)
    if face is None:
        drowsy_data = drowsiness_monitor.update(0.3)
        return {
            "timestamp": timestamp, "modality": "video",
            "emotion": "neutral", "confidence": 0.0,
            "all_scores": {e: 0.0 for e in config.EMOTIONS},
            "window_ms": window_ms, "tta_enabled": use_tta,
            "drowsiness": drowsy_data, "engagement": 50,
            "features": _empty, "error": "face crop failed",
        }

    landmarks, action_units, lm_raw = extract_face_features(face, save_landmarks)
    blink_data  = blink_tracker.update(lm_raw)
    subtle_data = subtle_tracker.update(face)
    head_pose   = estimate_head_pose(lm_raw, face.shape[1], face.shape[0])
    gaze_data   = gaze_tracker.compute_gaze(lm_raw)    # NEW
    posture     = extract_posture(frame_bgr)

    drowsy_data = drowsiness_monitor.update(blink_data["ear_avg"])

    if drowsy_data["level"] == "CRITICAL" and frame_idx % 90 == 0:
        _beep()

    face_rgb   = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    raw_scores = tta_predict(model, face_rgb, use_tta=use_tta)

    top_emotion, confidence, all_scores = smoother.update(raw_scores)
    top_emotion, confidence = suppressor.apply(top_emotion, confidence, raw_scores)

    engagement = engagement_scorer.compute(
        raw_scores,
        blink_data["blink_rate_bpm"],
        head_pose["yaw"],
        drowsy_data["level"],
    )

    personal_deltas = None
    if user_profile is not None:
        personal_deltas = user_profile.compute_deltas(
            ear=blink_data["ear_avg"],
            bpm=blink_data["blink_rate_bpm"],
            tilt=head_pose.get("roll", 0.0),
            shoulder_raise=posture.get("shoulder_raise", 0.0),
            forward_lean=posture.get("forward_lean", 0.0),
        )

    return {
        "timestamp":   timestamp,
        "modality":    "video",
        "emotion":     top_emotion,
        "confidence":  confidence,
        "all_scores":  all_scores,
        "window_ms":   window_ms,
        "tta_enabled": use_tta,
        "drowsiness":  drowsy_data,
        "engagement":  engagement,
        "features": {
            "landmarks":       landmarks,
            "action_units":    action_units,
            "frame_idx":       frame_idx,
            "subtle_expr":     subtle_data,
            "blink":           blink_data,
            "head_pose":       head_pose,
            "gaze":            gaze_data,    # NEW
            "posture":         posture,
            "personal_deltas": personal_deltas,
        },
        "error": None,
    }


# ─────────────────────────────────────────────────────────────
#  TRACKER FACTORY
# ─────────────────────────────────────────────────────────────

def _make_trackers(fps=30, user_profile: UserProfile = None):
    cal = EarCalibrator(fps=fps)
    bt  = BlinkTracker(fps=fps)
    dm  = DrowsinessMonitor(fps=fps)

    if user_profile is not None and user_profile.face.ear_mean > 0:
        cal.set_baseline(user_profile.face.ear_mean)
        _sync_thresholds(cal, bt, dm)
        print(f"  [Personal Calibration] Active for '{user_profile.user_id}' "
              f"(Resting EAR: {user_profile.face.ear_mean:.3f}, Blink: {user_profile.face.blink_rate_bpm:.1f} bpm)")

    return (
        bt,
        SubtleExprTracker(),
        SoftmaxSmoother(),
        NeutralSuppressor(),
        dm,
        EngagementScorer(),
        EmotionTimeline(fps=fps),
        SessionStats(),
        cal,
        GazeTracker(),   # NEW — always last after cal
    )


def _sync_thresholds(calibrator: EarCalibrator,
                     blink_tracker: BlinkTracker,
                     drowsiness_monitor: DrowsinessMonitor):
    if calibrator.is_done:
        blink_tracker.set_threshold(calibrator.threshold)
        drowsiness_monitor.set_threshold(calibrator.threshold)


# ─────────────────────────────────────────────────────────────
#  MODES
# ─────────────────────────────────────────────────────────────

def run_image(model, path: str, save_landmarks=False, use_tta=True,
              user_profile: UserProfile = None):
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: Cannot read: {path}"); return

    # Resize to display width BEFORE processing so overlay coords match display
    frame   = resize_for_display(frame)
    frame_g = apply_gamma(frame)

    bt, st, sm, sp, dm, es, tl, ss, cal, gz = _make_trackers(user_profile=user_profile)
    res = predict_frame(model, frame_g, 0.0, 0, bt, st, sm, sp, dm, es, gz,
                        save_landmarks=save_landmarks, use_tta=use_tta,
                        user_profile=user_profile)
    print(json.dumps(res, indent=2))
    draw_overlay(frame, res, cal, tl)
    cv2.imshow("Emotion [LOCAL ONLY]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model, path: str, window_ms=500, save_landmarks=False,
              keep_session=False, use_tta=True, user_profile: UserProfile = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {path}"); return

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * window_ms / 1000))
    bt, st, sm, sp, dm, es, tl, ss, cal, gz = _make_trackers(int(fps), user_profile=user_profile)

    outputs, frame_idx = [], 0
    print(f"FPS: {fps:.1f}  TTA: {'ON' if use_tta else 'OFF'}  | Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = resize_for_display(frame)   # fix dimension mismatch
        frame_g = apply_gamma(frame)

        if frame_idx % step == 0:
            if not cal.is_done:
                bbox = detect_face_bbox(frame_g)
                if bbox:
                    face_c = crop_face(frame_g, bbox)
                    if face_c is not None:
                        _, _, lm_raw = extract_face_features(face_c)
                        if lm_raw:
                            ear_l = _ear(lm_raw, LEFT_EYE_IDX)
                            ear_r = _ear(lm_raw, RIGHT_EYE_IDX)
                            cal.collect((ear_l + ear_r) / 2)
                _sync_thresholds(cal, bt, dm)

            ts  = round(frame_idx / fps, 3)
            res = predict_frame(model, frame_g, ts, frame_idx,
                                bt, st, sm, sp, dm, es, gz,
                                window_ms, save_landmarks, use_tta,
                                user_profile=user_profile)
            outputs.append(res)
            tl.update(res["emotion"])
            ss.update(res["emotion"], res["engagement"], res["drowsiness"]["level"])
            print(f"t={ts:6.2f}s  {res['emotion']:<10}  "
                  f"eng:{res['engagement']:3d}  "
                  f"drowsy:{res['drowsiness']['level']}  "
                  f"gaze:{res['features']['gaze']['gaze_x']:+.2f}")
            draw_overlay(frame, res, cal, tl)

        cv2.imshow("Emotion [LOCAL ONLY]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    ss.print_report()

    out_file = Path(path).stem + "_emotion_results.json"
    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {len(outputs)} windows → {out_file}")


def run_webcam(model, window_ms=500, save_landmarks=False,
               keep_session=False, use_tta=True, user_profile: UserProfile = None):
    print("\n" + "=" * 58)
    print("  PRIVACY NOTICE — all processing local, nothing saved")
    print(f"  TTA: {'ON' if use_tta else 'OFF'}")
    if user_profile:
        print(f"  User Profile: '{user_profile.user_id}' loaded (EAR baseline: {user_profile.face.ear_mean:.3f})")
    else:
        print("  Sit normally for 5 seconds — EAR will auto-calibrate")
        print("  (compensates for glasses + dim lighting)")
    print("  Press Q to stop and see session report")
    print("=" * 58 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return

    fps  = 30
    step = max(1, int(fps * window_ms / 1000))
    bt, st, sm, sp, dm, es, tl, ss, cal, gz = _make_trackers(fps, user_profile=user_profile)

    frame_idx = 0
    last_res  = None
    outputs   = []

    _cal_dummy = {
        "error": None, "emotion": "neutral", "confidence": 0.0,
        "all_scores": {}, "tta_enabled": use_tta,
        "drowsiness": {}, "engagement": 50,
        "features": {
            "head_pose": {"pitch": 0, "yaw": 0, "roll": 0},
            "posture":   {"shoulder_raise": 0, "forward_lean": 0, "asymmetry": 0},
            "blink":     {"blink_rate_bpm": 0, "ear_avg": 0.3},
            "gaze":      {"gaze_x": 0.0},
        },
    }

    print("Live — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = resize_for_display(frame)   # fix dimension mismatch
        frame_g = apply_gamma(frame)

        if frame_idx % step == 0:
            # ── EAR calibration window ──
            if not cal.is_done:
                bbox = detect_face_bbox(frame_g)
                if bbox:
                    face_c = crop_face(frame_g, bbox)
                    if face_c is not None:
                        _, _, lm_raw = extract_face_features(face_c)
                        if lm_raw:
                            ear_l = _ear(lm_raw, LEFT_EYE_IDX)
                            ear_r = _ear(lm_raw, RIGHT_EYE_IDX)
                            cal.collect((ear_l + ear_r) / 2)
                _sync_thresholds(cal, bt, dm)
                draw_overlay(frame, _cal_dummy, cal, tl)
                cv2.imshow("Live Emotion [LOCAL ONLY]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_idx += 1
                continue

            ts       = round(frame_idx / fps, 3)
            last_res = predict_frame(model, frame_g, ts, frame_idx,
                                     bt, st, sm, sp, dm, es, gz,
                                     window_ms, save_landmarks, use_tta,
                                     user_profile=user_profile)
            outputs.append(last_res)
            tl.update(last_res["emotion"])
            ss.update(last_res["emotion"], last_res["engagement"],
                      last_res["drowsiness"]["level"])
            print(f"t={ts:6.2f}s  {last_res['emotion']:<10}  "
                  f"eng:{last_res['engagement']:3d}  "
                  f"drowsy:{last_res['drowsiness']['level']}  "
                  f"gaze:{last_res['features']['gaze']['gaze_x']:+.2f}")

        if last_res:
            draw_overlay(frame, last_res, cal, tl)
        elif not cal.is_done:
            draw_overlay(frame, _cal_dummy, cal, tl)

        cv2.imshow("Live Emotion [LOCAL ONLY]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    ss.print_report()

    if keep_session and outputs:
        out_file = f"webcam_session_{int(time.time())}.json"
        with open(out_file, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"Session saved → {out_file}")
    else:
        print("Session discarded (no raw frames were ever saved).")


# ─────────────────────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Privacy-first emotion + drowsiness inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str,            help="Path to image file")
    group.add_argument("--video",  type=str,            help="Path to video file")
    group.add_argument("--webcam", action="store_true", help="Live webcam mode")

    parser.add_argument("--user",           type=str,  default=None,
                        help="User ID for personal calibration profile (e.g. --user alice)")
    parser.add_argument("--window_ms",      type=int,  default=500)
    parser.add_argument("--consent",        action="store_true",
                        help="Required for webcam mode")
    parser.add_argument("--save_landmarks", action="store_true")
    parser.add_argument("--keep_session",   action="store_true",
                        help="Save session JSON after quitting")
    parser.add_argument("--no_tta",         action="store_true",
                        help="Disable TTA — faster on slow machines")
    parser.add_argument("--gamma",          type=float, default=GAMMA,
                        help=f"Gamma for frame brightening (default {GAMMA}). "
                             "Increase if your room is very dark.")
    parser.add_argument("--display_width",  type=int, default=DISPLAY_WIDTH,
                        help=f"Window display width in pixels (default {DISPLAY_WIDTH}). "
                             "Increase for large monitors.")

    args    = parser.parse_args()
    use_tta = not args.no_tta

    # Allow gamma override from command line
    if args.gamma != GAMMA:
        new_lut = np.array([
            min(255, int(((i / 255.0) ** (1.0 / args.gamma)) * 255))
            for i in range(256)
        ], dtype=np.uint8)
        _gamma_lut[:] = new_lut
        print(f"Gamma override: {args.gamma}")

    # Allow display width override from command line
    if args.display_width != DISPLAY_WIDTH:
        import inference as _self
        _self.DISPLAY_WIDTH = args.display_width
        print(f"Display width override: {args.display_width}px")

    if args.webcam and not args.consent:
        print("\nERROR: Webcam mode requires --consent flag.")
        print("Run: python inference.py --webcam --consent\n")
        exit(1)

    user_profile = None
    if args.user:
        user_profile = UserProfile.load(args.user)
        if user_profile is None:
            print(f"Notice: No saved profile found for user '{args.user}'. Starting with auto-calibration.")

    print(f"\nLoading model from {config.CKPT_PATH} ...")
    model, device = load_model(config.CKPT_PATH, device)
    if args.user:
        head_path = Path("profiles") / f"{args.user}_head.pt"
        if head_path.exists():
            head_state = torch.load(str(head_path), map_location=device)
            model.head.load_state_dict(head_state)
            print(f"Personal Head: Loaded user-adapted weights from {head_path}")
    print(f"TTA          : {'ON' if use_tta else 'OFF'}")
    print(f"Gamma        : {args.gamma} (frame brightening)")
    print(f"CLAHE        : ON (face contrast normalisation)")
    print(f"Gaze tracker : ON (iris landmark-based)")
    print(f"Display width: {args.display_width}px")
    if user_profile:
        print(f"User profile : '{args.user}' (calibrated)")
    print()

    if args.image:
        run_image(model, args.image, args.save_landmarks, use_tta, user_profile=user_profile)
    elif args.video:
        run_video(model, args.video, args.window_ms,
                  args.save_landmarks, args.keep_session, use_tta, user_profile=user_profile)
    elif args.webcam:
        run_webcam(model, args.window_ms,
                   args.save_landmarks, args.keep_session, use_tta, user_profile=user_profile)