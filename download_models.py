"""
download_models.py — run this once to download MediaPipe model files.

Unchanged from before.

Usage:
    python download_models.py
"""

import urllib.request
import os

os.makedirs("models", exist_ok=True)

models = {
    "models/face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "models/pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

for path, url in models.items():
    if os.path.exists(path):
        print(f"Already exists: {path} — skipping")
        continue
    print(f"Downloading {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Done  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nAll done. You can now run:")
print("  python inference.py --image  face.jpg")
print("  python inference.py --webcam --consent")
