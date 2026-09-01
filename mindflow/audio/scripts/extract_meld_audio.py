"""
MELD ships train/dev/test as .mp4 video clips (dia{d}_utt{u}.mp4), one per
utterance, alongside train_sent_emo.csv / dev_sent_emo.csv / test_sent_emo.csv.
metadata/dataset_scanners.py::scan_meld expects those already converted to
.wav files under {MELD_ROOT}/audio/{split}/dia{d}_utt{u}.wav — this script
does that conversion.

Requires ffmpeg on PATH. On Windows:
    winget install ffmpeg
  or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its
  bin/ folder to PATH, then restart your terminal.

Usage:
    python scripts/extract_meld_audio.py
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIRS, TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mindflow.extract_meld_audio")

# Confirmed against the actual downloaded MELD.Raw release.
SPLIT_VIDEO_DIRS = {
    "train": ["train/train_splits"],
    "dev": ["dev/dev_splits_complete"],
    "test": ["test/output_repeated_splits_test"],
}


def find_clip_dir(meld_root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        candidate = meld_root / name
        if candidate.exists():
            return candidate
    return None


def extract_split(meld_root: Path, split: str, video_dir: Path) -> None:
    out_dir = meld_root / "audio" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = list(video_dir.glob("*.mp4"))
    logger.info("%s: found %d mp4 clips in %s", split, len(mp4_files), video_dir)

    n_ok, n_fail, n_skip = 0, 0, 0
    for mp4_path in mp4_files:
        wav_path = out_dir / (mp4_path.stem + ".wav")
        if wav_path.exists():
            n_skip += 1
            continue

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4_path),
                "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
                "-vn",  # no video stream
                str(wav_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning("ffmpeg failed on %s: %s", mp4_path.name, result.stderr[-300:])

    logger.info("%s: %d extracted, %d already existed, %d failed", split, n_ok, n_skip, n_fail)


def main():
    if shutil.which("ffmpeg") is None:
        logger.error(
            "ffmpeg not found on PATH. Install it first:\n"
            "  winget install ffmpeg\n"
            "or download from https://www.gyan.dev/ffmpeg/builds/ and add "
            "its bin/ folder to your PATH, then restart your terminal."
        )
        sys.exit(1)

    meld_root = RAW_DIRS["MELD"]
    if not meld_root.exists():
        logger.error("MELD folder not found at %s — check config/settings.py", meld_root)
        sys.exit(1)

    for split, candidates in SPLIT_VIDEO_DIRS.items():
        video_dir = find_clip_dir(meld_root, candidates)
        if video_dir is None:
            logger.warning(
                "Couldn't find a clip folder for split '%s' under %s "
                "(looked for: %s) — skipping. Check your MELD folder's "
                "actual subfolder names and add them to SPLIT_VIDEO_DIRS "
                "in this script if they differ.",
                split, meld_root, candidates,
            )
            continue
        extract_split(meld_root, split, video_dir)


if __name__ == "__main__":
    main()
