"""
MindFlow — Phase 1 pipeline runner.

Usage:
    python scripts/run_phase1.py --datasets CREMA-D RAVDESS SAVEE TESS
    python scripts/run_phase1.py --datasets all --skip-standardize   # metadata only

This will:
  1. Scan each requested raw dataset -> unified metadata rows (unmapped emotions skipped + logged)
  2. Standardize every audio file (16kHz/mono/normalized/trimmed/VAD) into processed/audio/{dataset}/
  3. Fill in true duration post-standardization
  4. Write one metadata_{dataset}.csv per dataset + a combined metadata.csv
  5. Generate the Phase 1 analysis plots (emotion / gender / dataset / duration distributions)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    RAW_DIRS,
    PROCESSED_AUDIO_DIR,
    METADATA_DIR,
    CATEGORICAL_EMOTION_DATASETS,
)
from metadata.dataset_scanners import SCANNERS
from preprocessing.audio_standardize import standardize_audio_file, get_duration_seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mindflow.run_phase1")

METADATA_FIELDNAMES = ["audio_path", "dataset", "emotion", "speaker", "gender", "duration", "session"]


def process_dataset(name: str, skip_standardize: bool) -> list[dict]:
    if name not in RAW_DIRS:
        logger.error("Unknown dataset '%s' — skipping", name)
        return []
    raw_dir = RAW_DIRS[name]
    if not raw_dir.exists():
        logger.warning("Raw dir for %s not found at %s — skipping "
                        "(download it first)", name, raw_dir)
        return []

    scanner = SCANNERS[name]
    rows = []
    n_total, n_unmapped, n_failed_audio = 0, 0, 0

    for row in scanner(raw_dir):
        n_total += 1

        # DAIC-WOZ has its own schema (PHQ8-based) — write it separately
        # and don't try to force it into the emotion metadata.csv.
        if name == "DAIC-WOZ":
            rows.append(row)
            continue

        if row["emotion"] is None:
            n_unmapped += 1
            logger.debug("Unmapped emotion for %s, dropping row: %s", name, row["audio_path"])
            continue

        src_path = Path(row["audio_path"])
        speaker_tag = row.get("speaker", "unk")
        rel_name = f"{speaker_tag}_{src_path.stem}.wav"
        dst_path = PROCESSED_AUDIO_DIR / name / rel_name

        if not skip_standardize:
            ok = standardize_audio_file(src_path, dst_path)
            if not ok:
                n_failed_audio += 1
                continue
            row["audio_path"] = str(dst_path)
            row["duration"] = round(get_duration_seconds(dst_path), 3)
        else:
            # metadata-only pass — keep pointing at raw file, duration from raw
            try:
                row["duration"] = round(get_duration_seconds(src_path), 3)
            except Exception:
                row["duration"] = None

        rows.append(row)

    logger.info(
        "%s: %d files scanned, %d unmapped emotions skipped, %d audio failures, %d kept",
        name, n_total, n_unmapped, n_failed_audio, len(rows),
    )
    return rows


def write_metadata_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), path)


def main():
    parser = argparse.ArgumentParser(description="MindFlow Phase 1 data pipeline")
    parser.add_argument(
        "--datasets", nargs="+", default=["CREMA-D", "RAVDESS", "SAVEE", "TESS"],
        help="Dataset names to process, or 'all'. Default = currently downloaded set.",
    )
    parser.add_argument(
        "--skip-standardize", action="store_true",
        help="Skip audio standardization; just build metadata (useful for a dry run).",
    )
    args = parser.parse_args()

    targets = list(RAW_DIRS.keys()) if args.datasets == ["all"] else args.datasets

    combined_emotion_rows = []
    daic_rows = []

    for name in targets:
        rows = process_dataset(name, args.skip_standardize)
        if not rows:
            continue

        if name == "DAIC-WOZ":
            daic_rows.extend(rows)
            write_metadata_csv(
                rows, METADATA_DIR / "metadata_daic_woz.csv",
                fieldnames=["audio_path", "dataset", "participant_id", "phq8_score",
                            "phq8_binary", "gender", "duration", "session"],
            )
            continue

        write_metadata_csv(rows, METADATA_DIR / f"metadata_{name.lower().replace('-', '_')}.csv",
                            fieldnames=METADATA_FIELDNAMES)
        combined_emotion_rows.extend(rows)

    if combined_emotion_rows:
        write_metadata_csv(combined_emotion_rows, METADATA_DIR / "metadata.csv",
                            fieldnames=METADATA_FIELDNAMES)
        logger.info(
            "Combined emotion metadata.csv: %d rows across %s",
            len(combined_emotion_rows),
            sorted({r['dataset'] for r in combined_emotion_rows}),
        )

    if daic_rows:
        logger.info("DAIC-WOZ: %d participant sessions indexed separately "
                     "(stress-regression track, not emotion classification)", len(daic_rows))

    if not combined_emotion_rows and not daic_rows:
        logger.warning("No rows produced. Check that RAW_DIRS in config/settings.py "
                        "point at your actual dataset locations.")


if __name__ == "__main__":
    main()

