"""
Phase 1, steps 1 & 4 — Extract + Metadata Generator.

Each scan_* function walks a raw dataset folder and yields dict rows with
the unified metadata.csv schema:

    audio_path, dataset, emotion, speaker, gender, duration, session

`audio_path` here is the RAW source path; the pipeline script standardizes
the audio afterward and rewrites this field to the processed path.
`emotion` is already mapped to the unified 7-class schema (None if unmapped).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from labels.label_maps import map_label

logger = logging.getLogger("mindflow.scanners")

# CREMA-D actor demographics (from the official VideoDemographics.csv).
# Load lazily so the module doesn't hard-fail if the demo file isn't present.
def _load_crema_demographics(raw_dir: Path) -> dict[str, str]:
    demo_path = raw_dir / "VideoDemographics.csv"
    gender_by_actor = {}
    if demo_path.exists():
        with open(demo_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                actor_id = row.get("ActorID", "").strip()
                sex = row.get("Sex", "").strip().lower()
                gender_by_actor[actor_id] = "male" if sex.startswith("m") else "female"
    else:
        logger.warning("CREMA-D: VideoDemographics.csv not found under %s "
                        "— gender will be left blank", raw_dir)
    return gender_by_actor


def scan_crema_d(raw_dir: Path) -> Iterator[dict]:
    gender_by_actor = _load_crema_demographics(raw_dir)
    for wav_path in raw_dir.rglob("*.wav"):
        parts = wav_path.stem.split("_")
        if len(parts) < 4:
            logger.warning("CREMA-D: unexpected filename %s", wav_path.name)
            continue
        actor_id, _sentence, emo_code, _level = parts[0], parts[1], parts[2], parts[3]
        emotion = map_label("CREMA-D", emo_code)
        yield {
            "audio_path": str(wav_path),
            "dataset": "CREMA-D",
            "emotion": emotion,
            "speaker": actor_id,
            "gender": gender_by_actor.get(actor_id, ""),
            "duration": None,  # filled in during standardization pass
            "session": "",
        }


def scan_ravdess(raw_dir: Path) -> Iterator[dict]:
    # RAVDESS convention: odd actor numbers = male, even = female
    for wav_path in raw_dir.rglob("*.wav"):
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            logger.warning("RAVDESS: unexpected filename %s", wav_path.name)
            continue
        _modality, _channel, emo_code, _intensity, _stmt, _rep, actor = parts
        emotion = map_label("RAVDESS", emo_code)
        actor_num = int(actor)
        gender = "male" if actor_num % 2 == 1 else "female"
        yield {
            "audio_path": str(wav_path),
            "dataset": "RAVDESS",
            "emotion": emotion,
            "speaker": actor,
            "gender": gender,
            "duration": None,
            "session": "",
        }


def scan_savee(raw_dir: Path) -> Iterator[dict]:
    """
    SAVEE encodes the speaker as a FOLDER name (AudioData/DC/a01.wav, ...).
    Only scan raw_dir/AudioData/{speaker}/*.wav directly -- some SAVEE
    downloads contain a duplicate nested AudioData/AudioData/{speaker}/
    tree from a bad extraction; explicitly skip it so files aren't
    counted twice.
    """
    audio_root = raw_dir / "AudioData"
    if not audio_root.exists():
        audio_root = raw_dir

    for speaker_dir in sorted(p for p in audio_root.iterdir()
                               if p.is_dir() and p.name != "AudioData"):
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            stem = wav_path.stem
            speaker = speaker_dir.name
            emo_code = "".join(ch for ch in stem if not ch.isdigit())

            emotion = map_label("SAVEE", emo_code)
            if emotion is None:
                logger.warning("SAVEE: unrecognized emotion code \'%s\' in %s", emo_code, wav_path.name)
                continue

            yield {
                "audio_path": str(wav_path),
                "dataset": "SAVEE",
                "emotion": emotion,
                "speaker": speaker,
                "gender": "male",
                "duration": None,
                "session": "",
            }


def scan_tess(raw_dir: Path) -> Iterator[dict]:
    # Folder pattern: {OAF|YAF}_{emotion}/  filenames: {OAF|YAF}_{word}_{emotion}.wav
    for wav_path in raw_dir.rglob("*.wav"):
        stem_parts = wav_path.stem.split("_")
        if len(stem_parts) < 3:
            logger.warning("TESS: unexpected filename %s", wav_path.name)
            continue
        speaker_tag = stem_parts[0]  # OAF or YAF
        emo_tag = "_".join(stem_parts[2:]).lower()  # handles "pleasant_surprise"
        emotion = map_label("TESS", emo_tag)
        yield {
            "audio_path": str(wav_path),
            "dataset": "TESS",
            "emotion": emotion,
            "speaker": speaker_tag,   # only 2 actresses total (OAF, YAF)
            "gender": "female",
            "duration": None,
            "session": "",
        }


def scan_meld(raw_dir: Path) -> Iterator[dict]:
    """
    MELD ships train/dev/test *_sent_emo.csv files alongside per-utterance
    video clips (dia{d}_utt{u}.mp4). Audio must be extracted from video
    first (see scripts/extract_meld_audio.py) — this scanner expects that
    extraction has already produced .wav files under raw_dir/audio/{split}/.

    Confirmed against the actual downloaded MELD.Raw release — CSV location
    is genuinely inconsistent between splits (a known quirk of the official
    download, not a local extraction issue):
        train_sent_emo.csv  -> raw_dir/train/train_sent_emo.csv
        dev_sent_emo.csv    -> raw_dir/dev_sent_emo.csv   (at the root)
        test_sent_emo.csv   -> raw_dir/test_sent_emo.csv  (at the root)
    """
    csv_candidates = {
        "train": [raw_dir / "train" / "train_sent_emo.csv", raw_dir / "train_sent_emo.csv"],
        "dev": [raw_dir / "dev_sent_emo.csv", raw_dir / "dev" / "dev_sent_emo.csv"],
        "test": [raw_dir / "test_sent_emo.csv", raw_dir / "test" / "test_sent_emo.csv"],
    }

    for split, candidates in csv_candidates.items():
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            logger.warning("MELD: no CSV found for split '%s' (checked %s), skipping",
                            split, candidates)
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dia_id = row.get("Dialogue_ID")
                utt_id = row.get("Utterance_ID")
                speaker = row.get("Speaker", "")
                emo_native = row.get("Emotion", "")
                emotion = map_label("MELD", emo_native)

                wav_name = f"dia{dia_id}_utt{utt_id}.wav"
                wav_path = raw_dir / "audio" / split / wav_name

                yield {
                    "audio_path": str(wav_path),
                    "dataset": "MELD",
                    "emotion": emotion,
                    "speaker": speaker,
                    "gender": "",  # not labeled in MELD
                    "duration": None,
                    "session": split,
                }


def scan_daic_woz(raw_dir: Path) -> Iterator[dict]:
    """
    DAIC-WOZ is a clinical interview corpus labeled with PHQ-8 depression
    scores, not the 7-class emotion schema — it feeds the stress-regression
    stage, not the emotion classifier. This scanner emits a parallel
    metadata file (metadata/daic_woz_metadata.csv) with a `phq8_score` /
    `phq8_binary` column instead of `emotion`.

    Confirmed against actual downloaded data — this is the E-DAIC (Extended
    DAIC-WOZ) release, which differs from the original AVEC2017 DAIC-WOZ in
    two ways handled below:

    1. Audio layout is nested one level deeper than the original release:
        {participant_id}_P/{participant_id}_P/{participant_id}_AUDIO.wav

    2. Labels live in a separate `labels/` subfolder as
       train_split.csv / dev_split.csv / test_split.csv, with E-DAIC's own
       column names (not AVEC2017's PHQ8_Score/PHQ8_Binary):
           Participant_ID, Gender ("male"/"female" string), PHQ_Binary,
           PHQ_Score, PCL-C (PTSD), PTSD Severity
       A `detailed_lables.csv` / `Detailed_PHQ8_Labels.csv` may also be
       present with additional subscales — not used here, but easy to pull
       in later if you want PTSD severity as an auxiliary signal.
    """
    labels_dir = raw_dir / "labels" if (raw_dir / "labels").exists() else raw_dir
    label_files = list(labels_dir.glob("*split*.csv"))
    phq8_by_pid: dict[str, dict] = {}
    for lf in label_files:
        with open(lf, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("Participant_ID", "").strip()
                if pid:
                    phq8_by_pid[pid] = row

    if not label_files:
        logger.warning(
            "DAIC-WOZ: no *split*.csv label file found under %s or %s — "
            "PHQ scores will be blank for every row.",
            raw_dir, labels_dir,
        )
    else:
        logger.info("DAIC-WOZ: loaded PHQ labels for %d participants from %s",
                    len(phq8_by_pid), labels_dir)

    for participant_dir in sorted(raw_dir.glob("*_P")):
        pid = participant_dir.name.replace("_P", "")

        # Nested layout confirmed as the actual structure; flat kept as fallback.
        nested_dir = participant_dir / participant_dir.name
        audio_dir = nested_dir if nested_dir.exists() else participant_dir

        audio_path = audio_dir / f"{pid}_AUDIO.wav"
        if not audio_path.exists():
            logger.warning("DAIC-WOZ: missing audio for participant %s (looked in %s)",
                            pid, audio_dir)
            continue

        phq_row = phq8_by_pid.get(pid, {})
        yield {
            "audio_path": str(audio_path),
            "dataset": "DAIC-WOZ",
            "participant_id": pid,
            "phq8_score": phq_row.get("PHQ_Score", ""),
            "phq8_binary": phq_row.get("PHQ_Binary", ""),
            "gender": phq_row.get("Gender", ""),
            "duration": None,
            "session": "",
        }



import re as _re

# IEMOCAP's native emotion codes -> unified 7-class schema.
# 'exc' (excited) and 'fru' (frustration) aren't in the unified schema;
# common practice in SER literature merges exc->happy, fru->angry.
# 'oth'/'xxx' (no agreement / other) are dropped as unmapped.
_IEMOCAP_EMOTION_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "ang": "angry",
    "fru": "angry",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": None,
    "xxx": None,
}

_IEMOCAP_LINE_RE = _re.compile(
    r"^\[\d+\.\d+\s*-\s*\d+\.\d+\]\s+(\S+)\s+(\w+)\s+\["
)


def scan_iemocap(raw_dir: Path) -> Iterator[dict]:
    """
    Parses IEMOCAP's per-dialogue consensus label files:
        SessionX/dialog/EmoEvaluation/{dialogue_name}.txt
    Each matching line looks like:
        [6.2901 - 8.2357]    Ses01F_impro01_F000    neu    [2.5000, 2.5000, 2.5000]
    -> utterance_id="Ses01F_impro01_F000", emotion_code="neu"

    Audio for that utterance lives at:
        SessionX/sentences/wav/{dialogue_name}/{utterance_id}.wav

    Only reads the top-level EmoEvaluation/*.txt files (consensus labels) --
    NOT the Categorical/, Attribute/, Self-evaluation/ subfolders, which hold
    raw per-annotator votes already summarized into the consensus line.
    """
    for session_dir in sorted(raw_dir.glob("Session*")):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        if not eval_dir.exists():
            logger.warning("IEMOCAP: no EmoEvaluation dir for %s", session_dir.name)
            continue

        for txt_path in sorted(eval_dir.glob("*.txt")):
            dialogue_name = txt_path.stem
            wav_dir = session_dir / "sentences" / "wav" / dialogue_name

            with open(txt_path, newline="", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = _IEMOCAP_LINE_RE.match(line)
                    if not match:
                        continue
                    utterance_id, emo_code = match.group(1), match.group(2)
                    emotion = _IEMOCAP_EMOTION_MAP.get(emo_code)
                    if emotion is None:
                        continue

                    wav_path = wav_dir / f"{utterance_id}.wav"
                    if not wav_path.exists():
                        logger.warning("IEMOCAP: missing audio for %s (looked in %s)",
                                        utterance_id, wav_path)
                        continue

                    speaker_tag = dialogue_name.split("_")[0]
                    gender = "female" if utterance_id.split("_")[-1].startswith("F") else "male"

                    yield {
                        "audio_path": str(wav_path),
                        "dataset": "IEMOCAP",
                        "emotion": emotion,
                        "speaker": speaker_tag,
                        "gender": gender,
                        "duration": None,
                        "session": session_dir.name,
                    }


SCANNERS = {
    "CREMA-D": scan_crema_d,
    "RAVDESS": scan_ravdess,
    "SAVEE": scan_savee,
    "TESS": scan_tess,
    "MELD": scan_meld,
    "DAIC-WOZ": scan_daic_woz,
    "IEMOCAP": scan_iemocap,
}
