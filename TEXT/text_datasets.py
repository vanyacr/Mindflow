"""Download text datasets required for the TEXT analysis module.

Targets:
- GoEmotions
- DepressionEmo (tries multiple likely dataset IDs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import DatasetDict, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "text_datasets"
GOEMOTIONS_DATASET_ID = "go_emotions"
DEPRESSION_EMO_CANDIDATES = (
    "DepressionEmo",
    "depression_emo",
    "suraj520/DepressionEmo",
    "shahules786/DepressionEmo",
    "Amod/mental_health_counseling_conversations",
    "emotion",
)


def _save_dataset(dataset: DatasetDict, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir))


def _try_load_candidates(dataset_ids: Iterable[str]) -> tuple[str, DatasetDict]:
    last_error: Exception | None = None
    for dataset_id in dataset_ids:
        try:
            loaded = load_dataset(dataset_id)
            return dataset_id, loaded
        except Exception as exc:  # noqa: BLE001 - keep trying other dataset IDs
            last_error = exc

    raise RuntimeError(
        "Could not download a DepressionEmo dataset. "
        "Update DEPRESSION_EMO_CANDIDATES with your exact dataset ID."
    ) from last_error


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading GoEmotions...")
    go_emotions = load_dataset(GOEMOTIONS_DATASET_ID)
    _save_dataset(go_emotions, DATA_DIR / "go_emotions")
    print(f"Saved GoEmotions to: {DATA_DIR / 'go_emotions'}")

    print("Downloading DepressionEmo candidate dataset...")
    chosen_id, depression_emo = _try_load_candidates(DEPRESSION_EMO_CANDIDATES)
    _save_dataset(depression_emo, DATA_DIR / "depression_emo")
    print(f"Saved DepressionEmo candidate '{chosen_id}' to: {DATA_DIR / 'depression_emo'}")


if __name__ == "__main__":
    main()
