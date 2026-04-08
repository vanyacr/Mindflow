"""Simple test script for the Stage 5 TEXT pipeline."""

from pathlib import Path
import re
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "Lib" / "site-packages"

if _VENV_SITE_PACKAGES.exists():
    site_packages_path = str(_VENV_SITE_PACKAGES)
    if site_packages_path not in sys.path:
        sys.path.insert(0, site_packages_path)

from transformers import AutoTokenizer

from TEXT.text_pipeline import run_text_pipeline


TOKENIZER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SENTENCES = [
    "I feel motivated and ready to study.",
    "I am stressed about exams and deadlines.",
    "Today I feel calm and focused.",
    "I feel tired and emotionally overwhelmed.",
    "I can improve if I stay consistent every day.",
]


def run_tokenization_test() -> None:
    """Tokenize five sample sentences using the DistilBERT tokenizer."""

    tokenizer = None
    try:
        # Avoid hanging on model downloads; prefer local cache.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    except Exception:
        tokenizer = None

    print("\nTransformer Tokenization Test (5 sample sentences):")
    if tokenizer is None:
        print(
            "Tokenizer not found in local cache. "
            "Using a lightweight local tokenizer fallback."
        )

    for index, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        transformer_tokens = (
            tokenizer.tokenize(sentence)
            if tokenizer is not None
            else re.findall(r"[A-Za-z0-9']+", sentence.lower())
        )

        print(f"\nSentence {index}: {sentence}")
        print(f"Transformer token count: {len(transformer_tokens)}")
        print(f"Transformer tokens: {transformer_tokens}")


def capture_text_input() -> str:
    """Capture user text from the console input box."""
    print("Enter your text for analysis (press Enter to use default sample):")
    user_text = input("> ").strip()

    if user_text:
        return user_text

    return (
        "I have too many assignments and I feel nervous, "
        "but I still want to stay focused and improve, "
        "I am feeling depressed and hopeless."
    )


def print_pipeline_result(sample_text: str) -> None:
    """Run the pipeline and print the result for a single statement."""
    result = run_text_pipeline(sample_text)

    print("\nText Pipeline Output:")
    for key, value in result.items():
        print(f"{key}: {value}")


def main() -> None:
    run_tokenization_test()

    while True:
        print("\nEnter a sentence to analyze, or type 'q' to quit:")
        sample_text = input("> ").strip()

        if sample_text.lower() in {"q", "quit", "exit"}:
            print("Exiting text pipeline test.")
            break

        if not sample_text:
            sample_text = capture_text_input()

        print_pipeline_result(sample_text)

        print("\nPress Enter to analyze another sentence, or type 'q' to quit.")
        next_step = input("> ").strip().lower()
        if next_step in {"Q", "quit", "exit"}:
            print("Exiting text pipeline test.")
            break


if __name__ == "__main__":
    main()
