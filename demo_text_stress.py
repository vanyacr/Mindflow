"""Simple interactive demo for the text stress pipeline."""

from __future__ import annotations

from TEXT.text_pipeline import run_text_pipeline


def print_result(text: str) -> None:
    result = run_text_pipeline(text)
    print("-" * 70)
    print(f"Input: {text}")
    print(f"Sentiment: {result['sentiment_polarity']} ({result['sentiment_score']})")
    print(f"Stress score: {result['stress_score']}")
    print(f"Anxiety probability: {result['anxiety_prob']}")
    print(f"Emotional tone: {result['emotional_tone']}")
    print(f"Motivation level: {result['motivation_level']}")
    print(f"Emotion breakdown: {result['all_emotions']}")
    print("-" * 70)


def main() -> None:
    print("Mindflow Text Stress Demo")
    print("Type a sentence and press Enter. Type 'q' to quit.\n")

    while True:
        try:
            user_input = input("Your text: ").strip()
        except EOFError:
            print("\nNo more input. Exiting.")
            break

        if not user_input:
            print("Please enter some text.")
            continue

        if user_input.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        print_result(user_input)


if __name__ == "__main__":
    main()
