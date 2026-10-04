import unittest

from TEXT.text_pipeline import run_text_pipeline


class TextPipelineAccuracyTest(unittest.TestCase):
    def test_stressed_text_raises_stress_score(self):
        result = run_text_pipeline(
            "I am overwhelmed, anxious, and exhausted about deadlines and exams. "
            "I feel stuck and hopeless, and I cannot focus."
        )

        self.assertIn("stress_score", result)
        self.assertGreater(result["stress_score"], 0.6)
        self.assertGreater(result["anxiety_prob"], 0.4)


if __name__ == "__main__":
    unittest.main()
