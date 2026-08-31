import unittest

from inference import weighted_fusion


class FusionWeightingTest(unittest.TestCase):
    def test_confidence_adjusts_modality_weights(self):
        result = weighted_fusion(
            score_keystroke=0.85,
            score_text=0.35,
            confidence_map={
                "keystroke": 0.95,
                "text": 0.35,
            },
        )

        self.assertGreater(result["weights_used"]["keystroke"], result["weights_used"]["text"])
        self.assertIn("keystroke", result["used_modalities"])
        self.assertIn("text", result["used_modalities"])


if __name__ == "__main__":
    unittest.main()
