import unittest

from KEYSTROKE.keystroke_features import KeystrokeFeatureExtractor
from KEYSTROKE.keystroke_listener import KeystrokeBuffer, KeystrokeEvent


class KeystrokeFeatureDepthTest(unittest.TestCase):
    def test_extra_keystroke_features_are_extracted(self):
        import time

        buffer = KeystrokeBuffer(user_baseline_wpm=60.0)
        base_time = time.time()

        for i in range(12):
            press_time = base_time + i * 0.30
            release_time = press_time + (0.09 if i % 3 else 0.12)
            buffer.add_event(KeystrokeEvent(key="a", event_type="press", timestamp=press_time, duration=0.0))
            buffer.add_event(KeystrokeEvent(key="a", event_type="release", timestamp=release_time, duration=release_time - press_time))

        for i in range(3):
            press_time = base_time + 12 * 0.30 + i * 0.70
            release_time = press_time + 0.18
            buffer.add_event(KeystrokeEvent(key="\b", event_type="press", timestamp=press_time, duration=0.0))
            buffer.add_event(KeystrokeEvent(key="\b", event_type="release", timestamp=release_time, duration=release_time - press_time))

        features = KeystrokeFeatureExtractor.extract_all_features(buffer, window_seconds=60)

        self.assertGreaterEqual(features["keystroke_count"], 15)
        self.assertIn("latency_std_ms", features)
        self.assertIn("backspace_rate", features)
        self.assertIn("burst_ratio", features)
        self.assertIn("key_variation", features)
        self.assertGreaterEqual(features["burst_ratio"], 0.0)
        self.assertGreaterEqual(features["backspace_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
