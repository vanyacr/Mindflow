import unittest

from keystroke import KeystrokeTracker


class UserBaselineDeviationTest(unittest.TestCase):
    def test_personal_baseline_detection(self):
        tracker = KeystrokeTracker(auto_load_baseline=False)

        baseline_means = {
            "velocity": 60.0,
            "dwell_mean_ms": 100.0,
            "dwell_std_ms": 25.0,
            "latency_mean_ms": 150.0,
            "latency_std_ms": 40.0,
            "pause_freq": 2.0,
            "error_count": 1.5,
            "backspace_rate": 1.0,
            "burst_ratio": 0.45,
            "key_variation": 0.7,
        }
        baseline_stds = {
            "velocity": 12.0,
            "dwell_mean_ms": 15.0,
            "dwell_std_ms": 7.0,
            "latency_mean_ms": 30.0,
            "latency_std_ms": 10.0,
            "pause_freq": 0.8,
            "error_count": 0.8,
            "backspace_rate": 0.6,
            "burst_ratio": 0.1,
            "key_variation": 0.15,
        }
        tracker.set_baseline_stats(baseline_means, baseline_stds)

        normal_features = {
            "velocity": 62.0,
            "dwell_mean_ms": 101.0,
            "dwell_std_ms": 26.0,
            "latency_mean_ms": 155.0,
            "latency_std_ms": 39.0,
            "pause_freq": 2.1,
            "error_count": 1.7,
            "backspace_rate": 1.1,
            "burst_ratio": 0.44,
            "key_variation": 0.72,
        }
        stressed_features = {
            "velocity": 95.0,
            "dwell_mean_ms": 170.0,
            "dwell_std_ms": 50.0,
            "latency_mean_ms": 260.0,
            "latency_std_ms": 80.0,
            "pause_freq": 5.0,
            "error_count": 6.0,
            "backspace_rate": 4.0,
            "burst_ratio": 0.8,
            "key_variation": 0.9,
        }

        normal_result = tracker.compare_to_baseline(normal_features)
        stressed_result = tracker.compare_to_baseline(stressed_features)

        self.assertLess(normal_result["overall_score"], 0.35)
        self.assertGreater(stressed_result["overall_score"], 0.7)
        self.assertEqual(normal_result["status"], "normal")
        self.assertEqual(stressed_result["status"], "stressed")


if __name__ == "__main__":
    unittest.main()
