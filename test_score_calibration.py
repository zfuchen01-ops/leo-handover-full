import unittest

from score_calibration import aggregate_errors, candidate_score, passes_guardrails


class CalibrationScoreTests(unittest.TestCase):
    def test_aggregate_errors_uses_primary_methods_only(self):
        rows = [
            {"method": "MGCS", "metric": "delay", "rel_error": "0.10"},
            {"method": "CAHS", "metric": "delay", "rel_error": "0.06"},
            {"method": "RS", "metric": "delay", "rel_error": "0.80"},
        ]

        summary = aggregate_errors(rows)

        self.assertAlmostEqual(summary["delay"], 0.08)

    def test_candidate_score_uses_requested_weights(self):
        summary = {"delay": 0.08, "throughput": 0.04, "handover": 0.10}

        self.assertAlmostEqual(candidate_score(summary), 0.072)

    def test_guardrail_rejects_throughput_worsening_over_five_points(self):
        baseline = {"throughput": 0.05, "handover": 0.08}
        candidate = {"throughput": 0.101, "handover": 0.08}

        self.assertFalse(passes_guardrails(candidate, baseline))
        self.assertTrue(passes_guardrails({"throughput": 0.10, "handover": 0.13}, baseline))


if __name__ == "__main__":
    unittest.main()
