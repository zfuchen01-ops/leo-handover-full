import unittest

from overnight_handover_screen import (
    Candidate,
    candidate_score,
    config_key,
    generate_mgcs_stage1,
    generate_rs_stage1,
    merge_final_rows,
    passes_guardrails,
    stage_timeout_seconds,
)


class OvernightHandoverScreenTests(unittest.TestCase):
    def test_score_uses_handover_heavier_weights(self):
        errors = {"throughput_mean": 0.10, "delay_mean": 0.20, "handover_mean": 0.30}

        self.assertAlmostEqual(candidate_score(errors), 0.21)

    def test_guardrails_reject_three_point_metric_worsening_and_double_max(self):
        baseline = {
            "throughput_mean": 0.10,
            "delay_mean": 0.10,
            "throughput_max": 0.20,
            "delay_max": 0.20,
            "handover_max": 0.20,
        }

        self.assertTrue(passes_guardrails({**baseline, "throughput_mean": 0.13}, baseline))
        self.assertFalse(passes_guardrails({**baseline, "throughput_mean": 0.131}, baseline))
        self.assertFalse(passes_guardrails({**baseline, "handover_max": 0.401}, baseline))

    def test_config_key_is_stable_for_environment_order(self):
        first = Candidate("RS", "A", "x", {"B": "2", "A": "1"})
        second = Candidate("RS", "A", "x", {"A": "1", "B": "2"})

        self.assertEqual(config_key(first, [100, 350, 600], 80), config_key(second, [100, 350, 600], 80))

    def test_rs_stage1_covers_destination_modes_service_times_and_reset(self):
        candidates = generate_rs_stage1("A")
        envs = [candidate.env for candidate in candidates]

        self.assertTrue(any(env.get("LEO_DESTINATION_DECISION_MODE") == "CHANNEL_QUALITY" for env in envs))
        self.assertTrue(any(env.get("LEO_DESTINATION_MIN_SERVICE_TIME") == "120" for env in envs))
        self.assertTrue(any(env.get("LEO_DELAY_HYSTERESIS") == "0.10" for env in envs))
        self.assertTrue(any(env.get("LEO_RESET_HANDOVER_AFTER_INITIAL") == "0" for env in envs))

    def test_mgcs_stage1_covers_hold_average_and_interference(self):
        candidates = generate_mgcs_stage1("B")
        envs = [candidate.env for candidate in candidates]

        self.assertTrue(any(env.get("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS") == "8" for env in envs))
        self.assertTrue(any(env.get("LEO_CHANNEL_QUALITY_DECISION_NOISE") == "average" for env in envs))
        self.assertTrue(any(env.get("LEO_PAPER_INTERFERENCE_SCALE") == "0.10" for env in envs))
        self.assertTrue(any("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS" in env for env in envs))

    def test_stage_timeout_prevents_one_candidate_consuming_the_budget(self):
        self.assertEqual(stage_timeout_seconds("A_RS_stage1_key80"), 15 * 60)
        self.assertEqual(stage_timeout_seconds("A_MGCS_top5_full80"), 30 * 60)
        self.assertEqual(stage_timeout_seconds("B_MGCS_top2_key400"), 90 * 60)
        self.assertEqual(stage_timeout_seconds("B_RS_winner_full400"), 90 * 60)

    def test_continuation_replaces_same_group_but_preserves_other_groups(self):
        existing = [
            {"constellation": "A", "method": "RS", "candidate": "old"},
            {"constellation": "B", "method": "RS", "candidate": "keep"},
        ]
        new = [{"constellation": "A", "method": "RS", "candidate": "new"}]

        merged = merge_final_rows(existing, new)

        self.assertEqual([(row["constellation"], row["method"], row["candidate"]) for row in merged], [
            ("A", "RS", "new"),
            ("B", "RS", "keep"),
        ])


if __name__ == "__main__":
    unittest.main()
