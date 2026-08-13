import unittest

from focused_handover_optimization import a_mgcs_candidates, rs_candidates


class FocusedHandoverOptimizationTests(unittest.TestCase):
    def test_rs_candidates_cover_physical_destination_rules(self):
        envs = [candidate.env for candidate in rs_candidates("A")]

        self.assertIn({"LEO_DESTINATION_DECISION_MODE": "DISTANCE"}, envs)
        self.assertIn({"LEO_DESTINATION_DECISION_MODE": "ELEVATION"}, envs)

    def test_a_mgcs_candidates_keep_winner_and_add_fine_delay_weights(self):
        candidates = a_mgcs_candidates()

        self.assertTrue(all(candidate.env["LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS"] == "4" for candidate in candidates))
        self.assertTrue(any(candidate.env.get("LEO_MGCS_DELAY_WEIGHT") == "-0.05" for candidate in candidates))
        self.assertTrue(any(candidate.env.get("LEO_MGCS_DELAY_WEIGHT") == "-0.75" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
