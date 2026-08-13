import unittest
import os

import Handover as handover_module
import Network as network_module
import paper_model


class FakeUser:
    def __init__(self, user_id, outgoing=True):
        self.user_ID = user_id
        self.sat_covered = set()
        self.user_to_connect_to = {object(): 0} if outgoing else {}

    def __hash__(self):
        return hash(self.user_ID)

    def __eq__(self, other):
        return isinstance(other, FakeUser) and self.user_ID == other.user_ID


class FakeSatellite:
    def __init__(self):
        self.user_connected = set()


class SameSatelliteInterferenceTest(unittest.TestCase):
    def test_path_delay_weight_adjusts_candidate_score(self):
        self.assertEqual(handover_module.Apply_Path_Delay_Weight(500.0, 80.0, -2.0), 340.0)
        self.assertAlmostEqual(handover_module.Apply_Path_Delay_Weight(0.5, 80.0, 0.1, 100.0), 0.58)

    def test_mgcs_shortlist_prefers_low_delay_within_capacity_tolerance(self):
        best_capacity = object()
        lower_delay = object()
        outside_tolerance = object()
        candidates = [
            (best_capacity, 100.0, 90.0),
            (lower_delay, 96.0, 70.0),
            (outside_tolerance, 80.0, 60.0),
        ]

        self.assertIs(handover_module.Select_MGCS_Candidate(candidates, 0.0), best_capacity)
        self.assertIs(handover_module.Select_MGCS_Candidate(candidates, 0.05), lower_delay)

    def test_source_handover_noise_includes_same_satellite_users(self):
        ho = object.__new__(handover_module.Handover)
        sat = FakeSatellite()
        target = FakeUser(1)
        interferer_a = FakeUser(2)
        interferer_b = FakeUser(3)
        sat.user_connected.update({target, interferer_a, interferer_b})

        def fake_signal_power(_sat, user, _time=-1.0):
            return float(user.user_ID * 10)

        ho.Calc_Signal_Power = fake_signal_power

        expected = handover_module.GAUSE_NOISE * handover_module.C_BAND * 1.0e6 + 20.0 + 30.0
        self.assertAlmostEqual(ho.Calc_Download_Noise(sat, target), expected)

    def test_paper_weights_use_candidate_average_ipq_not_agc_weighting(self):
        f_weight, g_weight = paper_model.paper_utility_weights(
            ipq_values=[0.2, 0.8],
            agc_values=[100.0, 1.0],
            h_weight=0.1,
        )

        self.assertAlmostEqual(g_weight, 0.45)
        self.assertAlmostEqual(f_weight, 0.45)

    def test_paper_rate_cap_limits_gsl_capacity_when_enabled(self):
        original_cap = paper_model.GSL_RATE_CAP
        try:
            paper_model.GSL_RATE_CAP = 500.0
            self.assertEqual(paper_model.cap_gsl_rate(1200.0), 500.0)
            self.assertEqual(paper_model.cap_gsl_rate(320.0), 320.0)
        finally:
            paper_model.GSL_RATE_CAP = original_cap

    def test_load_dependent_rate_cap_switches_at_threshold(self):
        original_cap = paper_model.GSL_RATE_CAP
        original_high_cap = paper_model.GSL_RATE_CAP_HIGH_LOAD
        original_threshold = paper_model.GSL_RATE_CAP_LOAD_THRESHOLD
        try:
            paper_model.GSL_RATE_CAP = 1000.0
            paper_model.GSL_RATE_CAP_HIGH_LOAD = 1400.0
            paper_model.GSL_RATE_CAP_LOAD_THRESHOLD = 600
            self.assertEqual(paper_model.cap_gsl_rate_for_load(1800.0, 350), 1000.0)
            self.assertEqual(paper_model.cap_gsl_rate_for_load(1800.0, 600), 1400.0)
        finally:
            paper_model.GSL_RATE_CAP = original_cap
            paper_model.GSL_RATE_CAP_HIGH_LOAD = original_high_cap
            paper_model.GSL_RATE_CAP_LOAD_THRESHOLD = original_threshold

    def test_rate_cap_points_interpolate_by_user_count(self):
        self.assertEqual(
            paper_model.interpolated_load_value(100, "100:500,350:1000,600:2200"),
            500.0,
        )
        self.assertEqual(
            paper_model.interpolated_load_value(600, "100:500,350:1000,600:2200"),
            2200.0,
        )
        self.assertAlmostEqual(
            paper_model.interpolated_load_value(225, "100:500,350:1000,600:2200"),
            750.0,
        )

    def test_load_dependent_isl_bandwidth_switches_at_threshold(self):
        original = {
            name: os.environ.get(name)
            for name in [
                "LEO_ISL_BANDWIDTH_POINTS",
                "LEO_ISL_BANDWIDTH_HIGH_LOAD",
                "LEO_ISL_BANDWIDTH_LOAD_THRESHOLD",
                "LEO_ACTIVE_USER_COUNT",
            ]
        }
        try:
            os.environ.pop("LEO_ISL_BANDWIDTH_POINTS", None)
            os.environ["LEO_ISL_BANDWIDTH_HIGH_LOAD"] = "10000"
            os.environ["LEO_ISL_BANDWIDTH_LOAD_THRESHOLD"] = "600"
            os.environ["LEO_ACTIVE_USER_COUNT"] = "350"
            self.assertEqual(network_module.Effective_ISL_Bandwidth(), network_module.BANDWIDTH)
            os.environ["LEO_ACTIVE_USER_COUNT"] = "600"
            self.assertEqual(network_module.Effective_ISL_Bandwidth(), 10000.0)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_isl_bandwidth_points_interpolate_by_user_count(self):
        original = {
            name: os.environ.get(name)
            for name in [
                "LEO_ISL_BANDWIDTH_POINTS",
                "LEO_ACTIVE_USER_COUNT",
            ]
        }
        try:
            os.environ["LEO_ISL_BANDWIDTH_POINTS"] = "100:3000,600:8000"
            os.environ["LEO_ACTIVE_USER_COUNT"] = "100"
            self.assertEqual(network_module.Effective_ISL_Bandwidth(), 3000.0)
            os.environ["LEO_ACTIVE_USER_COUNT"] = "350"
            self.assertEqual(network_module.Effective_ISL_Bandwidth(), 5500.0)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_paper_link_sending_quality_uses_configured_free_ratio_exponent(self):
        original_alpha = paper_model.SQ_FREE_ALPHA
        try:
            paper_model.SQ_FREE_ALPHA = 0.5
            self.assertAlmostEqual(paper_model.paper_link_sending_quality(0.25), 0.5)
            self.assertEqual(paper_model.paper_link_sending_quality(2.0), 1.0)
            self.assertEqual(paper_model.paper_link_sending_quality(-1.0), 0.0)
        finally:
            paper_model.SQ_FREE_ALPHA = original_alpha

    def test_utility_hysteresis_keeps_current_when_gain_is_small(self):
        original_hysteresis = paper_model.UTILITY_HYSTERESIS
        original_points = paper_model.UTILITY_HYSTERESIS_POINTS
        current = object()
        target = object()
        try:
            paper_model.UTILITY_HYSTERESIS = 0.05
            paper_model.UTILITY_HYSTERESIS_POINTS = ""
            self.assertIs(
                paper_model.apply_utility_hysteresis(current, target, 1.00, 1.03),
                current,
            )
            self.assertIs(
                paper_model.apply_utility_hysteresis(current, target, 1.00, 1.06),
                target,
            )
        finally:
            paper_model.UTILITY_HYSTERESIS = original_hysteresis
            paper_model.UTILITY_HYSTERESIS_POINTS = original_points

    def test_utility_hysteresis_points_override_scalar_by_user_count(self):
        original_hysteresis = paper_model.UTILITY_HYSTERESIS
        original_points = paper_model.UTILITY_HYSTERESIS_POINTS
        current = object()
        target = object()
        try:
            paper_model.UTILITY_HYSTERESIS = 0.0
            paper_model.UTILITY_HYSTERESIS_POINTS = "100:0.02,600:0.20"
            self.assertIs(
                paper_model.apply_utility_hysteresis(current, target, 1.00, 1.10, 100),
                target,
            )
            self.assertIs(
                paper_model.apply_utility_hysteresis(current, target, 1.00, 1.10, 600),
                current,
            )
        finally:
            paper_model.UTILITY_HYSTERESIS = original_hysteresis
            paper_model.UTILITY_HYSTERESIS_POINTS = original_points

    def test_utility_h_points_override_scalar_by_user_count(self):
        original_h = paper_model.UTILITY_H
        original_points = paper_model.UTILITY_H_POINTS
        try:
            paper_model.UTILITY_H = 0.1
            paper_model.UTILITY_H_POINTS = "100:0.05,600:0.15"
            self.assertEqual(paper_model.effective_utility_h(100), 0.05)
            self.assertEqual(paper_model.effective_utility_h(600), 0.15)
            self.assertAlmostEqual(paper_model.effective_utility_h(350), 0.1)
        finally:
            paper_model.UTILITY_H = original_h
            paper_model.UTILITY_H_POINTS = original_points

    def test_source_handover_noise_excludes_ground_backbone_endpoints(self):
        ho = object.__new__(handover_module.Handover)
        sat = FakeSatellite()
        target = FakeUser(1)
        source_interferer = FakeUser(2)
        ground_endpoint = FakeUser(9, outgoing=False)
        sat.user_connected.update({target, source_interferer, ground_endpoint})

        def fake_signal_power(_sat, user, _time=-1.0):
            return float(user.user_ID * 10)

        ho.Calc_Signal_Power = fake_signal_power

        expected = handover_module.GAUSE_NOISE * handover_module.C_BAND * 1.0e6 + 20.0
        self.assertAlmostEqual(ho.Calc_Download_Noise(sat, target), expected)

    def test_paper_rebuild_noise_includes_same_satellite_users(self):
        original_scale = paper_model.INTERFERENCE_SCALE
        ho = object.__new__(paper_model.PaperRebuildHandover)
        sat = FakeSatellite()
        target = FakeUser(1)
        interferer = FakeUser(4)
        sat.user_connected.update({target, interferer})

        def fake_signal_power(_sat, user, _time=-1.0):
            return float(user.user_ID)

        ho.Calc_Signal_Power = fake_signal_power

        try:
            paper_model.INTERFERENCE_SCALE = 1.0
            expected = paper_model.NOISE_W_PER_HZ * paper_model.C_BAND * 1.0e6 + 4.0
            self.assertAlmostEqual(ho.Calc_Download_Noise(sat, target), expected)
        finally:
            paper_model.INTERFERENCE_SCALE = original_scale

    def test_paper_rebuild_noise_applies_interference_scale(self):
        original_scale = paper_model.INTERFERENCE_SCALE
        ho = object.__new__(paper_model.PaperRebuildHandover)
        sat = FakeSatellite()
        target = FakeUser(1)
        interferer = FakeUser(4)
        sat.user_connected.update({target, interferer})

        def fake_signal_power(_sat, user, _time=-1.0):
            return float(user.user_ID)

        ho.Calc_Signal_Power = fake_signal_power

        try:
            paper_model.INTERFERENCE_SCALE = 0.25
            expected = paper_model.NOISE_W_PER_HZ * paper_model.C_BAND * 1.0e6 + 1.0
            self.assertAlmostEqual(ho.Calc_Download_Noise(sat, target), expected)
        finally:
            paper_model.INTERFERENCE_SCALE = original_scale

    def test_average_channel_quality_decision_averages_future_rates(self):
        original_mode = handover_module.CHANNEL_QUALITY_DECISION_NOISE
        original_window = paper_model.CHANNEL_QUALITY_AVG_WINDOW
        original_samples = paper_model.CHANNEL_QUALITY_AVG_SAMPLES
        ho = object.__new__(paper_model.PaperRebuildHandover)
        ho.topo = type("Topo", (), {"current_time": 10, "user": [object(), object()]})()

        def fake_rate(time, _sat, _user):
            return time

        ho.Calc_Trans_Rate = fake_rate
        try:
            handover_module.CHANNEL_QUALITY_DECISION_NOISE = "average"
            paper_model.CHANNEL_QUALITY_AVG_WINDOW = 20
            paper_model.CHANNEL_QUALITY_AVG_SAMPLES = 3
            self.assertEqual(ho.Calc_Channel_Quality_Decision(object(), object()), 20.0)
        finally:
            handover_module.CHANNEL_QUALITY_DECISION_NOISE = original_mode
            paper_model.CHANNEL_QUALITY_AVG_WINDOW = original_window
            paper_model.CHANNEL_QUALITY_AVG_SAMPLES = original_samples


if __name__ == "__main__":
    unittest.main()
