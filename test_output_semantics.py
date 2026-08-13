import unittest

import make_report
import plot_results
from Handover import delay_components_ms


class OutputSemanticsTests(unittest.TestCase):
    def test_plot_groups_paper_methods_in_fixed_order(self):
        grouped = {
            "CAHS": [{"users": "100"}],
            "RS": [{"users": "100"}],
            "MGCS": [{"users": "100"}],
            "MSTS": [{"users": "100"}],
        }

        self.assertEqual(
            [method for method, _ in plot_results.ordered_groups(grouped)],
            ["RS", "MSTS", "MGCS", "CAHS"],
        )

    def test_plot_converts_throughput_mhz_to_bps(self):
        self.assertEqual(plot_results.metric_value("avg_allocated_mhz", "12.5"), 12_500_000.0)
        self.assertEqual(plot_results.metric_value("avg_delay_ms", "12.5"), 12.5)

    def test_report_key_points_labels_throughput_as_bps(self):
        table = make_report.key_points_table("Example", make_report.Path("missing.csv"))
        self.assertIn("Throughput (bps)", table)

    def test_delay_components_convert_distance_to_per_flow_milliseconds(self):
        components = delay_components_ms(300_000.0, 600_000.0, 900_000.0, 2, 1_200_000.0)

        self.assertEqual(components["source_gsl_ms"], 0.5)
        self.assertEqual(components["isl_ms"], 1.0)
        self.assertEqual(components["destination_gsl_ms"], 1.5)
        self.assertEqual(components["total_ms"], 3.0)
        self.assertEqual(components["legacy_hop_isl_ms"], 2.0)
        self.assertEqual(components["legacy_hop_total_ms"], 4.0)


if __name__ == "__main__":
    unittest.main()
