import unittest

import numpy as np

from duckcurve.data.paper_reliability import paper_analytic_base_indices
from duckcurve.grid.timeseries import Scenario
from duckcurve.objectives.resilience_mcs import (
    evaluate_mcs_case,
    generate_mcs_scenario_bank,
    paired_mcs_effect,
    summarize_mcs_case,
)


def _no_der_scenario():
    return Scenario(
        pv_buses=[],
        pv_unit_capacity_mw=0.0,
        bess_buses=[],
        bess_power_mw=np.zeros((0, 24)),
        bess_energy_capacity_mwh=np.zeros(0),
        bess_init_soc_mwh=np.zeros(0),
    )


class PaperReliabilityTests(unittest.TestCase):
    def test_service_recovery_metrics_are_bounded_and_summarized(self):
        bank = generate_mcs_scenario_bank(
            50, seed=31, duration_distribution="deterministic"
        )
        rows = evaluate_mcs_case(
            "no_der",
            _no_der_scenario(),
            bank,
            bess_power_limit_mw=0.0,
            critical_service_threshold=0.95,
        )
        for row in rows:
            self.assertGreaterEqual(row["service_recovery_time_hours"], 0.0)
            self.assertLessEqual(
                row["service_recovery_time_hours"], row["outage_duration_hours"]
            )
            self.assertGreaterEqual(row["critical_load_served_percent"], 0.0)
            self.assertLessEqual(row["critical_load_served_percent"], 100.0)
        summary = summarize_mcs_case(rows)
        self.assertIn("mean_service_recovery_time_hours", summary)
        self.assertIn("cvar95_service_recovery_time_hours", summary)
        self.assertIn("critical_service_compliance_probability", summary)

    def test_paper_table_reproduction_is_explicit(self):
        audit = paper_analytic_base_indices()
        self.assertAlmostEqual(
            audit["calculated_saifi_interruptions_per_customer_year"],
            1.5342261103633918,
        )
        self.assertAlmostEqual(
            audit["calculated_aens_kwh_per_customer_year"],
            28.806325706594887,
        )
        self.assertNotAlmostEqual(
            audit["calculated_aens_kwh_per_customer_year"],
            audit["published_aens_kwh_per_customer_year"],
            places=2,
        )

    def test_no_der_mcs_matches_direct_aens_with_deterministic_repairs(self):
        bank = generate_mcs_scenario_bank(
            scenario_count=200_000,
            seed=1947,
            duration_distribution="deterministic",
        )
        rows = evaluate_mcs_case(
            "no_der",
            _no_der_scenario(),
            bank,
            bess_power_limit_mw=0.0,
        )
        result = summarize_mcs_case(rows)
        direct = paper_analytic_base_indices()[
            "calculated_aens_kwh_per_customer_year"
        ]
        self.assertAlmostEqual(
            result["annual_aens_kwh_per_customer_year"], direct, delta=0.15
        )

    def test_scenario_bank_and_paired_effect_are_reproducible(self):
        first = generate_mcs_scenario_bank(100, seed=22)
        second = generate_mcs_scenario_bank(100, seed=22)
        self.assertEqual(first, second)
        rows = evaluate_mcs_case(
            "reference", _no_der_scenario(), first, bess_power_limit_mw=0.0
        )
        effect = paired_mcs_effect(rows, rows)
        self.assertFalse(effect["supports_improvement_at_95pct"])
        self.assertIn("does not support", effect["claim"])


if __name__ == "__main__":
    unittest.main()
