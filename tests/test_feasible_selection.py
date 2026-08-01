import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from duckcurve.experiments.ezoa_run import _select_feasible_primary
from duckcurve.optimizer.encoding import DecisionSpec, encode_bounds


class FeasibilityFirstSelectionTests(unittest.TestCase):
    def test_lower_objective_infeasible_member_is_rejected(self):
        spec = DecisionSpec()
        lo, hi = encode_bounds(spec)
        feasible_x = (lo + hi) / 2.0
        infeasible_x = feasible_x.copy()
        feasible_x[:3] = [2, 3, 4]
        infeasible_x[:3] = [10, 11, 12]
        feasible_x[3:6] = [5, 6, 7]
        infeasible_x[3:6] = [5, 6, 7]
        X = np.vstack([infeasible_x, feasible_x])
        F = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        def fake_timeseries(scenario, **kwargs):
            voltage = 0.01 if scenario.pv_buses[0] == 10 else 0.0
            return SimpleNamespace(constraint_violations={
                "soc_low": 0.0, "soc_high": 0.0,
                "soc_neutrality": 0.0, "power_limit": 0.0,
                "voltage": voltage,
            })

        with patch(
            "duckcurve.experiments.ezoa_run.run_timeseries",
            side_effect=fake_timeseries,
        ):
            index, feasible_count, fallback = _select_feasible_primary(
                X, F, spec, True, 0.94, 1.05, np.ones((2, 24)) * 0.5
            )
        self.assertEqual(index, 1)
        self.assertEqual(feasible_count, 1)
        self.assertFalse(fallback)

    def test_no_feasible_member_uses_explicit_least_violation_fallback(self):
        spec = DecisionSpec()
        lo, hi = encode_bounds(spec)
        X = np.vstack([(lo + hi) / 2.0, (lo + hi) / 2.0])
        X[0, :3] = [2, 3, 4]
        X[1, :3] = [10, 11, 12]
        X[:, 3:6] = [5, 6, 7]
        F = np.array([[1.0, 1.0], [2.0, 2.0]])

        def fake_timeseries(scenario, **kwargs):
            voltage = 0.02 if scenario.pv_buses[0] == 2 else 0.01
            return SimpleNamespace(constraint_violations={
                "soc_low": 0.0, "soc_high": 0.0,
                "soc_neutrality": 0.0, "power_limit": 0.0,
                "voltage": voltage,
            })

        with patch(
            "duckcurve.experiments.ezoa_run.run_timeseries",
            side_effect=fake_timeseries,
        ):
            index, feasible_count, fallback = _select_feasible_primary(
                X, F, spec, True, 0.94, 1.05, None
            )
        self.assertEqual(index, 1)
        self.assertEqual(feasible_count, 0)
        self.assertTrue(fallback)


if __name__ == "__main__":
    unittest.main()
