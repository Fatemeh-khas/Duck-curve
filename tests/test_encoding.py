import unittest

import numpy as np

from duckcurve.optimizer.encoding import (
    DecisionSpec,
    decode,
    encode_bounds,
)


class EncodingTests(unittest.TestCase):
    def test_decode_enforces_exact_daily_soc_closure(self):
        spec = DecisionSpec()
        rng = np.random.default_rng(991)
        x = rng.uniform(-1.0, 1.0, size=spec.dim)
        x[: spec.n_pv + spec.n_bess] = [6, 18, 33, 9, 15, 32]
        scenario = decode(x, spec)
        for dispatch in scenario.bess_power_mw:
            change = np.where(
                dispatch < 0.0,
                (-dispatch) * spec.eta_c,
                -dispatch / spec.eta_d,
            )
            self.assertAlmostEqual(float(np.sum(change)), 0.0, places=12)

    def test_initial_soc_is_an_explicit_bounded_decision(self):
        spec = DecisionSpec()
        lower, upper = encode_bounds(spec)
        soc_slice = slice(spec.n_pv + spec.n_bess, spec.dispatch_offset)
        np.testing.assert_allclose(lower[soc_slice], spec.soc_min)
        np.testing.assert_allclose(upper[soc_slice], spec.soc_max)

        x = 0.5 * (lower + upper)
        low = x.copy()
        high = x.copy()
        low[soc_slice] = spec.soc_min
        high[soc_slice] = spec.soc_max
        np.testing.assert_allclose(decode(low, spec).bess_init_soc_mwh, spec.soc_min)
        np.testing.assert_allclose(decode(high, spec).bess_init_soc_mwh, spec.soc_max)

    def test_dispatch_is_projected_inside_soc_window(self):
        spec = DecisionSpec()
        lower, upper = encode_bounds(spec)
        rng = np.random.default_rng(77)
        x = rng.uniform(lower, upper)
        scenario = decode(x, spec)
        for initial, dispatch in zip(
            scenario.bess_init_soc_mwh, scenario.bess_power_mw
        ):
            change = np.where(
                dispatch < 0.0,
                (-dispatch) * spec.eta_c,
                -dispatch / spec.eta_d,
            )
            soc = initial + np.concatenate(([0.0], np.cumsum(change)))
            self.assertGreaterEqual(float(np.min(soc)), spec.soc_min - 1e-12)
            self.assertLessEqual(float(np.max(soc)), spec.soc_max + 1e-12)


if __name__ == "__main__":
    unittest.main()
