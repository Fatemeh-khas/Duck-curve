"""Baseline scenario — PV at default mid-feeder buses, *no* BESS optimization.

This produces the *red dashed* original-net-load curve in Figure 4.
"""
from __future__ import annotations

import numpy as np

from ..grid.timeseries import Scenario, TimeSeriesResult, run_timeseries
from ..optimizer.encoding import DecisionSpec


# Default PV placement for the baseline — mid-feeder buses, no optimization.
# These are kept for reproducibility; the EZOA "after" scenario chooses its own.
DEFAULT_PV_BUSES = [14, 24, 31]
DEFAULT_BESS_BUSES = [9, 15, 32]


def build_baseline_scenario(spec: DecisionSpec) -> Scenario:
    """Baseline: PV at defaults, BESS at defaults but holding zero dispatch."""
    n_bess, T = spec.n_bess, spec.horizon
    return Scenario(
        pv_buses=list(DEFAULT_PV_BUSES[:spec.n_pv]),
        pv_unit_capacity_mw=spec.pv_unit_capacity_mw,
        bess_buses=list(DEFAULT_BESS_BUSES[:n_bess]),
        bess_power_mw=np.zeros((n_bess, T)),
        bess_energy_capacity_mwh=np.full(n_bess, spec.per_unit_energy_mwh),
        bess_init_soc_mwh=np.full(n_bess, 0.5 * spec.per_unit_energy_mwh),
        eta_c=spec.eta_c,
        eta_d=spec.eta_d,
    )


def baseline_net_load(spec: DecisionSpec) -> TimeSeriesResult:
    return run_timeseries(build_baseline_scenario(spec), per_unit_power_mw=spec.per_unit_power_mw)


def build_no_pv_no_bess_scenario(spec: DecisionSpec) -> Scenario:
    """Raw scenario: no PV, no BESS at all -- the pre-DER feeder load."""
    T = spec.horizon
    return Scenario(
        pv_buses=[],
        pv_unit_capacity_mw=0.0,
        bess_buses=[],
        bess_power_mw=np.zeros((0, T)),
        bess_energy_capacity_mwh=np.zeros(0),
        bess_init_soc_mwh=np.zeros(0),
        eta_c=spec.eta_c,
        eta_d=spec.eta_d,
    )


def baseline_no_pv_no_bess(spec: DecisionSpec) -> TimeSeriesResult:
    """Raw feeder net load with zero PV and zero BESS -- the 'before any DER'
    reference curve used alongside the PV-only baseline in Figure 4."""
    return run_timeseries(build_no_pv_no_bess_scenario(spec), per_unit_power_mw=spec.per_unit_power_mw)
