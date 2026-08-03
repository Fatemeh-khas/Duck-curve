"""24-hour time-series runner over the IEEE 33-bus network.

Given a `Scenario` describing PV placement, BESS placement, and BESS dispatch,
this module produces the system-level net-load curve and per-bus voltage
trajectories. The net-load curve is the primary input to the duck-curve
objective and to Figure 4.

Decision D-009: when `include_losses=True`, an analytic DistFlow loss term is
folded into the net-load curve so that PV placement affects the duck-curve
metric (PV deeper in the feeder lowers upstream line currents and thus losses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data import (
    IEEE33_BUSES,
    residential_24h_profile,
    solar_24h_profile,
)
from ..data.ieee33 import IEEE33_LINES, Line
from .network import build_network, run_power_flow


@dataclass
class Scenario:
    """Decoded decision variables for one 24-h scenario."""
    pv_buses: List[int]
    pv_unit_capacity_mw: float
    bess_buses: List[int]
    bess_power_mw: np.ndarray
    bess_energy_capacity_mwh: np.ndarray
    bess_init_soc_mwh: np.ndarray
    eta_c: float = 0.95
    eta_d: float = 0.95


@dataclass
class TimeSeriesResult:
    net_load_kw: np.ndarray
    total_load_kw: np.ndarray
    total_pv_kw: np.ndarray
    total_bess_kw: np.ndarray
    soc_mwh: np.ndarray
    bus_voltage_pu: Optional[np.ndarray] = None
    losses_kw: Optional[np.ndarray] = None
    constraint_violations: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topology pre-computation.
# ---------------------------------------------------------------------------

def _downstream_buses() -> Dict[int, frozenset]:
    children: Dict[int, List[int]] = {}
    for ln in IEEE33_LINES:
        children.setdefault(ln.from_bus, []).append(ln.to_bus)

    def descendants(root: int) -> set:
        out = {root}
        stack = [root]
        while stack:
            n = stack.pop()
            for c in children.get(n, []):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    return {i: frozenset(descendants(ln.to_bus)) for i, ln in enumerate(IEEE33_LINES)}


def _path_to_slack() -> Dict[int, Tuple[int, ...]]:
    """For each bus, the tuple of line indices on the path from slack (bus 1)."""
    parent_line: Dict[int, int] = {}
    for i, ln in enumerate(IEEE33_LINES):
        parent_line[ln.to_bus] = i
    paths: Dict[int, Tuple[int, ...]] = {}
    for b in IEEE33_BUSES:
        if b.idx == 1:
            paths[b.idx] = ()
            continue
        chain: List[int] = []
        cur = b.idx
        while cur != 1:
            li = parent_line.get(cur)
            if li is None:
                break
            chain.append(li)
            cur = IEEE33_LINES[li].from_bus
        paths[b.idx] = tuple(chain)
    return paths


_DOWNSTREAM = _downstream_buses()
_BUS_IDX = {b.idx: i for i, b in enumerate(IEEE33_BUSES)}
_PATH_TO_SLACK = _path_to_slack()
_LINE_R_OHM = np.array([ln.r_ohm for ln in IEEE33_LINES])
_LINE_X_OHM = np.array([ln.x_ohm for ln in IEEE33_LINES])


def _analytic_line_losses(
    bus_p_kw: np.ndarray,
    bus_q_kvar: np.ndarray,
    base_kv: float = 12.66,
) -> np.ndarray:
    """Per-timestep total line losses (kW) using the DistFlow approximation
    P_line ~ sum_downstream P, Q_line ~ sum_downstream Q, loss = (P^2+Q^2)R/V^2.
    """
    T = bus_p_kw.shape[0]
    losses = np.zeros(T)
    for ln_idx, ln in enumerate(IEEE33_LINES):
        ds = _DOWNSTREAM[ln_idx]
        ds_positions = np.array([_BUS_IDX[b] for b in ds if b in _BUS_IDX])
        if ds_positions.size == 0:
            continue
        P_line_kw = bus_p_kw[:, ds_positions].sum(axis=1)
        Q_line_kvar = bus_q_kvar[:, ds_positions].sum(axis=1)
        losses += (P_line_kw ** 2 + Q_line_kvar ** 2) * ln.r_ohm / (base_kv ** 2 * 1000.0)
    return losses


def _analytic_bus_voltages_pu(
    bus_p_kw: np.ndarray,
    bus_q_kvar: np.ndarray,
    base_kv: float = 12.66,
) -> np.ndarray:
    """Per-timestep per-bus voltage magnitude (pu) via cumulative DistFlow drop.

    dV_pu_line = (P_kw * R + Q_kvar * X) / (V_base_kv^2 * 1000)
    v_pu_bus   = 1 - sum_{lines on path slack->bus} dV_pu_line

    Reverse flow gives dV_pu < 0 -> voltage rise, so both 0.95 (under-voltage)
    and 1.05 (over-voltage) excursions are captured.
    """
    T = bus_p_kw.shape[0]
    n_buses = bus_p_kw.shape[1]
    n_lines = len(IEEE33_LINES)
    P_line = np.zeros((T, n_lines))
    Q_line = np.zeros((T, n_lines))
    for ln_idx in range(n_lines):
        ds = _DOWNSTREAM[ln_idx]
        ds_pos = np.array([_BUS_IDX[b] for b in ds if b in _BUS_IDX])
        if ds_pos.size == 0:
            continue
        P_line[:, ln_idx] = bus_p_kw[:, ds_pos].sum(axis=1)
        Q_line[:, ln_idx] = bus_q_kvar[:, ds_pos].sum(axis=1)
    drop_pu = (P_line * _LINE_R_OHM[None, :]
               + Q_line * _LINE_X_OHM[None, :]) / (base_kv ** 2 * 1000.0)
    v_pu = np.ones((T, n_buses))
    for b in IEEE33_BUSES:
        i = _BUS_IDX[b.idx]
        path = _PATH_TO_SLACK[b.idx]
        if not path:
            continue
        v_pu[:, i] = 1.0 - drop_pu[:, list(path)].sum(axis=1)
    return v_pu


def run_timeseries(
    scenario: Scenario,
    run_pf: bool = False,
    include_losses: bool = True,
    base_kv: float = 12.66,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    per_unit_power_mw: float | None = None,
    pv_profile: np.ndarray | None = None,
) -> TimeSeriesResult:
    """Simulate the 24-h response and report constraint violations.

    `run_pf=True` runs full BFS / pandapower power flow per timestep (slow).
    `include_losses=True` (default) folds analytic DistFlow losses into the
    net-load curve (decision D-009).

    Voltage band [v_min_pu, v_max_pu] defaults to 0.95-1.05 pu (ANSI C84.1).
    An analytic per-bus voltage trajectory is ALWAYS computed (cheap) and
    used to populate `constraint_violations["voltage"]`. The optimiser sees
    voltage-band excursions without paying for a real power flow.

    Cycle closure: `constraint_violations["soc_neutrality"]` = sum_k |SOC[k,T]
    - SOC[k,0]|, penalised very strongly in ezoa_run so initial == final SOC
    is effectively a hard equality constraint per BESS unit.
    """
    T = 24
    load_profile = residential_24h_profile()
    pv_profile = solar_24h_profile() if pv_profile is None else np.asarray(pv_profile, dtype=float)
    if pv_profile.shape != (T,) or not np.all(np.isfinite(pv_profile)):
        raise ValueError("pv_profile must contain 24 finite hourly values")
    if np.any((pv_profile < 0.0) | (pv_profile > 1.0)):
        raise ValueError("pv_profile values must lie in [0, 1]")

    bus_p_nominal = np.array([b.p_kw for b in IEEE33_BUSES])
    bus_q_nominal = np.array([b.q_kvar for b in IEEE33_BUSES])

    bus_p_kw_t = bus_p_nominal[None, :] * load_profile[:, None]
    bus_q_kvar_t = bus_q_nominal[None, :] * load_profile[:, None]

    pv_per_unit_kw = scenario.pv_unit_capacity_mw * 1000.0
    for bus in scenario.pv_buses:
        if bus in _BUS_IDX:
            bus_p_kw_t[:, _BUS_IDX[bus]] -= pv_per_unit_kw * pv_profile

    n_bess = len(scenario.bess_buses)
    bess_power = scenario.bess_power_mw
    for k, bus in enumerate(scenario.bess_buses):
        if bus in _BUS_IDX:
            bus_p_kw_t[:, _BUS_IDX[bus]] -= bess_power[k] * 1000.0

    total_load_kw = bus_p_nominal.sum() * load_profile
    total_pv_kw = len(scenario.pv_buses) * pv_per_unit_kw * pv_profile
    total_bess_kw = bess_power.sum(axis=0) * 1000.0

    losses_kw_arr = _analytic_line_losses(bus_p_kw_t, bus_q_kvar_t, base_kv=base_kv)

    net_load_kw = total_load_kw - total_pv_kw - total_bess_kw
    if include_losses:
        net_load_kw = net_load_kw + losses_kw_arr

    soc = np.zeros((n_bess, T + 1))
    soc[:, 0] = scenario.bess_init_soc_mwh
    for t in range(T):
        for k in range(n_bess):
            p = bess_power[k, t]
            if p >= 0:
                soc[k, t + 1] = soc[k, t] - (p / scenario.eta_d) * 1.0
            else:
                soc[k, t + 1] = soc[k, t] + (-p * scenario.eta_c) * 1.0

    e_caps = scenario.bess_energy_capacity_mwh
    soc_min = 0.20 * e_caps[:, None]
    soc_max = 0.90 * e_caps[:, None]
    if per_unit_power_mw is not None:
        p_max_per_unit = per_unit_power_mw
    else:
        p_max_per_unit = float(e_caps.mean()) / 2.0

    v_pu_analytic = _analytic_bus_voltages_pu(bus_p_kw_t, bus_q_kvar_t, base_kv=base_kv)
    voltage_violation_pu = float(
        (np.maximum(0.0, v_pu_analytic - v_max_pu)
         + np.maximum(0.0, v_min_pu - v_pu_analytic)).sum()
    )

    violations = {
        "soc_low":         float(np.maximum(0.0, soc_min - soc).sum()),
        "soc_high":        float(np.maximum(0.0, soc - soc_max).sum()),
        "soc_neutrality":  float(np.abs(soc[:, -1] - soc[:, 0]).sum()),
        "power_limit":     float(np.maximum(0.0, np.abs(bess_power) - p_max_per_unit).sum()),
        "voltage":         voltage_violation_pu,
    }

    bus_v = v_pu_analytic
    pf_losses = None
    if run_pf:
        net = build_network()
        bus_v = np.zeros((T, len(IEEE33_BUSES)))
        pf_losses = np.zeros(T)
        for t in range(T):
            p_kw = {b.idx: float(bus_p_kw_t[t, _BUS_IDX[b.idx]]) for b in IEEE33_BUSES}
            q_kvar = {b.idx: float(bus_q_kvar_t[t, _BUS_IDX[b.idx]]) for b in IEEE33_BUSES}
            res = run_power_flow(net, p_kw, q_kvar)
            bus_v[t] = res.bus_voltages_pu
            pf_losses[t] = res.losses_kw

    return TimeSeriesResult(
        net_load_kw=net_load_kw,
        total_load_kw=total_load_kw,
        total_pv_kw=total_pv_kw,
        total_bess_kw=total_bess_kw,
        soc_mwh=soc,
        bus_voltage_pu=bus_v,
        losses_kw=losses_kw_arr if pf_losses is None else pf_losses,
        constraint_violations=violations,
    )
