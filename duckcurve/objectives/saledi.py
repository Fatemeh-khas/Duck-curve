from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from ..data.ieee33 import IEEE33_BUSES, IEEE33_LINES


def _downstream_buses_per_line() -> Dict[int, Set[int]]:
    children: Dict[int, List[int]] = {}
    for ln in IEEE33_LINES:
        children.setdefault(ln.from_bus, []).append(ln.to_bus)

    def descendants(root: int) -> Set[int]:
        out = {root}
        stack = [root]
        while stack:
            n = stack.pop()
            for c in children.get(n, []):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    return {i: descendants(ln.to_bus) for i, ln in enumerate(IEEE33_LINES)}


_DOWNSTREAM = _downstream_buses_per_line()
_BUS_IDX_TO_POS = {b.idx: i for i, b in enumerate(IEEE33_BUSES)}
_BUS_NOMINAL_PKW = {b.idx: b.p_kw for b in IEEE33_BUSES}


def saledi_metric(
    net_load_kw: np.ndarray,
    pv_buses: List[int],
    pv_unit_capacity_mw: float,
    pv_profile: np.ndarray,
    bess_buses: List[int],
    bess_power_mw: np.ndarray,
    load_profile: np.ndarray,
    outage_duration_hours: float = 4.0,
) -> float:
    T = len(load_profile)

    load_kw_t = np.stack(
        [
            np.array([_BUS_NOMINAL_PKW[b.idx] * load_profile[t] for b in IEEE33_BUSES], dtype=float)
            for t in range(T)
        ],
        axis=0,
    )

    pv_kw_t = np.zeros_like(load_kw_t)
    pv_per_unit_kw = float(pv_unit_capacity_mw) * 1000.0
    pv_profile = np.asarray(pv_profile, dtype=float)
    for bus in pv_buses:
        if bus in _BUS_IDX_TO_POS:
            pv_kw_t[:, _BUS_IDX_TO_POS[bus]] += pv_per_unit_kw * pv_profile

    bess_kw_t = np.zeros_like(load_kw_t)
    bess_power_mw = np.asarray(bess_power_mw, dtype=float)
    for k, bus in enumerate(bess_buses):
        if bus in _BUS_IDX_TO_POS and k < bess_power_mw.shape[0]:
            bess_kw_t[:, _BUS_IDX_TO_POS[bus]] += bess_power_mw[k] * 1000.0

    total = 0.0
    duration_weight = float(outage_duration_hours) / 24.0

    for _, ds in _DOWNSTREAM.items():
        ds_positions = [_BUS_IDX_TO_POS[b] for b in ds if b in _BUS_IDX_TO_POS]
        if not ds_positions:
            continue

        for t in range(T):
            island_load = load_kw_t[t, ds_positions].sum()
            island_supply = pv_kw_t[t, ds_positions].sum() + bess_kw_t[t, ds_positions].sum()
            unserved = max(0.0, island_load - island_supply)
            total += np.log1p(unserved) * duration_weight

    return float(total)

def resilience_indices(
    pv_buses: List[int],
    pv_unit_capacity_mw: float,
    pv_profile: np.ndarray,
    load_profile: np.ndarray,
    bess_buses: List[int],
    bess_power_mw: np.ndarray,
    bess_energy_capacity_mwh: np.ndarray,
    bess_init_soc_mwh: np.ndarray,
    eta_c: float = 0.95,
    eta_d: float = 0.95,
    bess_power_limit_mw: float = 1.0,
    outage_duration_hours: int = 4,
    enable_bess: bool = True,
) -> dict:
    """Evaluate four-hour downstream-islanding resilience over all line outages.

    For every IEEE-33 line and every possible outage start hour, the downstream
    buses form an island. Local PV serves load first. BESS units in that island
    then serve the remaining deficit subject to their power rating and the
    deliverable energy above 20% SOC at the outage start. PV-surplus charging
    during the outage is omitted, making the assessment conservative.
    """
    load_profile = np.asarray(load_profile, dtype=float)
    pv_profile = np.asarray(pv_profile, dtype=float)
    horizon = len(load_profile)
    duration = max(1, int(outage_duration_hours))

    load_kw_t = np.stack([
        np.array([_BUS_NOMINAL_PKW[b.idx] * load_profile[t] for b in IEEE33_BUSES], dtype=float)
        for t in range(horizon)
    ])
    pv_kw_t = np.zeros_like(load_kw_t)
    for bus in pv_buses:
        if bus in _BUS_IDX_TO_POS:
            pv_kw_t[:, _BUS_IDX_TO_POS[bus]] += float(pv_unit_capacity_mw) * 1000.0 * pv_profile

    bess_power_mw = np.asarray(bess_power_mw, dtype=float)
    capacities = np.asarray(bess_energy_capacity_mwh, dtype=float)
    init_soc = np.asarray(bess_init_soc_mwh, dtype=float)
    n_bess = len(bess_buses) if enable_bess else 0
    soc = np.zeros((n_bess, horizon + 1), dtype=float)
    if n_bess:
        soc[:, 0] = init_soc[:n_bess]
        for t in range(horizon):
            p = bess_power_mw[:n_bess, t]
            soc[:, t + 1] = soc[:, t] + np.where(p < 0.0, (-p) * eta_c, -p / eta_d)

    line_eens = np.zeros(len(IEEE33_LINES), dtype=float)
    all_unserved = []
    all_demand = []
    all_line_numbers = []
    all_start_hours = []
    for line_idx, downstream in _DOWNSTREAM.items():
        positions = [_BUS_IDX_TO_POS[b] for b in downstream if b in _BUS_IDX_TO_POS]
        local_units = [k for k, bus in enumerate(bess_buses[:n_bess]) if bus in downstream]
        scenario_unserved = []
        for start in range(horizon):
            available_kwh = float(sum(
                max(0.0, soc[k, start] - 0.20 * capacities[k]) * eta_d * 1000.0
                for k in local_units
            ))
            power_limit_kw = len(local_units) * float(bess_power_limit_mw) * 1000.0
            unserved_kwh = 0.0
            demand_kwh = 0.0
            for offset in range(duration):
                t = (start + offset) % horizon
                demand = float(load_kw_t[t, positions].sum())
                local_pv = float(pv_kw_t[t, positions].sum())
                deficit = max(0.0, demand - local_pv)
                bess_supply = min(deficit, power_limit_kw, available_kwh)
                available_kwh -= bess_supply
                unserved_kwh += deficit - bess_supply
                demand_kwh += demand
            scenario_unserved.append(unserved_kwh)
            all_unserved.append(unserved_kwh)
            all_demand.append(demand_kwh)
            all_line_numbers.append(line_idx + 1)
            all_start_hours.append(start)
        line_eens[line_idx] = float(np.mean(scenario_unserved))

    unserved = np.asarray(all_unserved, dtype=float)
    demand = np.asarray(all_demand, dtype=float)
    served_fraction = np.where(demand > 0.0, 1.0 - unserved / demand, 1.0)
    return {
        "eens_kwh": float(unserved.mean()),
        "worst_case_ens_kwh": float(unserved.max()),
        "load_served_percent": float(100.0 * (1.0 - unserved.sum() / demand.sum())),
        "resilience_index": float(np.mean(np.clip(served_fraction, 0.0, 1.0))),
        "line_eens_kwh": line_eens,
        "scenario_unserved_kwh": unserved,
        "scenario_demand_kwh": demand,
        "scenario_load_served_percent": 100.0 * np.clip(served_fraction, 0.0, 1.0),
        "scenario_line_number": np.asarray(all_line_numbers, dtype=int),
        "scenario_start_hour": np.asarray(all_start_hours, dtype=int),
        "outage_scenarios": int(len(unserved)),
        "outage_duration_hours": duration,
    }