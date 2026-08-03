from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..grid.timeseries import Scenario

_T = 24


@dataclass
class DecisionSpec:
    n_pv: int = 3
    n_bess: int = 3
    horizon: int = _T
    pv_unit_capacity_mw: float = 1.0
    bess_total_power_mw: float = 3.0
    bess_total_energy_mwh: float = 9.0
    eta_c: float = 0.95
    eta_d: float = 0.95

    @property
    def dim(self) -> int:
        # PV buses, BESS buses, initial SOC for each BESS, then hourly dispatch.
        return self.n_pv + 2 * self.n_bess + self.n_bess * self.horizon

    @property
    def dispatch_offset(self) -> int:
        return self.n_pv + 2 * self.n_bess

    @property
    def per_unit_power_mw(self) -> float:
        return self.bess_total_power_mw / self.n_bess

    @property
    def per_unit_energy_mwh(self) -> float:
        return self.bess_total_energy_mwh / self.n_bess

    @property
    def soc_min(self) -> float:
        return self.per_unit_energy_mwh * 0.20

    @property
    def soc_max(self) -> float:
        return self.per_unit_energy_mwh * 0.90


def _cumulative_soc_change(dispatch_mw: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    e = np.zeros(spec.horizon + 1, dtype=float)
    for t in range(spec.horizon):
        p = float(dispatch_mw[t])
        e[t + 1] = e[t] + ((-p) * spec.eta_c if p < 0 else -p / spec.eta_d)
    return e


def _project_exact_cycle(dispatch_mw: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """Project one daily schedule to exact efficiency-adjusted SOC closure.

    Only the larger energy side is scaled down, preserving signs and all power
    limits. The resulting cumulative SOC change is zero to floating precision.
    """
    dispatch = np.clip(
        np.asarray(dispatch_mw, dtype=float),
        -spec.per_unit_power_mw,
        spec.per_unit_power_mw,
    ).copy()
    charged = float(
        np.sum(-dispatch[dispatch < 0.0]) * spec.eta_c
    )
    discharged = float(
        np.sum(dispatch[dispatch > 0.0]) / spec.eta_d
    )
    if charged > discharged and charged > 0.0:
        dispatch[dispatch < 0.0] *= discharged / charged
    elif discharged > charged and discharged > 0.0:
        dispatch[dispatch > 0.0] *= charged / discharged
    return dispatch


def _project_to_soc_window(
    dispatch_mw: np.ndarray, initial_soc_mwh: float, spec: DecisionSpec
) -> np.ndarray:
    """Scale a cycle-closed schedule to fit around the chosen initial SOC."""
    dispatch = _project_exact_cycle(dispatch_mw, spec)
    energy = _cumulative_soc_change(dispatch, spec)
    scale = 1.0
    positive = float(np.max(energy))
    negative = float(np.min(energy))
    if positive > 0.0:
        scale = min(scale, (spec.soc_max - initial_soc_mwh) / positive)
    if negative < 0.0:
        scale = min(scale, (initial_soc_mwh - spec.soc_min) / (-negative))
    return dispatch * float(np.clip(scale, 0.0, 1.0))


def _analytical_soc0(dispatch_mw: np.ndarray, spec: DecisionSpec) -> Tuple[float, np.ndarray]:
    disp = _project_exact_cycle(dispatch_mw, spec)
    soc_min = spec.soc_min
    soc_max = spec.soc_max

    energy = _cumulative_soc_change(disp, spec)
    lo = soc_min - float(energy.min())
    hi = soc_max - float(energy.max())

    if lo > hi + 1e-9:
        span = float(energy.max() - energy.min())
        usable = soc_max - soc_min
        scale = usable / (span + 1e-12)
        disp *= min(scale, 1.0)
        energy = _cumulative_soc_change(disp, spec)
        lo = soc_min - float(energy.min())
        hi = soc_max - float(energy.max())

    soc0 = float(np.clip((lo + hi) / 2.0, soc_min, soc_max))
    return soc0, disp


def encode_bounds(spec: DecisionSpec) -> Tuple[np.ndarray, np.ndarray]:
    n_pv, n_bess, T = spec.n_pv, spec.n_bess, spec.horizon
    p_max = spec.per_unit_power_mw
    lo = np.concatenate(
        [
            np.full(n_pv, 2.0),
            np.full(n_bess, 2.0),
            np.full(n_bess, spec.soc_min),
            np.full(n_bess * T, -p_max),
        ]
    )
    hi = np.concatenate(
        [
            np.full(n_pv, 33.0),
            np.full(n_bess, 33.0),
            np.full(n_bess, spec.soc_max),
            np.full(n_bess * T, +p_max),
        ]
    )
    return lo, hi


def decode(x: np.ndarray, spec: DecisionSpec) -> Scenario:
    n_pv, n_bess, T = spec.n_pv, spec.n_bess, spec.horizon
    e_max = spec.per_unit_energy_mwh
    p_max = spec.per_unit_power_mw

    off = 0
    pv_buses = np.clip(np.round(x[off : off + n_pv]).astype(int), 2, 33).tolist()
    off += n_pv

    bess_buses = np.clip(np.round(x[off : off + n_bess]).astype(int), 2, 33).tolist()
    off += n_bess
    init_soc = np.clip(
        np.asarray(x[off : off + n_bess], dtype=float),
        spec.soc_min,
        spec.soc_max,
    )
    off += n_bess

    bess_power = np.zeros((n_bess, T), dtype=float)

    for k in range(n_bess):
        raw = np.clip(np.asarray(x[off : off + T], dtype=float), -p_max, p_max)
        bess_power[k] = _project_to_soc_window(raw, float(init_soc[k]), spec)
        off += T

    return Scenario(
        pv_buses=pv_buses,
        pv_unit_capacity_mw=spec.pv_unit_capacity_mw,
        bess_buses=bess_buses,
        bess_power_mw=bess_power,
        bess_energy_capacity_mwh=np.full(n_bess, e_max),
        bess_init_soc_mwh=init_soc,
        eta_c=spec.eta_c,
        eta_d=spec.eta_d,
    )


def decode_with_info(x: np.ndarray, spec: DecisionSpec):
    scn = decode(x, spec)
    n_bess, T = spec.n_bess, spec.horizon
    e_max = spec.per_unit_energy_mwh

    units = []
    for k in range(n_bess):
        soc0 = float(scn.bess_init_soc_mwh[k])
        disp = np.asarray(scn.bess_power_mw[k], dtype=float)
        soc = soc0
        soc_peak = soc0

        for t in range(T):
            p = float(disp[t])
            soc += (-p) * spec.eta_c if p < 0 else -p / spec.eta_d
            soc = float(np.clip(soc, spec.soc_min, spec.soc_max))
            soc_peak = max(soc_peak, soc)

        soc_final = soc
        charge_energy = sum((-float(disp[t])) * spec.eta_c for t in range(T) if disp[t] < 0)
        discharge_energy = sum(float(disp[t]) / spec.eta_d for t in range(T) if disp[t] >= 0)

        charge_hours = [t for t in range(T) if disp[t] < -0.02]
        discharge_hours = [t for t in range(T) if disp[t] > 0.02]

        units.append(
            {
                "unit": k + 1,
                "bus": scn.bess_buses[k],
                "soc0_mwh": round(soc0, 4),
                "soc0_pct": round(soc0 / e_max * 100, 1),
                "soc_peak_mwh": round(soc_peak, 4),
                "soc_peak_pct": round(soc_peak / e_max * 100, 1),
                "soc_final_mwh": round(soc_final, 4),
                "soc_final_pct": round(soc_final / e_max * 100, 1),
                "charge_mwh": round(charge_energy, 4),
                "discharge_mwh": round(discharge_energy, 4),
                "charge_hours": len(charge_hours),
                "discharge_hours": len(discharge_hours),
                "charge_window": f"h{min(charge_hours):02d}-h{max(charge_hours):02d}" if charge_hours else "none",
                "disch_window": f"h{min(discharge_hours):02d}-h{max(discharge_hours):02d}" if discharge_hours else "none",
            }
        )

    return scn, {"units": units, "pv_buses": scn.pv_buses, "bess_buses": scn.bess_buses}


def time_biased_initial_sample(spec: DecisionSpec, strength: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pv, n_bess, T = spec.n_pv, spec.n_bess, spec.horizon
    p_max = spec.per_unit_power_mw
    x = np.zeros(spec.dim, dtype=float)
    off = 0

    x[off : off + n_pv] = rng.uniform(2, 33, size=n_pv)
    off += n_pv
    x[off : off + n_bess] = rng.uniform(2, 33, size=n_bess)
    off += n_bess
    x[off : off + n_bess] = rng.uniform(
        spec.soc_min, spec.soc_max, size=n_bess
    )
    off += n_bess

    for _ in range(n_bess):
        bias = np.zeros(T)
        bias[8:15] = -strength * p_max
        bias[16:21] = +strength * p_max
        noise = rng.normal(0.0, (1.0 - strength) * 0.35 * p_max, size=T)
        x[off : off + T] = np.clip(bias + noise, -p_max, p_max)
        off += T

    return x
