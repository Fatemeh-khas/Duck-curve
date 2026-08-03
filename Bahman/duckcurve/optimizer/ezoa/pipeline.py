"""Algorithm-specific optimizer pipeline for EZOA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ...data import residential_24h_profile, solar_24h_profile
from ...grid.timeseries import TimeSeriesResult, run_timeseries
from ...objectives import (
    evaluate_mcs_case,
    generate_critical_restoration_scenario_bank,
    generate_mcs_scenario_bank,
    sum_slope_squared,
    summarize_mcs_case,
)
from ..archive import ParetoArchive
from ..encoding import DecisionSpec, decode, decode_with_info, encode_bounds, time_biased_initial_sample
from .engine import EZOA, OptimizerResult


@dataclass
class OptimizerSeedResult:
    seed: int
    best_duck_x: np.ndarray
    best_duck_scenario_result: TimeSeriesResult
    best_duck_objectives: Tuple[float, ...]
    evaluation_count: int = 0


@dataclass
class OptimizerPipelineResult:
    optimizer_name: str
    spec: DecisionSpec
    optimizer_result: OptimizerResult
    knee_idx: int
    knee_scenario_result: TimeSeriesResult
    knee_objectives: Tuple[float, ...]
    best_duck_idx: int
    best_duck_scenario_result: TimeSeriesResult
    best_duck_objectives: Tuple[float, ...]
    per_seed_results: List[OptimizerSeedResult] = field(default_factory=list)
    feasible_archive_count: int = 0
    selection_is_feasible: bool = False
    selection_used_fallback: bool = False

    @property
    def ezoa(self) -> OptimizerResult:
        return self.optimizer_result


EZOASeedResult = OptimizerSeedResult
EZOAPipelineResult = OptimizerPipelineResult


# Soft-constraint penalty coefficients.  The SOC-neutrality term is deliberately
# strong because each BESS must finish the day at its initial SOC.
_PENALTY = {
    "soc_low": 5.0e5,
    "soc_high": 5.0e5,
    "soc_neutrality": 2.0e6,
    "power_limit": 3.0e5,
    # Voltage is a hard publication constraint. The previous 1e6 coefficient
    # allowed small stochastic voltage violations to trade against SSS. At
    # 1e9, even a 1e-3 aggregate violation dominates normal objective gains,
    # steering the archive toward robust-feasible designs before final gating.
    "voltage": 1.0e9,
}


def _penalty(violations: dict) -> float:
    return float(sum(_PENALTY.get(k, 0.0) * float(v) for k, v in violations.items()))


def _preferred_buses(preferred: Sequence[int], count: int) -> List[int]:
    """Return a deterministic valid bus list of the requested length."""
    if count <= 0:
        return []
    return [int(preferred[i % len(preferred)]) for i in range(count)]


def _load_leveling_initial_sample(spec: DecisionSpec, include_losses: bool = True) -> np.ndarray:
    """Construct an efficiency-aware cyclic dispatch that directly reduces SSS.

    The aggregate BESS follows the PV-adjusted net load around a constant target.
    Bisection chooses that target so charge and discharge energy balance after
    efficiencies. The dispatch is then scaled, if necessary, to fit the usable
    20%-90% SOC window and split equally among all BESS units.
    """
    if spec.horizon != 24:
        raise ValueError("The load-leveling seed currently requires a 24-hour horizon")

    pv_buses = _preferred_buses([6, 18, 33], spec.n_pv)
    bess_buses = _preferred_buses([9, 15, 32], spec.n_bess)
    zero_dispatch = np.zeros(spec.n_bess * spec.horizon, dtype=float)
    zero_soc = np.full(spec.n_bess, 0.5 * (spec.soc_min + spec.soc_max))
    zero_x = np.concatenate((
        np.asarray(pv_buses), np.asarray(bess_buses), zero_soc, zero_dispatch
    ))

    base = run_timeseries(
        decode(zero_x, spec),
        include_losses=include_losses,
        per_unit_power_mw=spec.per_unit_power_mw,
    ).net_load_kw

    if spec.n_bess == 0 or spec.bess_total_power_mw <= 0.0:
        return zero_x.astype(float)

    target_lo = float(base.min())
    target_hi = float(base.max())
    for _ in range(80):
        target = 0.5 * (target_lo + target_hi)
        dispatch = np.clip(
            (base - target) / 1000.0,
            -spec.bess_total_power_mw,
            spec.bess_total_power_mw,
        )
        energy_balance = float(
            np.where(
                dispatch < 0.0,
                (-dispatch) * spec.eta_c,
                -dispatch / spec.eta_d,
            ).sum()
        )
        if energy_balance > 0.0:
            target_hi = target
        else:
            target_lo = target

    target = 0.5 * (target_lo + target_hi)
    dispatch = np.clip(
        (base - target) / 1000.0,
        -spec.bess_total_power_mw,
        spec.bess_total_power_mw,
    )
    energy_step = np.where(
        dispatch < 0.0,
        (-dispatch) * spec.eta_c,
        -dispatch / spec.eta_d,
    )
    energy_path = np.concatenate(([0.0], np.cumsum(energy_step)))
    energy_span = float(energy_path.max() - energy_path.min())
    usable_energy = spec.bess_total_energy_mwh * (0.90 - 0.20)
    if energy_span > usable_energy:
        dispatch *= usable_energy / energy_span

    per_unit_dispatch = np.tile(dispatch / spec.n_bess, (spec.n_bess, 1))
    energy = np.concatenate((
        [0.0],
        np.cumsum(np.where(
            per_unit_dispatch[0] < 0.0,
            (-per_unit_dispatch[0]) * spec.eta_c,
            -per_unit_dispatch[0] / spec.eta_d,
        )),
    ))
    soc0 = 0.5 * (
        spec.soc_min - float(energy.min())
        + spec.soc_max - float(energy.max())
    )
    return np.concatenate(
        (
            np.asarray(pv_buses),
            np.asarray(bess_buses),
            np.full(spec.n_bess, soc0),
            per_unit_dispatch.ravel(),
        )
    ).astype(float)


def _project_cycle_closed_dispatch(dispatch_mw: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """Project aggregate dispatch onto power, cycle, and usable-energy limits."""
    dispatch = np.clip(
        np.asarray(dispatch_mw, dtype=float),
        -spec.bess_total_power_mw,
        spec.bess_total_power_mw,
    ).copy()
    charge = float((-dispatch[dispatch < 0.0]).sum() * spec.eta_c)
    discharge = float(dispatch[dispatch > 0.0].sum() / spec.eta_d)
    if charge > discharge and charge > 0.0:
        dispatch[dispatch < 0.0] *= discharge / charge
    elif discharge > charge and discharge > 0.0:
        dispatch[dispatch > 0.0] *= charge / discharge

    energy_step = np.where(
        dispatch < 0.0,
        (-dispatch) * spec.eta_c,
        -dispatch / spec.eta_d,
    )
    energy_path = np.concatenate(([0.0], np.cumsum(energy_step)))
    energy_span = float(energy_path.max() - energy_path.min())
    usable_energy = spec.bess_total_energy_mwh * (0.90 - 0.20)
    if energy_span > usable_energy:
        dispatch *= usable_energy / energy_span
    return dispatch


def _set_aggregate_dispatch(x: np.ndarray, dispatch_mw: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """Return an encoding with a feasible aggregate schedule shared by BESS units."""
    candidate = np.asarray(x, dtype=float).copy()
    offset = spec.dispatch_offset
    per_unit = np.tile(dispatch_mw / spec.n_bess, (spec.n_bess, 1))
    candidate[offset:] = per_unit.ravel()
    return candidate


def _project_unit_dispatch(dispatch_mw: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """Project one BESS schedule onto its actual individual constraints."""
    dispatch = np.clip(
        np.asarray(dispatch_mw, dtype=float),
        -spec.per_unit_power_mw,
        spec.per_unit_power_mw,
    ).copy()
    charge = float((-dispatch[dispatch < 0.0]).sum() * spec.eta_c)
    discharge = float(dispatch[dispatch > 0.0].sum() / spec.eta_d)
    if charge > discharge and charge > 0.0:
        dispatch[dispatch < 0.0] *= discharge / charge
    elif discharge > charge and discharge > 0.0:
        dispatch[dispatch > 0.0] *= charge / discharge

    energy = np.concatenate((
        [0.0],
        np.cumsum(np.where(
            dispatch < 0.0,
            (-dispatch) * spec.eta_c,
            -dispatch / spec.eta_d,
        )),
    ))
    span = float(energy.max() - energy.min())
    usable = spec.per_unit_energy_mwh * (0.90 - 0.20)
    if span > usable:
        dispatch *= usable / span
    return dispatch


def _optimize_individual_bess_dispatch(
    x: np.ndarray,
    f: np.ndarray,
    spec: DecisionSpec,
    evaluate,
    sweeps: int,
    initial_step_mw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize every BESS schedule independently using only real objectives.

    Each trial changes one unit-hour coordinate and then projects only that
    unit back onto its physical limits and exact daily cycle closure. Distinct
    SOC trajectories can therefore emerge from feeder location and losses,
    but no diversity pattern or participation factor is imposed.
    """
    if spec.n_bess == 0 or sweeps <= 0:
        return np.asarray(x, dtype=float).copy(), np.asarray(f, dtype=float).copy()

    best_x = np.asarray(x, dtype=float).copy()
    best_f = np.asarray(f, dtype=float).copy()
    offset = spec.dispatch_offset
    dispatch = best_x[offset:].reshape(spec.n_bess, spec.horizon).copy()
    step = max(float(initial_step_mw) / spec.n_bess, 1.0e-6)

    for _ in range(sweeps):
        trials = []
        for unit in range(spec.n_bess):
            for hour in range(spec.horizon):
                for direction in (1.0, -1.0):
                    proposal = dispatch.copy()
                    proposal[unit, hour] += direction * step
                    proposal[unit] = _project_unit_dispatch(proposal[unit], spec)
                    candidate = best_x.copy()
                    candidate[offset:] = proposal.ravel()
                    trials.append(candidate)

        trial_x = np.asarray(trials, dtype=float)
        trial_f = np.asarray(evaluate(trial_x), dtype=float)
        idx = int(np.lexsort((trial_f[:, 1], trial_f[:, 0]))[0])
        if tuple(trial_f[idx]) < tuple(best_f):
            best_x = trial_x[idx]
            best_f = trial_f[idx]
            dispatch = best_x[offset:].reshape(spec.n_bess, spec.horizon).copy()
        else:
            step *= 0.5
        if step < 5.0e-4:
            break

    return best_x, best_f


def _optimize_paired_bess_redispatch(
    x: np.ndarray,
    f: np.ndarray,
    spec: DecisionSpec,
    evaluate,
    sweeps: int = 12,
    initial_step_mw: float = 0.16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Redistribute dispatch between BESS pairs using real feeder objectives.

    A trial transfers power from unit A to B at one hour and performs the
    opposite transfer at another hour in the same charge/discharge regime.
    Aggregate BESS power is unchanged at both hours. Because the two moves use
    the same efficiency branch, each unit also retains exact daily energy
    balance. Candidate generation is symmetric and contains no diversity
    target; a transfer is accepted only when evaluated SSS/constraints and
    SALEDI improve lexicographically.
    """
    if spec.n_bess < 2 or sweeps <= 0:
        return np.asarray(x, dtype=float).copy(), np.asarray(f, dtype=float).copy()

    best_x = np.asarray(x, dtype=float).copy()
    best_f = np.asarray(f, dtype=float).copy()
    offset = spec.dispatch_offset
    step = max(float(initial_step_mw), 1.0e-6)

    for _ in range(sweeps):
        dispatch = best_x[offset:].reshape(spec.n_bess, spec.horizon)
        trials = []
        for unit_a in range(spec.n_bess):
            for unit_b in range(unit_a + 1, spec.n_bess):
                for sign_mode in (-1.0, 1.0):
                    hours = [
                        hour
                        for hour in range(spec.horizon)
                        if np.sign(dispatch[unit_a, hour]) == sign_mode
                        and np.sign(dispatch[unit_b, hour]) == sign_mode
                        and abs(dispatch[unit_a, hour]) > 0.08
                        and abs(dispatch[unit_b, hour]) > 0.08
                    ]
                    hour_pairs = [
                        (hour_a, hour_b)
                        for idx, hour_a in enumerate(hours)
                        for hour_b in hours[idx + 1 :]
                    ]
                    hour_pairs = sorted(
                        hour_pairs,
                        key=lambda pair: abs(pair[1] - pair[0]),
                        reverse=True,
                    )[:14]

                    for hour_a, hour_b in hour_pairs:
                        for direction in (step, -step):
                            proposal = dispatch.copy()
                            proposal[unit_a, hour_a] += direction
                            proposal[unit_b, hour_a] -= direction
                            proposal[unit_a, hour_b] -= direction
                            proposal[unit_b, hour_b] += direction
                            changed = (
                                proposal[unit_a, hour_a],
                                proposal[unit_b, hour_a],
                                proposal[unit_a, hour_b],
                                proposal[unit_b, hour_b],
                            )
                            if np.max(np.abs(proposal)) > spec.per_unit_power_mw + 1.0e-10:
                                continue
                            if min(abs(value) for value in changed) < 0.015:
                                continue
                            if any(np.sign(value) != sign_mode for value in changed):
                                continue

                            candidate = best_x.copy()
                            candidate[offset:] = proposal.ravel()
                            decoded = decode(candidate, spec)
                            if np.max(np.abs(decoded.bess_power_mw - proposal)) > 1.0e-7:
                                continue
                            trials.append(candidate)

        if not trials:
            break
        trial_x = np.asarray(trials, dtype=float)
        trial_f = np.asarray(evaluate(trial_x), dtype=float)
        idx = int(np.lexsort((trial_f[:, 1], trial_f[:, 0]))[0])
        if tuple(trial_f[idx]) < tuple(best_f):
            best_x = trial_x[idx]
            best_f = trial_f[idx]
        else:
            step *= 0.55
        if step < 5.0e-3:
            break

    return best_x, best_f


def _polish_feasible_primary(
    x: np.ndarray,
    f: np.ndarray,
    spec: DecisionSpec,
    evaluate,
    dispatch_sweeps: int,
    initial_step_mw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrained derivative-free search for the minimum-SSS Pareto extreme.

    Bus coordinates are searched over every valid IEEE-33 bus. Dispatch moves
    are accepted only after projection onto exact efficiency-aware cycle closure,
    aggregate power bounds, and the 20%-90% usable SOC-energy window. Acceptance
    uses the evaluated objectives only; it contains no target SSS or percentage.
    """
    best_x = np.asarray(x, dtype=float).copy()
    best_f = np.asarray(f, dtype=float).copy()
    n_bus_variables = spec.n_pv + spec.n_bess

    def optimize_buses() -> bool:
        nonlocal best_x, best_f
        changed = False
        for dim in range(n_bus_variables):
            trials = np.tile(best_x, (32, 1))
            trials[:, dim] = np.arange(2, 34, dtype=float)
            trial_f = np.asarray(evaluate(trials), dtype=float)
            group = range(spec.n_pv) if dim < spec.n_pv else range(spec.n_pv, n_bus_variables)
            for row, trial in enumerate(trials):
                buses = np.rint(trial[list(group)]).astype(int)
                if len(set(buses)) < len(buses):
                    trial_f[row] = np.inf
            idx = int(np.lexsort((trial_f[:, 1], trial_f[:, 0]))[0])
            if tuple(trial_f[idx]) < tuple(best_f):
                best_x = trials[idx]
                best_f = trial_f[idx]
                changed = True
        return changed

    for _ in range(3):
        if not optimize_buses():
            break

    if spec.n_bess > 0 and dispatch_sweeps > 0:
        offset = spec.dispatch_offset
        aggregate = best_x[offset:].reshape(spec.n_bess, spec.horizon).sum(axis=0)
        aggregate = _project_cycle_closed_dispatch(aggregate, spec)
        step = max(float(initial_step_mw), 1.0e-6)

        for _ in range(dispatch_sweeps):
            trials = []
            for hour in range(spec.horizon):
                for direction in (1.0, -1.0):
                    proposal = aggregate.copy()
                    proposal[hour] += direction * step
                    proposal = _project_cycle_closed_dispatch(proposal, spec)
                    trials.append(_set_aggregate_dispatch(best_x, proposal, spec))

            trial_x = np.asarray(trials, dtype=float)
            trial_f = np.asarray(evaluate(trial_x), dtype=float)
            idx = int(np.lexsort((trial_f[:, 1], trial_f[:, 0]))[0])
            if tuple(trial_f[idx]) < tuple(best_f):
                best_x = trial_x[idx]
                best_f = trial_f[idx]
                aggregate = best_x[offset:].reshape(spec.n_bess, spec.horizon).sum(axis=0)
            else:
                step *= 0.5
            if step < 1.0e-3:
                break

    best_x, best_f = _optimize_individual_bess_dispatch(
        best_x,
        best_f,
        spec,
        evaluate,
        sweeps=max(8, dispatch_sweeps // 2),
        initial_step_mw=initial_step_mw,
    )
    best_x, best_f = _optimize_paired_bess_redispatch(
        best_x,
        best_f,
        spec,
        evaluate,
        sweeps=max(8, min(12, dispatch_sweeps)),
        initial_step_mw=0.80 * initial_step_mw,
    )
    for _ in range(3):
        if not optimize_buses():
            break

    return best_x, best_f


def _make_evaluate(
    spec: DecisionSpec,
    include_losses: bool = True,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    evening_weight: float = 0.0,
    peak_weight: float = 0.0,
    evening_window: Tuple[int, int] = (16, 20),
    pv_profiles: np.ndarray | None = None,
    pv_cvar_alpha: float = 0.90,
    pv_cvar_weight: float = 0.0,
    restoration_scenario_count: int = 0,
    restoration_seed: int = 32021,
    critical_service_threshold: float = 0.95,
    restoration_cvar_alpha: float = 0.95,
    restoration_cvar_weight: float = 0.25,
):
    """Build the two-objective batch evaluator.

    With the default zero auxiliary weights, objective 1 is exactly the sum of
    squared hourly net-load slopes.  Objective 2 is the SALEDI surrogate.  The
    objectives are returned separately; no weighted-sum aggregation is used.
    """
    load_profile = residential_24h_profile()
    if pv_profiles is None:
        pv_profiles = solar_24h_profile()[None, :]
    pv_profiles = np.asarray(pv_profiles, dtype=float)
    if pv_profiles.ndim != 2 or pv_profiles.shape[1] != 24:
        raise ValueError("pv_profiles must have shape (n_scenarios, 24)")
    if not 0.0 < pv_cvar_alpha < 1.0 or pv_cvar_weight < 0.0:
        raise ValueError("invalid PV CVaR settings")
    if not 0.0 < restoration_cvar_alpha < 1.0:
        raise ValueError("restoration_cvar_alpha must be in (0, 1)")
    restoration_bank = (
        generate_critical_restoration_scenario_bank(
            restoration_scenario_count, restoration_seed,
            duration_distribution="exponential", load_sigma=0.05, pv_sigma=0.10,
        )
        if restoration_scenario_count > 0 else ()
    )
    report_pv_profile = pv_profiles.mean(axis=0)
    t_start, t_end = evening_window

    def evaluate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        evaluate.evaluation_count += int(X.shape[0])
        F = np.zeros((X.shape[0], 3 if restoration_bank else 2), dtype=float)
        for i, x in enumerate(X):
            scn = decode(x, spec)
            results = [
                run_timeseries(
                    scn, include_losses=include_losses, v_min_pu=v_min_pu,
                    v_max_pu=v_max_pu, per_unit_power_mw=spec.per_unit_power_mw,
                    pv_profile=profile,
                )
                for profile in pv_profiles
            ]
            duplicate_placements = (
                len(scn.pv_buses) - len(set(scn.pv_buses))
                + len(scn.bess_buses) - len(set(scn.bess_buses))
            )
            penalties = np.array([_penalty(r.constraint_violations) for r in results])
            sss = np.array([sum_slope_squared(
                r.net_load_kw, evening_weight=evening_weight, peak_weight=peak_weight,
                t_start=t_start, t_end=t_end) for r in results])
            losses = np.array([float(np.sum(r.losses_kw)) for r in results])

            def risk_adjusted(values: np.ndarray) -> float:
                q = np.quantile(values, pv_cvar_alpha)
                tail = values[values >= q]
                cvar = float(tail.mean()) if tail.size else float(q)
                return float(values.mean() + pv_cvar_weight * cvar)

            pen = float(penalties.max()) + 1.0e9 * duplicate_placements
            F[i, 0] = risk_adjusted(sss) + pen
            F[i, 1] = risk_adjusted(losses) + pen
            if restoration_bank:
                recovery_rows = evaluate_mcs_case(
                    "optimizer_candidate", scn, restoration_bank,
                    bess_power_limit_mw=spec.per_unit_power_mw,
                    load_profile=load_profile, pv_profile=report_pv_profile,
                    critical_service_threshold=critical_service_threshold,
                )
                summary = summarize_mcs_case(recovery_rows)
                mean_rt = float(summary["mean_service_recovery_time_hours"])
                cvar_rt = float(summary["cvar95_service_recovery_time_hours"])
                F[i, 2] = (
                    mean_rt + restoration_cvar_weight * cvar_rt
                    + pen / 1.0e9
                )
        return F

    evaluate.evaluation_count = 0
    return evaluate


def _knee_point(F: np.ndarray) -> int:
    """Utopia-line knee point in normalized two-objective space."""
    if len(F) == 1:
        return 0
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    span = np.where(f_max - f_min > 1e-12, f_max - f_min, 1.0)
    F_norm = (F - f_min) / span
    a = F_norm[np.argmin(F_norm[:, 0])]
    b = F_norm[np.argmin(F_norm[:, 1])]
    ab = b - a
    ab_len = np.linalg.norm(ab) + 1e-12
    distances = [
        float(np.linalg.norm((p - a) - np.dot(p - a, ab) * ab / (ab_len**2)))
        for p in F_norm
    ]
    return int(np.argmax(distances))


def _prefer_primary_objective(F: np.ndarray) -> int:
    """Select minimum SSS, breaking exact ties by minimum SALEDI."""
    keys = tuple(F[:, col] for col in range(F.shape[1] - 1, -1, -1))
    return int(np.lexsort(keys)[0])


def _select_feasible_primary(
    X: np.ndarray,
    F: np.ndarray,
    spec: DecisionSpec,
    include_losses: bool,
    v_min_pu: float,
    v_max_pu: float,
    pv_profiles: np.ndarray | None,
    tolerance: float = 1.0e-7,
) -> tuple[int, int, bool]:
    """Select minimum SSS from candidates feasible for every PV profile."""
    profiles = (
        solar_24h_profile()[None, :]
        if pv_profiles is None
        else np.asarray(pv_profiles, dtype=float)
    )
    feasible: List[int] = []
    violation_scores = np.zeros(len(X), dtype=float)
    for i, x in enumerate(X):
        scn = decode(x, spec)
        duplicate_placements = (
            len(scn.pv_buses) - len(set(scn.pv_buses))
            + len(scn.bess_buses) - len(set(scn.bess_buses))
        )
        worst = {key: 0.0 for key in _PENALTY}
        for profile in profiles:
            result = run_timeseries(
                scn, include_losses=include_losses, v_min_pu=v_min_pu,
                v_max_pu=v_max_pu, per_unit_power_mw=spec.per_unit_power_mw,
                pv_profile=profile,
            )
            for key in worst:
                worst[key] = max(
                    worst[key], float(result.constraint_violations.get(key, 0.0))
                )
        violation_scores[i] = (
            sum(max(0.0, value - tolerance) for value in worst.values())
            + float(duplicate_placements)
        )
        if duplicate_placements == 0 and all(
            value <= tolerance for value in worst.values()
        ):
            feasible.append(i)
    if feasible:
        local = _prefer_primary_objective(F[np.asarray(feasible)])
        return int(feasible[local]), len(feasible), False
    minimum = float(np.min(violation_scores))
    candidates = np.flatnonzero(np.isclose(violation_scores, minimum))
    local = _prefer_primary_objective(F[candidates])
    return int(candidates[local]), 0, True


def run_ezoa_pipeline(
    spec: DecisionSpec,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    obl_init: bool = True,
    seed: int = 42,
    verbose: bool = True,
    include_losses: bool = True,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    init_bias_fraction: float = 0.25,
    evening_weight: float = 0.0,
    peak_weight: float = 0.0,
    evening_window: Tuple[int, int] = (16, 20),
    defense_step: float = 0.15,
    tradeoff_accept_prob: float = 0.35,
    primary_elite_trials: int = 20,
    primary_elite_step: float = 0.20,
    pv_profiles: np.ndarray | None = None,
    pv_cvar_alpha: float = 0.90,
    pv_cvar_weight: float = 0.0,
    restoration_scenario_count: int = 0,
    restoration_seed: int = 32021,
    critical_service_threshold: float = 0.95,
    restoration_cvar_alpha: float = 0.95,
    restoration_cvar_weight: float = 0.25,
) -> OptimizerPipelineResult:
    """Run MO-EZOA and evaluate the best-SSS and Pareto-knee solutions."""
    evaluate = _make_evaluate(
        spec,
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        evening_weight=evening_weight,
        peak_weight=peak_weight,
        evening_window=evening_window,
        pv_profiles=pv_profiles,
        pv_cvar_alpha=pv_cvar_alpha,
        pv_cvar_weight=pv_cvar_weight,
        restoration_scenario_count=restoration_scenario_count,
        restoration_seed=restoration_seed,
        critical_service_threshold=critical_service_threshold,
        restoration_cvar_alpha=restoration_cvar_alpha,
        restoration_cvar_weight=restoration_cvar_weight,
    )
    lo, hi = encode_bounds(spec)

    ezoa = EZOA(
        evaluate=evaluate,
        lo=lo,
        hi=hi,
        population_size=population_size,
        iterations=iterations,
        archive_capacity=archive_capacity,
        obl_init=obl_init,
        seed=seed,
        defense_step=defense_step,
        tradeoff_accept_prob=tradeoff_accept_prob,
        primary_elite_trials=0,
        primary_elite_step=primary_elite_step,
    )

    n_seed = min(population_size, max(0, int(round(population_size * init_bias_fraction))))
    for i in range(n_seed):
        ezoa.X[i] = time_biased_initial_sample(spec, seed=seed + i)

    ezoa.X[0] = _load_leveling_initial_sample(spec, include_losses=include_losses)

    def _on_iter(t: int, result: OptimizerResult) -> None:
        if not verbose:
            return
        best = result.history_best_per_obj[-1]
        print(
            f"  iter {t + 1:3d}/{iterations}  archive={len(result.archive):3d}  "
            f"f1(sss)={best[0]:.3e}  f2(loss_kWh/day)={best[1]:.3f}"
        )

    result = ezoa.run(on_iter=_on_iter)
    if primary_elite_trials > 0:
        initial_best = _prefer_primary_objective(result.archive.F)
        pre_polish_f1 = float(result.archive.F[initial_best, 0])
        polished_x, polished_f = _polish_feasible_primary(
            result.archive.X[initial_best],
            result.archive.F[initial_best],
            spec,
            evaluate,
            dispatch_sweeps=primary_elite_trials,
            initial_step_mw=primary_elite_step,
        )
        result.archive.add(polished_x[None, :], polished_f[None, :])
        if verbose:
            print(
                f"  feasible constrained polish: f1 {pre_polish_f1:.3e} -> "
                f"{float(polished_f[0]):.3e}"
            )
    if result.history_best_per_obj:
        result.history_best_per_obj[-1] = np.min(result.archive.F, axis=0)
    F = result.archive.F
    if len(F) == 0:
        raise RuntimeError("MO-EZOA completed without any Pareto-archive solution")

    knee = _knee_point(F)
    best_duck, feasible_count, used_fallback = _select_feasible_primary(
        result.archive.X, F, spec, include_losses, v_min_pu, v_max_pu, pv_profiles
    )

    report_pv_profile = None if pv_profiles is None else np.asarray(pv_profiles).mean(axis=0)
    knee_ts = run_timeseries(
        decode(result.archive.X[knee], spec),
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        per_unit_power_mw=spec.per_unit_power_mw,
        pv_profile=report_pv_profile,
    )
    best_duck_ts = run_timeseries(
        decode(result.archive.X[best_duck], spec),
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        per_unit_power_mw=spec.per_unit_power_mw,
        pv_profile=report_pv_profile,
    )

    return OptimizerPipelineResult(
        optimizer_name="ezoa",
        spec=spec,
        optimizer_result=result,
        knee_idx=knee,
        knee_scenario_result=knee_ts,
        knee_objectives=tuple(float(v) for v in F[knee]),
        best_duck_idx=best_duck,
        best_duck_scenario_result=best_duck_ts,
        best_duck_objectives=tuple(float(v) for v in F[best_duck]),
        per_seed_results=[
            OptimizerSeedResult(
                seed=int(seed),
                best_duck_x=result.archive.X[best_duck].copy(),
                best_duck_scenario_result=best_duck_ts,
                best_duck_objectives=tuple(float(v) for v in F[best_duck]),
                evaluation_count=int(evaluate.evaluation_count),
            )
        ],
        feasible_archive_count=feasible_count,
        selection_is_feasible=not used_fallback,
        selection_used_fallback=used_fallback,
    )


def run_ezoa_multiseed_pipeline(
    spec: DecisionSpec,
    seeds: Sequence[int],
    archive_capacity: int = 100,
    verbose: bool = True,
    include_losses: bool = True,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    **pipeline_kwargs,
) -> OptimizerPipelineResult:
    """Run MO-EZOA independently for each seed and merge their Pareto archives.

    Each seed's independently selected best-duck solution is also retained in
    ``per_seed_results``.  The merged archive supports an overall best-case
    result and plots, while the per-seed records support unbiased reporting of
    the full distribution.  The merged best must not replace multi-seed mean,
    spread, worst-case, success-rate, and placement-stability evidence.
    """
    merged = ParetoArchive(capacity=archive_capacity)
    history_best: List[np.ndarray] = []
    history_hv: List[float] = []
    per_seed_best_sss = []
    per_seed_results: List[OptimizerSeedResult] = []

    for k, seed in enumerate(seeds):
        if verbose:
            print(f"=== ezoa | seed {seed} ({k + 1}/{len(seeds)}) ===")
        pipe = run_ezoa_pipeline(
            spec=spec,
            archive_capacity=archive_capacity,
            seed=seed,
            verbose=verbose,
            include_losses=include_losses,
            v_min_pu=v_min_pu,
            v_max_pu=v_max_pu,
            **pipeline_kwargs,
        )
        merged.add(pipe.optimizer_result.archive.X, pipe.optimizer_result.archive.F)
        per_seed_best_sss.append(pipe.best_duck_objectives[0])
        per_seed_results.append(
            OptimizerSeedResult(
                seed=int(seed),
                best_duck_x=pipe.optimizer_result.archive.X[pipe.best_duck_idx].copy(),
                best_duck_scenario_result=pipe.best_duck_scenario_result,
                best_duck_objectives=pipe.best_duck_objectives,
                evaluation_count=int(pipe.per_seed_results[0].evaluation_count),
            )
        )
        history_best.extend(pipe.optimizer_result.history_best_per_obj)
        history_hv.extend(pipe.optimizer_result.history_hypervolume)

    if verbose:
        sss_str = ", ".join(f"{v:.3e}" for v in per_seed_best_sss)
        print(f"Per-seed best SSS: [{sss_str}]")
        print(f"Merged, deduplicated non-dominated archive size: {len(merged)}")

    F = merged.F
    if len(F) == 0:
        raise RuntimeError("Multi-seed MO-EZOA completed without any Pareto-archive solution")

    knee = _knee_point(F)
    pv_profiles = pipeline_kwargs.get("pv_profiles")
    best_duck, feasible_count, used_fallback = _select_feasible_primary(
        merged.X, F, spec, include_losses, v_min_pu, v_max_pu, pv_profiles
    )
    if verbose:
        print(
            f"Hard-feasible merged archive members: {feasible_count}/{len(F)}; "
            f"fallback used: {used_fallback}"
        )

    report_pv_profile = (
        None if pv_profiles is None else np.asarray(pv_profiles).mean(axis=0)
    )
    knee_ts = run_timeseries(
        decode(merged.X[knee], spec),
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        per_unit_power_mw=spec.per_unit_power_mw,
        pv_profile=report_pv_profile,
    )
    best_duck_ts = run_timeseries(
        decode(merged.X[best_duck], spec),
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        per_unit_power_mw=spec.per_unit_power_mw,
        pv_profile=report_pv_profile,
    )

    merged_result = OptimizerResult(
        archive=merged,
        history_best_per_obj=history_best,
        history_hypervolume=history_hv,
    )

    return OptimizerPipelineResult(
        optimizer_name="ezoa",
        spec=spec,
        optimizer_result=merged_result,
        knee_idx=knee,
        knee_scenario_result=knee_ts,
        knee_objectives=tuple(float(v) for v in F[knee]),
        best_duck_idx=best_duck,
        best_duck_scenario_result=best_duck_ts,
        best_duck_objectives=tuple(float(v) for v in F[best_duck]),
        per_seed_results=per_seed_results,
        feasible_archive_count=feasible_count,
        selection_is_feasible=not used_fallback,
        selection_used_fallback=used_fallback,
    )


def run_pipeline(**kwargs) -> OptimizerPipelineResult:
    return run_ezoa_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs) -> OptimizerPipelineResult:
    return run_ezoa_multiseed_pipeline(**kwargs)