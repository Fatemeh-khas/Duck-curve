"""Paired Monte Carlo reliability assessment using Ahmadi et al. data.

The component failure rates, customers, load classes, and mean repair times
come from Table I of Ahmadi et al. (UPEC 2021). Sampling an outage start hour,
using a repair-time distribution, and applying load/PV forecast uncertainty
are explicit extensions because the paper does not specify an MCS.

The principal attribution comparison is ``same_der_idle`` versus
``duck_schedule``: both cases have identical PV/BESS sites, ratings, and
initial SOC. Only the normal grid-connected BESS schedule differs. Common
random numbers make every comparison paired.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..data import residential_24h_profile, solar_24h_profile
from ..data.paper_reliability import (
    LATERAL_REPAIR_MEAN_HOURS,
    MAIN_REPAIR_MEAN_HOURS,
    PAPER_RELIABILITY_RECORDS,
)
from ..grid.timeseries import Scenario


TOTAL_PAPER_CUSTOMERS = int(sum(r.customers for r in PAPER_RELIABILITY_RECORDS))
LOAD_PRIORITY = ("industrial", "commercial", "residential")
_RECORD_BY_BRANCH = {r.branch_number: r for r in PAPER_RELIABILITY_RECORDS}


def _downstream_buses() -> Dict[int, frozenset[int]]:
    children: Dict[int, List[int]] = {}
    for record in PAPER_RELIABILITY_RECORDS:
        children.setdefault(record.from_bus, []).append(record.to_bus)

    def descendants(root: int) -> frozenset[int]:
        result = {root}
        stack = [root]
        while stack:
            parent = stack.pop()
            for child in children.get(parent, []):
                if child not in result:
                    result.add(child)
                    stack.append(child)
        return frozenset(result)

    return {
        record.branch_number: descendants(record.to_bus)
        for record in PAPER_RELIABILITY_RECORDS
    }


_DOWNSTREAM_BUSES = _downstream_buses()


@dataclass(frozen=True)
class MCSOutageScenario:
    """One annual component-failure event sampled conditionally on an outage."""

    scenario_id: int
    component_type: str
    branch_number: int
    start_hour: int
    duration_hours: float
    load_multiplier: float
    pv_multiplier: float


def total_annual_component_failure_rate() -> float:
    return float(
        sum(
            record.main_failure_rate_per_year
            + record.lateral_failure_rate_per_year
            for record in PAPER_RELIABILITY_RECORDS
        )
    )


def generate_mcs_scenario_bank(
    scenario_count: int,
    seed: int,
    duration_distribution: str = "exponential",
    load_sigma: float = 0.0,
    pv_sigma: float = 0.0,
    maximum_duration_hours: float = 72.0,
) -> Tuple[MCSOutageScenario, ...]:
    """Generate a reproducible bank shared by all counterfactual cases.

    Failure components are sampled in proportion to the annual rates in the
    paper. ``deterministic`` uses the paper's stated 4 h/2 h repair durations.
    ``exponential`` treats those values as distribution means, an explicitly
    declared analyst extension.
    """
    if scenario_count < 1:
        raise ValueError("scenario_count must be at least 1")
    if duration_distribution not in {"deterministic", "exponential"}:
        raise ValueError("duration_distribution must be deterministic or exponential")
    if load_sigma < 0.0 or pv_sigma < 0.0:
        raise ValueError("load_sigma and pv_sigma must be non-negative")
    if maximum_duration_hours <= 0.0:
        raise ValueError("maximum_duration_hours must be positive")

    components: List[Tuple[str, int, float, float]] = []
    for record in PAPER_RELIABILITY_RECORDS:
        components.append(
            (
                "main",
                record.branch_number,
                record.main_failure_rate_per_year,
                MAIN_REPAIR_MEAN_HOURS,
            )
        )
        components.append(
            (
                "lateral",
                record.branch_number,
                record.lateral_failure_rate_per_year,
                LATERAL_REPAIR_MEAN_HOURS,
            )
        )

    rates = np.asarray([component[2] for component in components], dtype=float)
    probabilities = rates / rates.sum()
    rng = np.random.default_rng(int(seed))
    component_indices = rng.choice(
        len(components), size=int(scenario_count), p=probabilities
    )
    start_hours = rng.integers(0, 24, size=int(scenario_count))

    bank: List[MCSOutageScenario] = []
    for scenario_id, (component_idx, start_hour) in enumerate(
        zip(component_indices, start_hours)
    ):
        component_type, branch_number, _, mean_duration = components[int(component_idx)]
        if duration_distribution == "deterministic":
            duration = mean_duration
        else:
            duration = float(rng.exponential(mean_duration))
        duration = float(np.clip(duration, 1.0 / 60.0, maximum_duration_hours))
        load_multiplier = (
            1.0 if load_sigma == 0.0 else max(0.0, float(rng.normal(1.0, load_sigma)))
        )
        pv_multiplier = (
            1.0 if pv_sigma == 0.0 else max(0.0, float(rng.normal(1.0, pv_sigma)))
        )
        bank.append(
            MCSOutageScenario(
                scenario_id=scenario_id,
                component_type=component_type,
                branch_number=branch_number,
                start_hour=int(start_hour),
                duration_hours=duration,
                load_multiplier=load_multiplier,
                pv_multiplier=pv_multiplier,
            )
        )
    return tuple(bank)


def scenario_bank_rows(
    scenario_bank: Sequence[MCSOutageScenario],
) -> List[dict]:
    return [asdict(scenario) for scenario in scenario_bank]


def generate_critical_restoration_scenario_bank(
    scenario_count: int,
    seed: int,
    **kwargs,
) -> Tuple[MCSOutageScenario, ...]:
    """Generate main-feeder outages that affect at least one critical load.

    This conditional bank is intended for the service-recovery optimization
    objective. The unconditional bank remains the basis for system reliability
    reporting, so conditioning is never hidden in annual reliability claims.
    """
    if scenario_count < 1:
        raise ValueError("scenario_count must be at least 1")
    selected: List[MCSOutageScenario] = []
    batch_seed = int(seed)
    while len(selected) < scenario_count:
        batch = generate_mcs_scenario_bank(
            max(64, 4 * scenario_count), batch_seed, **kwargs
        )
        for event in batch:
            if event.component_type != "main":
                continue
            if any(r.load_type == "industrial" for r in _event_records(event)):
                selected.append(event)
                if len(selected) == scenario_count:
                    break
        batch_seed += 1
    return tuple(
        MCSOutageScenario(
            scenario_id=i,
            component_type=e.component_type,
            branch_number=e.branch_number,
            start_hour=e.start_hour,
            duration_hours=e.duration_hours,
            load_multiplier=e.load_multiplier,
            pv_multiplier=e.pv_multiplier,
        )
        for i, e in enumerate(selected)
    )


def _scheduled_soc(scenario: Scenario) -> np.ndarray:
    n_bess = len(scenario.bess_buses)
    horizon = int(np.asarray(scenario.bess_power_mw).shape[1]) if n_bess else 24
    soc = np.zeros((n_bess, horizon + 1), dtype=float)
    if n_bess == 0:
        return soc
    soc[:, 0] = np.asarray(scenario.bess_init_soc_mwh, dtype=float)
    power = np.asarray(scenario.bess_power_mw, dtype=float)
    for hour in range(horizon):
        p = power[:, hour]
        soc[:, hour + 1] = soc[:, hour] + np.where(
            p < 0.0,
            (-p) * float(scenario.eta_c),
            -p / float(scenario.eta_d),
        )
    return soc


def _event_records(event: MCSOutageScenario):
    if event.component_type == "lateral":
        return (_RECORD_BY_BRANCH[event.branch_number],)
    downstream = _DOWNSTREAM_BUSES[event.branch_number]
    return tuple(
        record
        for record in PAPER_RELIABILITY_RECORDS
        if record.to_bus in downstream
    )


def _allocate_by_priority(
    demand_by_type_kw: Dict[str, float],
    available_supply_kw: float,
) -> Dict[str, float]:
    served: Dict[str, float] = {}
    remaining = max(0.0, float(available_supply_kw))
    for load_type in LOAD_PRIORITY:
        demand = max(0.0, float(demand_by_type_kw.get(load_type, 0.0)))
        served[load_type] = min(demand, remaining)
        remaining -= served[load_type]
    return served


def evaluate_mcs_case(
    case_name: str,
    scenario: Scenario,
    scenario_bank: Sequence[MCSOutageScenario],
    bess_power_limit_mw: float,
    grid_forming_bess: bool = True,
    load_profile: np.ndarray | None = None,
    pv_profile: np.ndarray | None = None,
    normalize_average_load_profile: bool = True,
    critical_service_threshold: float = 0.95,
) -> List[dict]:
    """Evaluate one DER counterfactual against a shared outage bank.

    For a main-feeder fault, downstream DER can operate only when the island
    contains a BESS and ``grid_forming_bess`` is true. For a load-lateral
    fault, the affected load point is disconnected and local DER cannot serve
    it. PV-surplus charging during the event is omitted conservatively.
    """
    load_profile = np.asarray(
        residential_24h_profile() if load_profile is None else load_profile,
        dtype=float,
    )
    pv_profile = np.asarray(
        solar_24h_profile() if pv_profile is None else pv_profile,
        dtype=float,
    )
    if load_profile.shape != (24,) or pv_profile.shape != (24,):
        raise ValueError("load_profile and pv_profile must each contain 24 values")
    if not np.isfinite(load_profile).all() or np.any(load_profile < 0.0):
        raise ValueError("load_profile must be finite and non-negative")
    if not np.isfinite(pv_profile).all() or np.any(pv_profile < 0.0):
        raise ValueError("pv_profile must be finite and non-negative")
    if normalize_average_load_profile:
        profile_mean = float(np.mean(load_profile))
        if profile_mean <= 0.0:
            raise ValueError("load_profile must have a positive mean")
        # Table I labels these values as average loads. Unit-mean normalization
        # preserves that annual average while retaining the supplied 24 h shape.
        load_profile = load_profile / profile_mean
    if bess_power_limit_mw < 0.0:
        raise ValueError("bess_power_limit_mw must be non-negative")
    if not 0.0 < critical_service_threshold <= 1.0:
        raise ValueError("critical_service_threshold must be in (0, 1]")

    soc = _scheduled_soc(scenario)
    capacities = np.asarray(scenario.bess_energy_capacity_mwh, dtype=float)
    rows: List[dict] = []

    for event in scenario_bank:
        records = _event_records(event)
        affected_buses = {record.to_bus for record in records}
        local_bess = [
            k for k, bus in enumerate(scenario.bess_buses) if bus in affected_buses
        ]
        island_enabled = bool(
            event.component_type == "main" and grid_forming_bess and local_bess
        )
        local_pv_count = (
            sum(bus in affected_buses for bus in scenario.pv_buses)
            if island_enabled
            else 0
        )

        available_bess_kwh = 0.0
        if island_enabled:
            schedule_hour = int(event.start_hour % soc.shape[1])
            available_bess_kwh = float(
                sum(
                    max(0.0, soc[k, schedule_hour] - 0.20 * capacities[k])
                    * float(scenario.eta_d)
                    * 1000.0
                    for k in local_bess
                )
            )
        bess_power_limit_kw = (
            len(local_bess) * float(bess_power_limit_mw) * 1000.0
            if island_enabled
            else 0.0
        )

        total_demand_kwh = 0.0
        total_served_kwh = 0.0
        industrial_demand_kwh = 0.0
        industrial_served_kwh = 0.0
        interrupted_types: set[str] = set()
        critical_interval_service: List[float] = []
        interval_durations: List[float] = []
        remaining_duration = float(event.duration_hours)
        offset = 0

        while remaining_duration > 1e-12:
            interval_hours = min(1.0, remaining_duration)
            hour = (int(event.start_hour) + offset) % 24
            demand_by_type_kw = {
                load_type: float(
                    sum(
                        record.active_load_kw
                        for record in records
                        if record.load_type == load_type
                    )
                    * load_profile[hour]
                    * event.load_multiplier
                )
                for load_type in LOAD_PRIORITY
            }
            total_demand_kw = float(sum(demand_by_type_kw.values()))

            pv_supply_kw = 0.0
            if island_enabled:
                pv_supply_kw = (
                    local_pv_count
                    * float(scenario.pv_unit_capacity_mw)
                    * 1000.0
                    * pv_profile[hour]
                    * event.pv_multiplier
                )
            pv_used_kw = min(total_demand_kw, max(0.0, pv_supply_kw))
            deficit_kw = max(0.0, total_demand_kw - pv_used_kw)
            bess_supply_kw = min(
                deficit_kw,
                bess_power_limit_kw,
                available_bess_kwh / interval_hours,
            )
            available_bess_kwh -= bess_supply_kw * interval_hours

            served_by_type_kw = _allocate_by_priority(
                demand_by_type_kw, pv_used_kw + bess_supply_kw
            )
            for load_type in LOAD_PRIORITY:
                if served_by_type_kw[load_type] < demand_by_type_kw[load_type] - 1e-9:
                    interrupted_types.add(load_type)

            served_kw = float(sum(served_by_type_kw.values()))
            total_demand_kwh += total_demand_kw * interval_hours
            total_served_kwh += served_kw * interval_hours
            industrial_demand_kwh += demand_by_type_kw["industrial"] * interval_hours
            industrial_served_kwh += served_by_type_kw["industrial"] * interval_hours
            critical_interval_service.append(
                served_by_type_kw["industrial"] / demand_by_type_kw["industrial"]
                if demand_by_type_kw["industrial"] > 0.0 else 1.0
            )
            interval_durations.append(interval_hours)
            remaining_duration -= interval_hours
            offset += 1

        ens_kwh = max(0.0, total_demand_kwh - total_served_kwh)
        served_percent = (
            100.0 * total_served_kwh / total_demand_kwh
            if total_demand_kwh > 0.0 else 100.0
        )
        industrial_served_percent = (
            100.0 * industrial_served_kwh / industrial_demand_kwh
            if industrial_demand_kwh > 0.0 else 100.0
        )
        # Earliest elapsed time after which critical service stays above the
        # declared threshold for the remainder of the outage. If this never
        # happens, service recovery coincides with physical grid repair.
        recovery_time_hours = float(event.duration_hours)
        elapsed = np.concatenate(([0.0], np.cumsum(interval_durations)))
        for idx in range(len(critical_interval_service)):
            if all(
                fraction + 1e-12 >= critical_service_threshold
                for fraction in critical_interval_service[idx:]
            ):
                recovery_time_hours = float(elapsed[idx])
                break
        interrupted_customers = int(
            sum(
                record.customers
                for record in records
                if record.load_type in interrupted_types
            )
        )
        rows.append(
            {
                "case": case_name,
                "scenario_id": int(event.scenario_id),
                "component_type": event.component_type,
                "branch_number": int(event.branch_number),
                "outage_start_hour": int(event.start_hour),
                "outage_duration_hours": float(event.duration_hours),
                "load_multiplier": float(event.load_multiplier),
                "pv_multiplier": float(event.pv_multiplier),
                "unserved_energy_kwh": ens_kwh,
                "affected_demand_kwh": total_demand_kwh,
                "load_served_percent": served_percent,
                "industrial_load_served_percent": industrial_served_percent,
                "critical_load_served_percent": industrial_served_percent,
                "critical_service_threshold_percent": (
                    100.0 * critical_service_threshold
                ),
                "service_recovery_time_hours": recovery_time_hours,
                "immediate_critical_load_recovery": bool(
                    recovery_time_hours <= 1e-12
                ),
                "interrupted_customers": interrupted_customers,
                "full_load_survival": bool(ens_kwh <= 1e-9),
                "local_bess_count": len(local_bess),
                "grid_forming_available": island_enabled,
                "average_load_profile_normalized": bool(
                    normalize_average_load_profile
                ),
            }
        )
    return rows


def _mean_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half_width, mean + half_width


def summarize_mcs_case(case_rows: Sequence[dict]) -> dict:
    """Return conditional-event and annualized reliability statistics."""
    if not case_rows:
        raise ValueError("case_rows must not be empty")
    ens = np.asarray([row["unserved_energy_kwh"] for row in case_rows], dtype=float)
    demand = np.asarray([row["affected_demand_kwh"] for row in case_rows], dtype=float)
    served = np.asarray([row["load_served_percent"] for row in case_rows], dtype=float)
    industrial_served = np.asarray(
        [row["industrial_load_served_percent"] for row in case_rows], dtype=float
    )
    interrupted = np.asarray(
        [row["interrupted_customers"] for row in case_rows], dtype=float
    )
    recovery = np.asarray(
        [row["service_recovery_time_hours"] for row in case_rows], dtype=float
    )
    critical_served = np.asarray(
        [row["critical_load_served_percent"] for row in case_rows], dtype=float
    )
    annual_rate = total_annual_component_failure_rate()

    mean_ens, mean_ens_lo, mean_ens_hi = _mean_ci95(ens)
    mean_interruptions, interrupted_lo, interrupted_hi = _mean_ci95(interrupted)
    annual_scale = annual_rate / TOTAL_PAPER_CUSTOMERS
    quantile_95 = float(np.quantile(ens, 0.95))
    tail = ens[ens >= quantile_95]
    recovery_q95 = float(np.quantile(recovery, 0.95))
    recovery_tail = recovery[recovery >= recovery_q95]
    mean_recovery, recovery_lo, recovery_hi = _mean_ci95(recovery)
    threshold = float(case_rows[0]["critical_service_threshold_percent"])

    return {
        "case": str(case_rows[0]["case"]),
        "scenario_count": int(len(case_rows)),
        "total_annual_component_failure_rate": annual_rate,
        "conditional_mean_ens_kwh_per_event": mean_ens,
        "conditional_mean_ens_ci95_low_kwh": max(0.0, mean_ens_lo),
        "conditional_mean_ens_ci95_high_kwh": mean_ens_hi,
        "annual_aens_kwh_per_customer_year": mean_ens * annual_scale,
        "annual_aens_ci95_low_kwh_per_customer_year": (
            max(0.0, mean_ens_lo) * annual_scale
        ),
        "annual_aens_ci95_high_kwh_per_customer_year": mean_ens_hi * annual_scale,
        "saifi_interruptions_per_customer_year": mean_interruptions * annual_scale,
        "saifi_ci95_low_interruptions_per_customer_year": (
            max(0.0, interrupted_lo) * annual_scale
        ),
        "saifi_ci95_high_interruptions_per_customer_year": (
            interrupted_hi * annual_scale
        ),
        "aggregate_load_served_percent": (
            100.0 * (1.0 - ens.sum() / demand.sum())
            if demand.sum() > 0.0 else 100.0
        ),
        "mean_event_load_served_percent": float(np.mean(served)),
        "mean_industrial_load_served_percent": float(np.mean(industrial_served)),
        "full_load_survival_probability": float(
            np.mean(
                np.asarray(
                    [row["full_load_survival"] for row in case_rows], dtype=float
                )
            )
        ),
        "p95_event_ens_kwh": quantile_95,
        "cvar95_event_ens_kwh": float(np.mean(tail)),
        "mean_service_recovery_time_hours": mean_recovery,
        "mean_service_recovery_time_ci95_low_hours": max(0.0, recovery_lo),
        "mean_service_recovery_time_ci95_high_hours": recovery_hi,
        "p95_service_recovery_time_hours": recovery_q95,
        "cvar95_service_recovery_time_hours": float(np.mean(recovery_tail)),
        "mean_critical_load_served_percent": float(np.mean(critical_served)),
        "critical_service_threshold_percent": threshold,
        "critical_service_compliance_probability": float(
            np.mean(critical_served + 1e-12 >= threshold)
        ),
        "immediate_critical_load_recovery_probability": float(
            np.mean(recovery <= 1e-12)
        ),
    }


def paired_recovery_time_effect(
    reference_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    comparison_name: str | None = None,
) -> dict:
    """Paired candidate-minus-reference service-recovery-time comparison."""
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise ValueError("paired cases must have the same non-zero scenario count")
    if [r["scenario_id"] for r in reference_rows] != [
        r["scenario_id"] for r in candidate_rows
    ]:
        raise ValueError("paired cases must use the same ordered scenario IDs")
    reference = np.asarray(
        [r["service_recovery_time_hours"] for r in reference_rows], dtype=float
    )
    candidate = np.asarray(
        [r["service_recovery_time_hours"] for r in candidate_rows], dtype=float
    )
    difference = candidate - reference
    mean_difference, ci_low, ci_high = _mean_ci95(difference)
    return {
        "comparison": comparison_name or "candidate_vs_reference",
        "scenario_count": len(reference_rows),
        "mean_paired_recovery_time_difference_hours": mean_difference,
        "paired_recovery_time_difference_ci95_low_hours": ci_low,
        "paired_recovery_time_difference_ci95_high_hours": ci_high,
        "probability_candidate_faster_recovery": float(
            np.mean(difference < -1e-12)
        ),
        "supports_faster_recovery_at_95pct": bool(ci_high < 0.0),
    }


def paired_mcs_effect(
    reference_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    comparison_name: str | None = None,
) -> dict:
    """Quantify a paired ENS change; improvement requires an upper CI below 0."""
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise ValueError("paired cases must have the same non-zero scenario count")
    reference_ids = [int(row["scenario_id"]) for row in reference_rows]
    candidate_ids = [int(row["scenario_id"]) for row in candidate_rows]
    if reference_ids != candidate_ids:
        raise ValueError("paired cases must use the same ordered scenario IDs")

    reference = np.asarray(
        [row["unserved_energy_kwh"] for row in reference_rows], dtype=float
    )
    candidate = np.asarray(
        [row["unserved_energy_kwh"] for row in candidate_rows], dtype=float
    )
    difference = candidate - reference
    mean_difference, ci_low, ci_high = _mean_ci95(difference)
    annual_scale = total_annual_component_failure_rate() / TOTAL_PAPER_CUSTOMERS
    reference_mean = float(np.mean(reference))
    reduction = (
        100.0 * (reference_mean - float(np.mean(candidate))) / reference_mean
        if reference_mean > 0.0 else 0.0
    )
    reference_name = str(reference_rows[0]["case"])
    candidate_name = str(candidate_rows[0]["case"])
    return {
        "comparison": comparison_name or f"{candidate_name}_vs_{reference_name}",
        "reference_case": reference_name,
        "candidate_case": candidate_name,
        "scenario_count": int(len(reference_rows)),
        "mean_paired_ens_difference_kwh": mean_difference,
        "paired_ens_difference_ci95_low_kwh": ci_low,
        "paired_ens_difference_ci95_high_kwh": ci_high,
        "annual_aens_difference_kwh_per_customer_year": (
            mean_difference * annual_scale
        ),
        "annual_aens_difference_ci95_low_kwh_per_customer_year": (
            ci_low * annual_scale
        ),
        "annual_aens_difference_ci95_high_kwh_per_customer_year": (
            ci_high * annual_scale
        ),
        "ens_reduction_percent": reduction,
        "probability_candidate_lower_ens": float(np.mean(difference < -1e-9)),
        "probability_cases_equal_ens": float(np.mean(np.abs(difference) <= 1e-9)),
        "supports_improvement_at_95pct": bool(ci_high < 0.0),
        "claim": (
            "Paired 95% confidence interval supports lower ENS."
            if ci_high < 0.0
            else "Paired 95% confidence interval does not support lower ENS."
        ),
    }


def mcs_methodology(
    scenario_count: int,
    seed: int,
    duration_distribution: str,
    load_sigma: float,
    pv_sigma: float,
    maximum_duration_hours: float,
) -> dict:
    """Return machine-readable attribution and limitations for publication."""
    return {
        "paper_derived": {
            "source_doi": "10.1109/UPEC50034.2021.9548208",
            "table": "Table I",
            "fields": [
                "radial topology",
                "main and lateral annual failure rates",
                "active average loads",
                "load classes",
                "customer counts",
                "main/lateral mean repair times of 4 h/2 h",
            ],
        },
        "analyst_assumptions": {
            "scenario_count": int(scenario_count),
            "seed": int(seed),
            "duration_distribution": duration_distribution,
            "load_multiplier_sigma": float(load_sigma),
            "pv_multiplier_sigma": float(pv_sigma),
            "maximum_duration_hours": float(maximum_duration_hours),
            "uniform_outage_start_hour": True,
            "average_load_profile_normalized_to_unit_mean": True,
            "component_sampling": "probability proportional to annual failure rate",
            "common_random_numbers": True,
            "load_priority": list(LOAD_PRIORITY),
            "pv_surplus_charging_during_outage": False,
            "minimum_bess_soc_fraction": 0.20,
        },
        "causal_comparison": {
            "reference": "same_der_idle",
            "candidate": "duck_schedule",
            "held_fixed": [
                "PV sites and ratings",
                "BESS sites, power and energy ratings",
                "initial BESS state of charge",
                "outage scenarios",
            ],
            "changed": "normal grid-connected BESS schedule before the outage",
        },
        "limitations": [
            "conditional-event sampling is annualized by the summed component rate",
            "component failures are independent and overlapping outages are omitted",
            "repair-time distribution and forecast errors are analyst assumptions",
            "islanding is an energy-adequacy model, not an AC or transient-stability proof",
            "partial class curtailment counts all customers in that class as interrupted",
            "normal-approximation confidence intervals quantify Monte Carlo sampling error",
        ],
    }
