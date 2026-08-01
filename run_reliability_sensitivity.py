"""Sensitivity and convergence analysis for a frozen selected DER design."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from duckcurve.data import residential_24h_profile, solar_24h_profile
from duckcurve.grid.timeseries import Scenario
from duckcurve.objectives.resilience_mcs import (
    evaluate_mcs_case,
    generate_mcs_scenario_bank,
    paired_mcs_effect,
    summarize_mcs_case,
)
from duckcurve.publication import sha256_file, write_json


def load_design(path: Path) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    power = np.asarray(data["bess_power_mw"], dtype=float)
    return Scenario(
        pv_buses=[int(v) for v in data["pv_buses"]],
        pv_unit_capacity_mw=float(data.get("pv_unit_capacity_mw", 1.0)),
        bess_buses=[int(v) for v in data["bess_buses"]],
        bess_power_mw=power,
        bess_energy_capacity_mwh=np.asarray(
            data["bess_energy_capacity_mwh"], dtype=float
        ),
        bess_init_soc_mwh=np.asarray(data["bess_initial_soc_mwh"], dtype=float),
        eta_c=float(data.get("eta_c", 0.95)),
        eta_d=float(data.get("eta_d", 0.95)),
    )


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_setting(
    design: Scenario,
    count: int,
    seed: int,
    duration: str,
    load_sigma: float,
    pv_sigma: float,
    grid_forming: bool,
    bess_power_limit_mw: float,
) -> tuple[dict, dict]:
    bank = generate_mcs_scenario_bank(
        count,
        seed,
        duration_distribution=duration,
        load_sigma=load_sigma,
        pv_sigma=pv_sigma,
    )
    idle = replace(design, bess_power_mw=np.zeros_like(design.bess_power_mw))
    reference = evaluate_mcs_case(
        "same_der_idle",
        idle,
        bank,
        bess_power_limit_mw=bess_power_limit_mw,
        grid_forming_bess=grid_forming,
        load_profile=residential_24h_profile(),
        pv_profile=solar_24h_profile(),
    )
    candidate = evaluate_mcs_case(
        "duck_schedule",
        design,
        bank,
        bess_power_limit_mw=bess_power_limit_mw,
        grid_forming_bess=grid_forming,
        load_profile=residential_24h_profile(),
        pv_profile=solar_24h_profile(),
    )
    return summarize_mcs_case(candidate), paired_mcs_effect(reference, candidate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-count", type=int, default=100_000)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = load_design(args.design)
    design_data = json.loads(args.design.read_text(encoding="utf-8"))
    bess_power_limit_mw = float(design_data.get("bess_power_limit_mw", 1.0))

    settings = [
        ("primary", "exponential", 0.05, 0.10, True),
        ("deterministic_repairs", "deterministic", 0.05, 0.10, True),
        ("no_forecast_error", "exponential", 0.0, 0.0, True),
        ("high_forecast_error", "exponential", 0.10, 0.20, True),
        ("no_grid_forming_bess", "exponential", 0.05, 0.10, False),
    ]
    sensitivity = []
    for label, duration, load_sigma, pv_sigma, grid_forming in settings:
        for seed in (2021, 3021, 4021):
            summary, effect = evaluate_setting(
                design,
                args.scenario_count,
                seed,
                duration,
                load_sigma,
                pv_sigma,
                grid_forming,
                bess_power_limit_mw,
            )
            sensitivity.append({
                "setting": label,
                "seed": seed,
                "duration_distribution": duration,
                "load_sigma": load_sigma,
                "pv_sigma": pv_sigma,
                "grid_forming_bess": grid_forming,
                "annual_aens_kwh_per_customer_year":
                    summary["annual_aens_kwh_per_customer_year"],
                **{
                    key: effect[key] for key in (
                        "mean_paired_ens_difference_kwh",
                        "paired_ens_difference_ci95_low_kwh",
                        "paired_ens_difference_ci95_high_kwh",
                        "ens_reduction_percent",
                        "supports_improvement_at_95pct",
                        "claim",
                    )
                },
            })

    convergence = []
    for count in sorted({1_000, 5_000, 20_000, args.scenario_count}):
        _, effect = evaluate_setting(
            design,
            count,
            2021,
            "exponential",
            0.05,
            0.10,
            True,
            bess_power_limit_mw,
        )
        convergence.append({
            "scenario_count": count,
            **{
                key: effect[key] for key in (
                    "mean_paired_ens_difference_kwh",
                    "paired_ens_difference_ci95_low_kwh",
                    "paired_ens_difference_ci95_high_kwh",
                    "supports_improvement_at_95pct",
                )
            },
        })

    save_csv(sensitivity, args.output_dir / "reliability_sensitivity.csv")
    save_csv(convergence, args.output_dir / "reliability_convergence.csv")
    write_json(
        args.output_dir / "sensitivity_manifest.json",
        {
            "selected_design_path": str(args.design.resolve()),
            "selected_design_sha256": sha256_file(args.design),
            "scenario_count_per_sensitivity_case": args.scenario_count,
            "settings": settings,
            "seeds": [2021, 3021, 4021],
            "claim_rule": "upper paired 95% CI for candidate-reference ENS < 0",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
