from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from duckcurve.data import (
    generate_stochastic_pv_profiles,
    residential_24h_profile,
    solar_24h_profile,
)
from duckcurve.experiments import (
    baseline_net_load, baseline_no_pv_no_bess, build_baseline_scenario,
    run_ezoa_multiseed_pipeline,
)
from duckcurve.objectives import (
    duck_curve_variance,
    evening_ramp_peak,
    evening_ramp_rate,
    resilience_indices,
    saledi_metric,
    sum_slope_squared,
)
from duckcurve.objectives.resilience_mcs import (
    evaluate_mcs_case,
    generate_mcs_scenario_bank,
    mcs_methodology,
    paired_mcs_effect,
    scenario_bank_rows,
    summarize_mcs_case,
)
from duckcurve.data.paper_reliability import paper_analytic_base_indices
from duckcurve.grid.timeseries import Scenario, run_timeseries
from duckcurve.optimizer.encoding import (
    DecisionSpec,
    decode,
    decode_with_info,
)
from duckcurve.publication import (
    build_manifest,
    finalize_manifest,
    publication_audit,
    write_json,
)
from duckcurve.viz import (
    apply_style,
    plot_convergence,
    plot_figure_4,
    plot_load_and_pv,
    plot_line_outage_resilience,
    plot_pareto_front,
    plot_resilience_indices,
    plot_resilience_violins,
    plot_soc,
    plot_voltage_heatmaps,
    plot_voltage_profile,
    save_fig,
)
from duckcurve.viz.resilience_mcs import (
    plot_mcs_resilience_violins,
    plot_paired_ens_differences,
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    with open(p, encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            return json.load(f)
        if yaml is None:
            raise ModuleNotFoundError(
                "PyYAML is required for YAML configurations; use a JSON config or install pyyaml."
            )
        return yaml.safe_load(f)


def _pct(before: float, after: float) -> float:
    return 0.0 if abs(before) < 1e-9 else (before - after) / abs(before) * 100.0


def compute_metrics(label, nl_kw, scn, pv_profile, load_profile) -> dict:
    sss = sum_slope_squared(nl_kw)
    var = duck_curve_variance(nl_kw)
    peak = float(nl_kw.max())
    trough = float(nl_kw.min())
    r_mean = float(evening_ramp_rate(nl_kw, 16, 20))
    r_peak = float(evening_ramp_peak(nl_kw, 16, 20))

    if scn is not None:
        sal = saledi_metric(
            nl_kw,
            scn.pv_buses,
            scn.pv_unit_capacity_mw,
            pv_profile,
            scn.bess_buses,
            scn.bess_power_mw,
            load_profile,
        )
        pv_buses_str = ";".join(str(b) for b in scn.pv_buses)
        bess_buses_str = ";".join(str(b) for b in scn.bess_buses)
    else:
        sal = saledi_metric(nl_kw, [], 0.0, pv_profile, [], np.zeros((0, 24)), load_profile)
        pv_buses_str = "N/A"
        bess_buses_str = "N/A"

    return dict(
        label=label,
        sum_slope_sq_kW2=round(sss, 2),
        variance_kW2=round(var, 2),
        peak_kW=round(peak, 2),
        trough_kW=round(trough, 2),
        range_kW=round(peak - trough, 2),
        ramp_mean_kWh=round(r_mean, 2),
        ramp_peak_kWh=round(r_peak, 2),
        saledi=round(sal, 6),
        pv_buses=pv_buses_str,
        bess_buses=bess_buses_str,
    )


def compute_improvement(B, O) -> dict:
    keys = [
        "sum_slope_sq_kW2",
        "variance_kW2",
        "peak_kW",
        "trough_kW",
        "range_kW",
        "ramp_mean_kWh",
        "ramp_peak_kWh",
        "saledi",
    ]
    row = {"label": "improvement_%"}
    for k in keys:
        row[k] = round(_pct(abs(B[k]), abs(O[k])), 2)
    row.update(pv_buses="", bess_buses="")
    return row


def save_csv(rows, out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(value, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")

def build_seed_validation_rows(
    pipe, baseline_metrics, spec, pv_profile, load_profile, v_min_pu, v_max_pu
):
    """Build one independently evaluated, constraint-audited row per RNG seed."""
    rows = []
    for seed_result in pipe.per_seed_results:
        scenario, _ = decode_with_info(seed_result.best_duck_x, spec)
        ts = seed_result.best_duck_scenario_result
        metrics = compute_metrics(
            f"seed_{seed_result.seed}",
            ts.net_load_kw,
            scenario,
            pv_profile,
            load_profile,
        )
        voltage = np.asarray(ts.bus_voltage_pu, dtype=float)
        soc = np.asarray(ts.soc_mwh, dtype=float)
        power = np.asarray(scenario.bess_power_mw, dtype=float)
        cycle_error_kwh = np.abs(soc[:, -1] - soc[:, 0]) * 1000.0
        reduction = _pct(
            baseline_metrics["sum_slope_sq_kW2"],
            metrics["sum_slope_sq_kW2"],
        )
        rows.append({
            "seed": int(seed_result.seed),
            "evaluation_count": int(seed_result.evaluation_count),
            "best_sss_kW2": metrics["sum_slope_sq_kW2"],
            "optimizer_objective_sss": round(float(seed_result.best_duck_objectives[0]), 6),
            "sss_reduction_percent": round(float(reduction), 4),
            "success_ge_81_percent": bool(reduction >= 81.0),
            "saledi": metrics["saledi"],
            "pv_buses": metrics["pv_buses"],
            "bess_buses": metrics["bess_buses"],
            "minimum_voltage_pu": round(float(voltage.min()), 6),
            "maximum_voltage_pu": round(float(voltage.max()), 6),
            "hours_below_minimum_pu": int(np.sum(np.min(voltage, axis=1) < v_min_pu)),
            "hours_above_maximum_pu": int(np.sum(np.max(voltage, axis=1) > v_max_pu)),
            "maximum_bess_power_mw": round(float(np.max(np.abs(power))), 6),
            "minimum_soc_mwh": round(float(soc.min()), 6),
            "maximum_soc_mwh": round(float(soc.max()), 6),
            "maximum_cycle_error_kwh": round(float(cycle_error_kwh.max()), 6),
            "power_soc_cycle_limits_ok": bool(
                np.max(np.abs(power)) <= spec.per_unit_power_mw + 1e-9
                and soc.min() >= spec.soc_min - 1e-9
                and soc.max() <= spec.soc_max + 1e-9
                and cycle_error_kwh.max() <= 1e-2
                and voltage.min() >= v_min_pu - 1e-9
                and voltage.max() <= v_max_pu + 1e-9
            ),
        })

    if rows:
        best_sss = min(row["best_sss_kW2"] for row in rows)
        for row in rows:
            row["is_best_sss_seed"] = abs(row["best_sss_kW2"] - best_sss) <= 1e-6
    return rows


def build_seed_statistics(seed_rows):
    """Summarize the complete seed distribution without selecting only the best run."""
    if not seed_rows:
        return []
    sss = np.asarray([row["best_sss_kW2"] for row in seed_rows], dtype=float)
    reduction = np.asarray(
        [row["sss_reduction_percent"] for row in seed_rows],
        dtype=float,
    )
    sample_sss_std = float(np.std(sss, ddof=1)) if len(sss) > 1 else 0.0
    success_count = int(np.sum(reduction >= 81.0))
    return [
        {"statistic": "seed_count", "value": len(seed_rows), "unit": "runs"},
        {"statistic": "mean_sss", "value": round(float(np.mean(sss)), 4), "unit": "kW2"},
        {"statistic": "median_sss", "value": round(float(np.median(sss)), 4), "unit": "kW2"},
        {"statistic": "sample_std_sss", "value": round(sample_sss_std, 4), "unit": "kW2"},
        {
            "statistic": "sss_coefficient_of_variation",
            "value": round(100.0 * sample_sss_std / float(np.mean(sss)), 4),
            "unit": "percent",
        },
        {
            "statistic": "mean_reduction",
            "value": round(float(np.mean(reduction)), 4),
            "unit": "percent",
        },
        {
            "statistic": "median_reduction",
            "value": round(float(np.median(reduction)), 4),
            "unit": "percent",
        },
        {
            "statistic": "minimum_reduction",
            "value": round(float(np.min(reduction)), 4),
            "unit": "percent",
        },
        {
            "statistic": "maximum_reduction",
            "value": round(float(np.max(reduction)), 4),
            "unit": "percent",
        },
        {
            "statistic": "success_count_ge_81_percent",
            "value": success_count,
            "unit": "runs",
        },
        {
            "statistic": "success_rate_ge_81_percent",
            "value": round(100.0 * success_count / len(seed_rows), 2),
            "unit": "percent",
        },
        {
            "statistic": "distinct_pv_placements",
            "value": len({row["pv_buses"] for row in seed_rows}),
            "unit": "patterns",
        },
        {
            "statistic": "distinct_bess_placements",
            "value": len({row["bess_buses"] for row in seed_rows}),
            "unit": "patterns",
        },
    ]

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None, help="Where to write figures/CSVs (default: ./outputs next to this script)")
    parser.add_argument(
        "--allow-nonempty-output",
        action="store_true",
        help="Development only: permit mixing files into a non-empty output directory",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    np.random.seed(cfg.get("seed", 42))

    obj_cfg = cfg.get("objectives", {})
    include_losses = bool(obj_cfg.get("include_losses_in_net_load", True))
    voltage_cfg = cfg.get("voltage", {})
    v_min_pu = float(voltage_cfg.get("minimum_pu", 0.94))
    v_max_pu = float(voltage_cfg.get("maximum_pu", 1.05))

    spec = DecisionSpec(
        n_pv=cfg["pv"]["count"],
        n_bess=cfg["bess"]["count"],
        horizon=cfg["horizon"]["hours"],
        pv_unit_capacity_mw=cfg["pv"]["unit_capacity_mw"],
        bess_total_power_mw=cfg["bess"]["total_power_mw"],
        bess_total_energy_mwh=cfg["bess"]["total_energy_mwh"],
        eta_c=cfg["bess"]["efficiency_charge"],
        eta_d=cfg["bess"]["efficiency_discharge"],
    )

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not args.allow_nonempty_output
    ):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. "
            "Use a new directory for a publication run."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    manifest = build_manifest(Path(__file__).parent, config_path, cfg)
    write_json(out_dir / "run_manifest.json", manifest)
    formats = cfg["output"]["formats"]
    stochastic_cfg = cfg.get("stochastic_pv", {})
    restoration_cfg = cfg.get("restoration_objective", {})
    pv_profiles = None
    pv_training_source = "deterministic"
    if bool(stochastic_cfg.get("enabled", False)):
        training_csv = stochastic_cfg.get("training_csv")
        if training_csv:
            training_path = Path(training_csv)
            if not training_path.is_absolute():
                training_path = Path(__file__).parent / training_path
            training_csv = training_path
        pv_profiles, pv_model, pv_training_source = generate_stochastic_pv_profiles(
            n_scenarios=int(stochastic_cfg.get("scenario_count", 12)),
            seed=int(stochastic_cfg.get("seed", cfg.get("seed", 42))),
            training_csv=training_csv,
            variance_retained=float(stochastic_cfg.get("variance_retained", 0.95)),
            synthetic_days=int(stochastic_cfg.get("synthetic_training_days", 365)),
            synthetic_variability=float(stochastic_cfg.get("synthetic_variability", 0.18)),
        )
        save_csv(
            [
                {"scenario": i, **{f"hour_{h:02d}": float(row[h]) for h in range(24)}}
                for i, row in enumerate(pv_profiles)
            ],
            out_dir / "stochastic_pv_scenarios.csv",
        )
        save_json(
            {
                "method": "PCA latent Gaussian sampling",
                "training_source": pv_training_source,
                "scenario_count": len(pv_profiles),
                "components": int(pv_model.components_.shape[0]),
                "variance_retained": float(stochastic_cfg.get("variance_retained", 0.95)),
                "seed": int(stochastic_cfg.get("seed", cfg.get("seed", 42))),
            },
            out_dir / "stochastic_pv_model.json",
        )
        print(
            f"Stochastic PV enabled: {len(pv_profiles)} PCA scenarios; "
            f"training source={pv_training_source}"
        )
    pv_profile = (
        solar_24h_profile() if pv_profiles is None else pv_profiles.mean(axis=0)
    )
    ld_profile = residential_24h_profile()
    apply_style()

    baseline_scn = build_baseline_scenario(spec)
    baseline_duck = run_timeseries(
        baseline_scn, include_losses=include_losses, v_min_pu=v_min_pu,
        v_max_pu=v_max_pu, per_unit_power_mw=spec.per_unit_power_mw,
        pv_profile=pv_profile,
    )
    baseline_raw = baseline_no_pv_no_bess(spec)
    base_nl = baseline_duck.net_load_kw
    base_v_pu = baseline_duck.bus_voltage_pu
    raw_v_pu = baseline_raw.bus_voltage_pu

    class _Dummy:
        pv_buses = []
        pv_unit_capacity_mw = 0.0
        bess_buses = []
        bess_power_mw = np.zeros((0, 24))

    B = compute_metrics("duck baseline (PV,no BESS)", base_nl, baseline_scn, pv_profile, ld_profile)
    B_raw = compute_metrics("no_pv_no_bess", baseline_raw.net_load_kw, _Dummy(), pv_profile, ld_profile)

    pv_figure_input = pv_profile if pv_profiles is None else pv_profiles
    save_fig(
        plot_load_and_pv(ld_profile, pv_figure_input),
        out_dir,
        "fig2_load_and_pv",
        formats,
    )

    base_seed = cfg.get("seed", 42)
    configured_seeds = cfg["ezoa"].get("seeds")
    if configured_seeds is not None:
        seeds = [int(seed) for seed in configured_seeds]
        if not seeds:
            raise ValueError("ezoa.seeds must contain at least one integer seed")
        if len(seeds) != len(set(seeds)):
            raise ValueError("ezoa.seeds must contain distinct values")
        n_seeds = len(seeds)
        declared_n_seeds = cfg["ezoa"].get("n_seeds")
        if declared_n_seeds is not None and int(declared_n_seeds) != n_seeds:
            raise ValueError(
                "ezoa.n_seeds must match the number of entries in ezoa.seeds"
            )
    else:
        n_seeds = int(cfg["ezoa"].get("n_seeds", 1))
        if n_seeds < 1:
            raise ValueError("ezoa.n_seeds must be at least 1")
        seeds = [base_seed + 1000 * i for i in range(n_seeds)]
    if n_seeds > 1:
        print(f"Running {n_seeds} independent seeds: {seeds}")
        print(
            "Each seed will be retained in seed_results.csv; "
            "archives are also merged for the overall result."
        )

    pipe = run_ezoa_multiseed_pipeline(
        spec=spec,
        seeds=seeds,
        population_size=cfg["ezoa"]["population_size"],
        iterations=cfg["ezoa"]["iterations"],
        archive_capacity=cfg["ezoa"]["archive_size_cap"],
        obl_init=cfg["ezoa"]["obl_init"],
        verbose=True,
        include_losses=include_losses,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        defense_step=cfg["ezoa"].get("defense_step", 0.15),
        tradeoff_accept_prob=cfg["ezoa"].get("tradeoff_accept_prob", 0.35),
        primary_elite_trials=cfg["ezoa"].get("primary_elite_trials", 20),
        primary_elite_step=cfg["ezoa"].get("primary_elite_step", 0.20),
        pv_profiles=pv_profiles,
        pv_cvar_alpha=float(stochastic_cfg.get("cvar_alpha", 0.90)),
        pv_cvar_weight=float(stochastic_cfg.get("cvar_weight", 0.0)),
        restoration_scenario_count=(
            int(restoration_cfg.get("scenario_count", 0))
            if bool(restoration_cfg.get("enabled", False)) else 0
        ),
        restoration_seed=int(restoration_cfg.get("seed", 32021)),
        critical_service_threshold=float(
            restoration_cfg.get("critical_service_threshold", 0.95)
        ),
        restoration_cvar_alpha=float(restoration_cfg.get("cvar_alpha", 0.95)),
        restoration_cvar_weight=float(restoration_cfg.get("cvar_weight", 0.25)),
    )

    cvar_cfg = cfg.get("cvar_constraint", {})
    cvar_screening_rows = []
    cvar_constraint_satisfied = not bool(cvar_cfg.get("enabled", False))
    if bool(cvar_cfg.get("enabled", False)):
        screen_bank = generate_mcs_scenario_bank(
            int(cvar_cfg.get("screening_scenarios", 5000)),
            int(cvar_cfg.get("screening_seed", 12021)),
            duration_distribution=str(
                cvar_cfg.get("duration_distribution", "exponential")
            ),
            load_sigma=float(cvar_cfg.get("load_sigma", 0.05)),
            pv_sigma=float(cvar_cfg.get("pv_sigma", 0.10)),
        )
        no_der_screen = Scenario(
            [], 0.0, [], np.zeros((0, 24)), np.zeros(0), np.zeros(0)
        )
        baseline_cvar = summarize_mcs_case(evaluate_mcs_case(
            "no_der_screening",
            no_der_screen,
            screen_bank,
            bess_power_limit_mw=spec.per_unit_power_mw,
            load_profile=ld_profile,
            pv_profile=pv_profile,
        ))["cvar95_event_ens_kwh"]
        cvar_limit = float(
            cvar_cfg.get("max_ratio_to_no_der", 0.80)
        ) * baseline_cvar
        for archive_index, x in enumerate(pipe.ezoa.archive.X):
            summary = summarize_mcs_case(evaluate_mcs_case(
                f"archive_{archive_index}",
                decode(x, spec),
                screen_bank,
                bess_power_limit_mw=spec.per_unit_power_mw,
                load_profile=ld_profile,
                pv_profile=pv_profile,
            ))
            value = float(summary["cvar95_event_ens_kwh"])
            cvar_screening_rows.append({
                "archive_index": archive_index,
                "objective_sss": float(pipe.ezoa.archive.F[archive_index, 0]),
                "objective_daily_loss_kwh": float(pipe.ezoa.archive.F[archive_index, 1]),
                "cvar95_ens_kwh_per_event": value,
                "cvar_limit_kwh_per_event": cvar_limit,
                "cvar_ratio_to_no_der": value / baseline_cvar,
                "cvar_admissible": value <= cvar_limit,
            })
        admissible = [r for r in cvar_screening_rows if r["cvar_admissible"]]
        pool = admissible or cvar_screening_rows
        selected = min(
            pool,
            key=lambda r: (r["objective_sss"], r["objective_daily_loss_kwh"]),
        )
        cvar_constraint_satisfied = bool(admissible)
        pipe.best_duck_idx = int(selected["archive_index"])
        pipe.best_duck_objectives = tuple(
            float(v) for v in pipe.ezoa.archive.F[pipe.best_duck_idx]
        )
        pipe.best_duck_scenario_result = run_timeseries(
            decode(pipe.ezoa.archive.X[pipe.best_duck_idx], spec),
            include_losses=include_losses,
            v_min_pu=float(cfg.get("voltage", {}).get("minimum_pu", 0.94)),
            v_max_pu=float(cfg.get("voltage", {}).get("maximum_pu", 1.05)),
            per_unit_power_mw=spec.per_unit_power_mw,
        )
        save_csv(cvar_screening_rows, out_dir / "cvar_pareto_screening.csv")

    bd_res = pipe.best_duck_scenario_result
    knee_res = pipe.knee_scenario_result
    bd_scn, bd_info = decode_with_info(pipe.ezoa.archive.X[pipe.best_duck_idx], spec)
    knee_scn, knee_info = decode_with_info(pipe.ezoa.archive.X[pipe.knee_idx], spec)

    O = compute_metrics("best_duck", bd_res.net_load_kw, bd_scn, pv_profile, ld_profile)
    K = compute_metrics("knee_point", knee_res.net_load_kw, knee_scn, pv_profile, ld_profile)
    I = compute_improvement(B, O)
    seed_rows = build_seed_validation_rows(
        pipe, B, spec, pv_profile, ld_profile, v_min_pu, v_max_pu
    )
    if n_seeds > 1:
        save_csv(seed_rows, out_dir / "seed_results.csv")
        save_csv(build_seed_statistics(seed_rows), out_dir / "seed_statistics.csv")
    resilience_before = resilience_indices(
        pv_buses=baseline_scn.pv_buses,
        pv_unit_capacity_mw=baseline_scn.pv_unit_capacity_mw,
        pv_profile=pv_profile,
        load_profile=ld_profile,
        bess_buses=baseline_scn.bess_buses,
        bess_power_mw=baseline_scn.bess_power_mw,
        bess_energy_capacity_mwh=baseline_scn.bess_energy_capacity_mwh,
        bess_init_soc_mwh=baseline_scn.bess_init_soc_mwh,
        eta_c=baseline_scn.eta_c,
        eta_d=baseline_scn.eta_d,
        bess_power_limit_mw=spec.per_unit_power_mw,
        outage_duration_hours=4,
        enable_bess=False,
    )
    resilience_after = resilience_indices(
        pv_buses=bd_scn.pv_buses,
        pv_unit_capacity_mw=bd_scn.pv_unit_capacity_mw,
        pv_profile=pv_profile,
        load_profile=ld_profile,
        bess_buses=bd_scn.bess_buses,
        bess_power_mw=bd_scn.bess_power_mw,
        bess_energy_capacity_mwh=bd_scn.bess_energy_capacity_mwh,
        bess_init_soc_mwh=bd_scn.bess_init_soc_mwh,
        eta_c=bd_scn.eta_c,
        eta_d=bd_scn.eta_d,
        bess_power_limit_mw=spec.per_unit_power_mw,
        outage_duration_hours=4,
        enable_bess=True,
    )
    resilience_before["saledi"] = B["saledi"]
    resilience_after["saledi"] = O["saledi"]

    mcs_effects = None
    mcs_cfg = cfg.get("reliability_mcs", {})
    if bool(mcs_cfg.get("enabled", False)):
        mcs_count = int(mcs_cfg.get("scenario_count", 20_000))
        mcs_seed = int(mcs_cfg.get("seed", 2021))
        duration_distribution = str(
            mcs_cfg.get("duration_distribution", "exponential")
        )
        load_sigma = float(mcs_cfg.get("load_sigma", 0.05))
        pv_sigma = float(mcs_cfg.get("pv_sigma", 0.10))
        maximum_duration = float(
            mcs_cfg.get("maximum_duration_hours", 72.0)
        )
        scenario_bank = generate_mcs_scenario_bank(
            scenario_count=mcs_count,
            seed=mcs_seed,
            duration_distribution=duration_distribution,
            load_sigma=load_sigma,
            pv_sigma=pv_sigma,
            maximum_duration_hours=maximum_duration,
        )
        no_der_scn = Scenario(
            pv_buses=[],
            pv_unit_capacity_mw=0.0,
            bess_buses=[],
            bess_power_mw=np.zeros((0, 24)),
            bess_energy_capacity_mwh=np.zeros(0),
            bess_init_soc_mwh=np.zeros(0),
            eta_c=bd_scn.eta_c,
            eta_d=bd_scn.eta_d,
        )
        same_der_idle_scn = replace(
            bd_scn, bess_power_mw=np.zeros_like(bd_scn.bess_power_mw)
        )
        mcs_cases = {
            "no_der": evaluate_mcs_case(
                "no_der",
                no_der_scn,
                scenario_bank,
                bess_power_limit_mw=spec.per_unit_power_mw,
                load_profile=ld_profile,
                pv_profile=pv_profile,
            ),
            "same_der_idle": evaluate_mcs_case(
                "same_der_idle",
                same_der_idle_scn,
                scenario_bank,
                bess_power_limit_mw=spec.per_unit_power_mw,
                load_profile=ld_profile,
                pv_profile=pv_profile,
            ),
            "duck_schedule": evaluate_mcs_case(
                "duck_schedule",
                bd_scn,
                scenario_bank,
                bess_power_limit_mw=spec.per_unit_power_mw,
                load_profile=ld_profile,
                pv_profile=pv_profile,
            ),
        }
        mcs_summaries = [
            summarize_mcs_case(mcs_cases[name]) for name in mcs_cases
        ]
        if bool(cvar_cfg.get("enabled", False)):
            summary_by_case = {row["case"]: row for row in mcs_summaries}
            validation_ratio = (
                summary_by_case["duck_schedule"]["cvar95_event_ens_kwh"]
                / summary_by_case["no_der"]["cvar95_event_ens_kwh"]
            )
            cvar_constraint_satisfied = (
                cvar_constraint_satisfied
                and validation_ratio
                <= float(cvar_cfg.get("max_ratio_to_no_der", 0.80))
            )
        mcs_effects = [
            paired_mcs_effect(
                mcs_cases["no_der"],
                mcs_cases["duck_schedule"],
                "duck_schedule_vs_no_der_total_package",
            ),
            paired_mcs_effect(
                mcs_cases["same_der_idle"],
                mcs_cases["duck_schedule"],
                "duck_schedule_vs_same_der_idle_schedule_effect",
            ),
        ]
        save_csv(
            scenario_bank_rows(scenario_bank),
            out_dir / "reliability_mcs_scenario_bank.csv",
        )
        save_csv(
            [row for rows in mcs_cases.values() for row in rows],
            out_dir / "reliability_mcs_event_results.csv",
        )
        save_csv(mcs_summaries, out_dir / "reliability_mcs_case_summary.csv")
        save_csv(mcs_effects, out_dir / "reliability_mcs_paired_effects.csv")
        save_json(
            {
                "paper_data_audit": paper_analytic_base_indices(),
                "methodology": mcs_methodology(
                    mcs_count,
                    mcs_seed,
                    duration_distribution,
                    load_sigma,
                    pv_sigma,
                    maximum_duration,
                ),
                "case_summaries": mcs_summaries,
                "paired_effects": mcs_effects,
            },
            out_dir / "reliability_mcs_report.json",
        )
        save_fig(
            plot_mcs_resilience_violins(mcs_cases),
            out_dir,
            "fig11_reliability_mcs_cases",
            formats,
        )
        save_fig(
            plot_paired_ens_differences(
                mcs_cases["same_der_idle"], mcs_cases["duck_schedule"]
            ),
            out_dir,
            "fig12_reliability_mcs_paired_schedule_effect",
            formats,
        )
        schedule_effect = mcs_effects[1]
        print(
            "Paired reliability conclusion: "
            f"{schedule_effect['claim']} "
            f"CI=[{schedule_effect['paired_ens_difference_ci95_low_kwh']:.3f}, "
            f"{schedule_effect['paired_ens_difference_ci95_high_kwh']:.3f}] "
            "kWh/event."
        )

    save_fig(
        plot_figure_4(base_nl, bd_res.net_load_kw, nl_no_pv_kw=baseline_raw.net_load_kw),
        out_dir,
        "fig4_netload_before_after",
        formats,
    )
    soc_labels = [f"BESS {k+1} (Bus {bd_scn.bess_buses[k]})" for k in range(spec.n_bess)]
    for soc_figure_name in ("fig5_soc", "fig5_soc_differentiated"):
        save_fig(
            plot_soc(
                bd_res.soc_mwh,
                soc_min_mwh=spec.soc_min,
                soc_max_mwh=spec.soc_max,
                unit_labels=soc_labels,
            ),
            out_dir,
            soc_figure_name,
            formats,
        )

    v_opt_pu = bd_res.bus_voltage_pu
    v_base_min = float(base_v_pu.min()) if base_v_pu is not None else 0.95
    v_opt_min = float(v_opt_pu.min()) if v_opt_pu is not None else 0.95
    if base_v_pu is not None and v_opt_pu is not None:
        save_fig(
            plot_voltage_profile(
                base_v_pu, v_opt_pu, no_pv_v_pu=raw_v_pu,
                v_min_limit_pu=v_min_pu, v_max_limit_pu=v_max_pu,
            ),
            out_dir,
            "fig6_voltage_profile",
            formats,
        )
        save_fig(
            plot_voltage_heatmaps(
                base_v_pu, v_opt_pu, v_min_limit_pu=v_min_pu
            ),
            out_dir,
            "fig6b_voltage_complete_24h",
            formats,
        )

    save_fig(plot_pareto_front(pipe.ezoa.archive.F, highlighted=pipe.knee_idx), out_dir, "fig7_pareto_front", formats)
    save_fig(plot_convergence(pipe.ezoa.history_hypervolume), out_dir, "fig8_convergence", formats)
    save_fig(
        plot_resilience_indices(resilience_before, resilience_after),
        out_dir,
        "fig9_resilience_indices",
        formats,
    )
    save_fig(
        plot_resilience_violins(resilience_before, resilience_after),
        out_dir,
        "fig9b_resilience_violins",
        formats,
    )
    save_fig(
        plot_line_outage_resilience(resilience_before, resilience_after),
        out_dir,
        "fig10_line_outage_resilience",
        formats,
    )

    save_csv([B_raw, B, O, K, I], out_dir / "results_summary.csv")
    save_csv(
        [
            {
                "archive_index": i,
                "objective_sss": float(objectives[0]),
                "objective_daily_loss_kwh": float(objectives[1]),
                **(
                    {"objective_recovery_time_hours": float(objectives[2])}
                    if len(objectives) > 2 else {}
                ),
                **{
                    f"x_{j:03d}": float(value)
                    for j, value in enumerate(pipe.ezoa.archive.X[i])
                },
            }
            for i, objectives in enumerate(pipe.ezoa.archive.F)
        ],
        out_dir / "pareto_archive.csv",
    )
    save_json(
        {
            "selection_rule": (
                "minimum SSS among archive designs satisfying all hard "
                "constraints for every stochastic PV profile; exact ties "
                "resolved lexicographically by remaining objectives"
            ),
            "hard_feasible_archive_count": int(pipe.feasible_archive_count),
            "selection_is_hard_feasible": bool(pipe.selection_is_feasible),
            "selection_used_least_violation_fallback": bool(
                pipe.selection_used_fallback
            ),
            "archive_index": int(pipe.best_duck_idx),
            "decision_vector": [
                float(value) for value in pipe.ezoa.archive.X[pipe.best_duck_idx]
            ],
            "objectives": {
                "sss": float(pipe.best_duck_objectives[0]),
                "daily_energy_loss_kwh": float(pipe.best_duck_objectives[1]),
                **(
                    {"risk_adjusted_recovery_time_hours": float(
                        pipe.best_duck_objectives[2]
                    )}
                    if len(pipe.best_duck_objectives) > 2 else {}
                ),
            },
            "pv_buses": [int(bus) for bus in bd_scn.pv_buses],
            "pv_unit_capacity_mw": float(bd_scn.pv_unit_capacity_mw),
            "bess_buses": [int(bus) for bus in bd_scn.bess_buses],
            "bess_power_mw": np.asarray(bd_scn.bess_power_mw).tolist(),
            "bess_energy_capacity_mwh": np.asarray(
                bd_scn.bess_energy_capacity_mwh
            ).tolist(),
            "bess_initial_soc_mwh": np.asarray(
                bd_scn.bess_init_soc_mwh
            ).tolist(),
            "eta_c": float(bd_scn.eta_c),
            "eta_d": float(bd_scn.eta_d),
            "bess_power_limit_mw": float(spec.per_unit_power_mw),
        },
        out_dir / "selected_design.json",
    )
    resilience_rows = []
    for label, values in (("before", resilience_before), ("after", resilience_after)):
        resilience_rows.append({
            "scenario": label,
            "eens_kwh": round(float(values["eens_kwh"]), 4),
            "worst_case_ens_kwh": round(float(values["worst_case_ens_kwh"]), 4),
            "load_served_percent": round(float(values["load_served_percent"]), 4),
            "resilience_index": round(float(values["resilience_index"]), 6),
            "saledi": round(float(values["saledi"]), 6),
            "outage_scenarios": int(values["outage_scenarios"]),
            "outage_duration_hours": int(values["outage_duration_hours"]),
        })
    save_csv(resilience_rows, out_dir / "resilience_indices.csv")

    resilience_scenario_rows = []
    for label, values in (("before", resilience_before), ("after", resilience_after)):
        count = int(values["outage_scenarios"])
        for i in range(count):
            resilience_scenario_rows.append({
                "scenario": label,
                "line_number": int(values["scenario_line_number"][i]),
                "outage_start_hour": int(values["scenario_start_hour"][i]),
                "unserved_energy_kwh": round(float(values["scenario_unserved_kwh"][i]), 4),
                "island_demand_kwh": round(float(values["scenario_demand_kwh"][i]), 4),
                "load_served_percent": round(float(values["scenario_load_served_percent"][i]), 4),
            })
    save_csv(resilience_scenario_rows, out_dir / "resilience_scenarios.csv")

    bess_audit_rows = []
    for k in range(spec.n_bess):
        unit_soc = np.asarray(bd_res.soc_mwh[k], dtype=float)
        unit_power = np.asarray(bd_scn.bess_power_mw[k], dtype=float)
        other_units = [j for j in range(spec.n_bess) if j != k]
        max_separation = max(
            (float(np.max(np.abs(unit_soc - bd_res.soc_mwh[j]))) for j in other_units),
            default=0.0,
        )
        bess_audit_rows.append({
            "unit": k + 1,
            "bus": int(bd_scn.bess_buses[k]),
            "energy_capacity_mwh": round(float(bd_scn.bess_energy_capacity_mwh[k]), 6),
            "power_limit_mw": round(float(spec.per_unit_power_mw), 6),
            "max_abs_power_mw": round(float(np.max(np.abs(unit_power))), 6),
            "soc_min_allowed_mwh": round(float(spec.soc_min), 6),
            "soc_max_allowed_mwh": round(float(spec.soc_max), 6),
            "soc_min_observed_mwh": round(float(np.min(unit_soc)), 6),
            "soc_max_observed_mwh": round(float(np.max(unit_soc)), 6),
            "initial_soc_mwh": round(float(unit_soc[0]), 6),
            "final_soc_mwh": round(float(unit_soc[-1]), 6),
            "cycle_error_kwh": round(float((unit_soc[-1] - unit_soc[0]) * 1000.0), 6),
            "max_soc_separation_kwh": round(max_separation * 1000.0, 3),
        })
    save_csv(bess_audit_rows, out_dir / "bess_constraint_audit.csv")
    hourly = []
    for h in range(24):
        v_bh = float(base_v_pu[h].min()) if base_v_pu is not None else ""
        v_oh = float(v_opt_pu[h].min()) if v_opt_pu is not None else ""
        row = {
            "hour": h,
            "baseline_kW": round(float(base_nl[h]), 1),
            "optimised_kW": round(float(bd_res.net_load_kw[h]), 1),
            "knee_kW": round(float(knee_res.net_load_kw[h]), 1),
            "load_kW": round(float(bd_res.total_load_kw[h]), 1),
            "pv_kW": round(float(bd_res.total_pv_kw[h]), 1),
            "bess_total_kW": round(float(bd_res.total_bess_kw[h]), 1),
            "slope_base": round(float(base_nl[min(h + 1, 23)] - base_nl[h]), 1),
            "slope_opt": round(float(bd_res.net_load_kw[min(h + 1, 23)] - bd_res.net_load_kw[h]), 1),
            "v_min_base": round(v_bh, 5) if v_bh != "" else "",
            "v_min_opt": round(v_oh, 5) if v_oh != "" else "",
        }
        for k in range(spec.n_bess):
            row[f"soc_bess{k+1}_kWh"] = round(float(bd_res.soc_mwh[k, h]) * 1000, 1)
        hourly.append(row)

    save_csv(hourly, out_dir / "results_hourly.csv")

    publication_cfg = cfg.get("publication", {})
    audit = publication_audit(
        seed_rows=seed_rows,
        selected_constraints=bd_res.constraint_violations,
        expected_seed_count=n_seeds,
        expected_evaluation_count=publication_cfg.get(
            "expected_evaluations_per_seed"
        ),
        mcs_effects=mcs_effects,
        require_supported_reliability_improvement=bool(
            publication_cfg.get(
                "require_supported_reliability_improvement", False
            )
        ),
        cvar_constraint_satisfied=cvar_constraint_satisfied,
        tolerance=float(publication_cfg.get("constraint_tolerance", 1.0e-7)),
    )
    save_json(audit, out_dir / "publication_audit.json")
    manifest = finalize_manifest(manifest, out_dir, audit)
    write_json(out_dir / "run_manifest.json", manifest)

    print(f"Saved results to: {out_dir}")
    print(
        "Publication audit: "
        + ("PASS" if audit["publication_checks_passed"] else "FAIL")
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
