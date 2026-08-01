# IEEE 33-bus PV-BESS duck-curve study

Version 1.3 adds optional ML-based stochastic PV profiles. See
`STOCHASTIC_PV_ML.md` for the PCA scenario methodology, historical-data CSV
format, CVaR settings, and the explicitly labeled synthetic-data fallback.
See `PAPER_METHODOLOGY.md` for the three-objective stochastic PV-BESS paper
formulation and the service-recovery interpretation boundary.

This repository is a reproducible research implementation for multi-objective
PV/BESS placement and scheduling on the radial IEEE 33-bus feeder. MO-EZOA
minimizes duck-curve SSS and daily feeder energy loss. It reports
duck-curve metrics, voltage/SOC constraints, multi-seed optimizer variability,
and a paired Monte Carlo outage assessment.

The Pareto archive is screened using the preregistered constraint
`CVaR_0.95(ENS) <= 0.80 * CVaR_0.95(no-DER ENS)`. CVaR is a constraint, not a
weighted third objective. The minimum-SSS CVaR-admissible design is validated
on the independent 100,000-scenario bank.

The operating horizon is one 24-hour day. Each BESS initial SOC is an explicit
20%-90% bounded decision with exact `SOC(24)=SOC(0)` closure. Each decoded
dispatch is projected to fit the upward and downward energy room around its
chosen initial SOC, so every evaluated trajectory remains inside 20%-90%.

## Scientific scope

The normal-operation feeder calculation uses an analytic DistFlow
approximation. Reliability islanding is an energy-adequacy calculation. These
models do not prove three-phase unbalanced AC feasibility, protection
coordination, grid-forming control stability, transient stability, or hardware
implementability.

Reliability inputs are transcribed from Ahmadi et al., UPEC 2021,
DOI `10.1109/UPEC50034.2021.9548208`. Its Table I reproduces SAIFI as 1.534
after rounding. Direct application of the published equations and 4 h/2 h
repair times gives AENS = 28.8063 kWh/customer-year rather than the reported
24.48. Both values are retained without calibration.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

## Publication run

Create a new, empty output directory:

```powershell
python run_paper_case.py `
  --config configs/publication_final.yaml `
  --output-dir outputs/publication_final_YYYYMMDD
```

The run is not accepted merely because it completes. Require
`publication_audit.json` to report `publication_checks_passed: true` and
`run_manifest.json` to report `status: complete`. The manifest contains the
configuration, source and output checksums, package versions, platform, seeds,
measured evaluation counts, constraints, and timestamps.

Run the frozen-design reliability sensitivity study afterward:

```powershell
python run_reliability_sensitivity.py `
  --design outputs/publication_final_YYYYMMDD/selected_design.json `
  --output-dir outputs/publication_final_YYYYMMDD/sensitivity `
  --scenario-count 100000
```

Report all seeds and sensitivity cases, including failures to support the
improvement hypothesis. The schedule-only causal comparison is
`duck_schedule` versus `same_der_idle`; DER sites, ratings, initial SOC, and
outage scenarios are fixed. The `no_der` comparison is a total-package effect.

See [RUN_ME.md](RUN_ME.md) for the output inventory and detailed cautions.

## Release checklist

1. Run only from the frozen configuration in a clean output directory.
2. Confirm the publication audit passes.
3. Review every seed rather than reporting only the merged best.
4. Run sensitivity and scenario-count convergence analyses.
5. Replace the contributor and repository placeholders in `CITATION.cff`.
6. Archive the exact source plus outputs and cite its SHA-256/DOI.

Historical folders bundled with earlier versions are not evidence for the
current source and must not be mixed into a final run.

The frozen configuration disables adaptive post-optimization polishing.
Consequently each seed receives exactly 25,040 objective evaluations:
`2N` for opposition-based initialization, `N` for the initialized population,
and `2*N*155` for the two MO-EZOA phases. This equal-budget choice takes
precedence over reproducing any earlier best-case percentage.
