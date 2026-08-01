# Run the verified >=81% SSS case

From this folder, run exactly:

```powershell
python run_81pct_case.py
```

No command-line options are required. Results are written to `results_81pct/`.
The verified case normally takes about two minutes on the test machine.

If Python reports a missing NumPy or Matplotlib package, install the two plotting/runtime dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Key outputs:

- `results_summary.csv`: SSS and duck-curve metrics before/after optimization.
- `resilience_indices.csv`: four-hour outage EENS, load served, resilience index, and SALEDI.
- `resilience_scenarios.csv`: all 1,536 before/after scenario records used by the violin plots.
- `bess_constraint_audit.csv`: per-unit power, SOC bounds, cycle closure, and actual curve separation.
- `fig4_netload_before_after.*`: net-load/duck-curve comparison.
- `fig5_soc.*` and `fig5_soc_differentiated.*`: single-panel actual SOC trajectories with BESS bus labels and physical limits.
- `fig6_voltage_profile.*`: literature-style bus voltage profiles with 0.95/1.05 p.u. limits.
- `fig6b_voltage_complete_24h.*`: complete 24-hour, all-bus voltage assessment.
- `fig9_resilience_indices.*`: before/after resilience index panels.
- `fig9b_resilience_violins.*`: before/after distributions over all 768 outage scenarios.
- `fig10_line_outage_resilience.*`: EENS effect for every IEEE-33 feeder line outage.
- `reliability_mcs_report.json`: paper-data audit, disclosed analyst assumptions,
  case summaries, paired confidence intervals, and claim-support flags.
- `reliability_mcs_scenario_bank.csv`: the common-random-number outage bank.
- `reliability_mcs_event_results.csv`: event-level results for `no_der`,
  `same_der_idle`, and `duck_schedule`.
- `reliability_mcs_case_summary.csv`: conditional and annualized reliability
  estimates for all three cases.
- `reliability_mcs_paired_effects.csv`: paired total-package and schedule-only
  comparisons. The schedule-only comparison holds DER placement, rating, and
  initial SOC fixed.
- `fig11_reliability_mcs_cases.*` and
  `fig12_reliability_mcs_paired_schedule_effect.*`: Monte Carlo distributions.

## Reliability publication caveat

The supplied Ahmadi et al. Table I reproduces SAIFI as 1.534 after rounding.
Applying the paper's equations and stated 4 h/2 h repair times directly gives
AENS = 28.8063 kWh/customer-year rather than the published 24.48. The JSON
report preserves both values and their difference; no calibration factor is
used.

The Monte Carlo extension uses common random numbers. It reports a reliability
improvement only when the paired 95% confidence interval supports it. Its
repair-time distribution, forecast-error distributions, uniform outage start
hour, grid-forming BESS assumption, and modeling limitations are recorded in
`reliability_mcs_report.json`. Keep that methodology block with any published
tables or archived results.

## What is optimized

The analytical load-leveling schedule is only a feasible warm start. MO-EZOA and
the constrained derivative-free phase evaluate the actual SSS, SALEDI, feeder
loss, voltage, placement, SOC, power, and cycle-closure model. Each BESS dispatch
is refined independently, followed by symmetric pairwise redispatch that preserves
aggregate power and each unit's daily energy balance. No SOC diversity pattern,
target SSS, target reduction
percentage, or final bus placement is hardcoded.

`run_paper_case.py` is the configurable research runner. Use it only when you
want to change the YAML/JSON settings or run the much slower paper-scale experiment.

## Run the 10-seed robustness study

From this folder, run:

```powershell
python run_10_seed_case.py
```

The seeds are explicitly fixed to `42, 1042, ..., 9042` in
`configs/seed_validation_10.json`. Results are written to `results_10_seeds/`.
In addition to the usual figures and CSV files, the multi-seed run writes:

- `seed_results.csv`: one independently evaluated result per seed, including
  SSS reduction, placements, voltage, BESS power, SOC, and cycle closure.
- `seed_statistics.csv`: mean, median, sample standard deviation, coefficient
  of variation, minimum/maximum reduction, and the success rate for reaching
  at least 81% reduction.

The ten runs use the same objective and constraints. Only the random seed
changes. Do not use only the best row as robustness evidence; report the full
distribution in `seed_results.csv`.
