# Stochastic PV extension

The deterministic 24-hour PV curve can now be replaced by an ensemble learned
with principal-component analysis (PCA). PCA learns correlated intraday modes
from rows of historical normalized PV output; scenarios are sampled in the
learned latent space and constrained to the physical interval [0, 1].

Configure `stochastic_pv` in YAML. For measured data, set `training_csv` to a
CSV containing one day per row and exactly 24 normalized hourly values. The
included `duckcurve/data/PVGIS_hourly_PV.csv` contains two complete years
(2019-2020) of hourly PVGIS PV output and weather for Istanbul. Its UTC
timestamps are converted to UTC+3, PV power is converted to capacity factor
using the inferred round 1 MW nameplate, and complete local days are used
directly as PCA training profiles. The training set therefore includes daily
weather, interannual, and seasonal variation without synthetic augmentation.
When the path
is empty, the program uses a reproducible synthetic training set based on the
original clear-sky curve and records `synthetic_fallback` in
`stochastic_pv_model.json`. This fallback is suitable for code testing and
method development, not an empirical forecasting claim.

MO-EZOA evaluates every candidate on the same common-random-number PV scenario
bank. Each objective is its scenario mean plus `cvar_weight` times its upper
tail CVaR at `cvar_alpha`. Setting `cvar_weight: 0` gives risk-neutral expected
optimization. Placement, BESS scheduling, SOC constraints, voltage constraints,
loss modeling, Pareto archiving, and multi-seed methodology are unchanged.

Run:

```text
python run_paper_case.py --config configs/quick_test.yaml --output-dir outputs/stochastic_quick
```

Outputs include `stochastic_pv_scenarios.csv` and
`stochastic_pv_model.json`, alongside the original study artifacts.
