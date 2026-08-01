# Proposed paper methodology

## Contribution

The paper contributes a risk-aware stochastic framework that jointly selects
PV sites, BESS sites, initial BESS SOC, and hourly dispatch on the IEEE 33-bus
feeder. PCA is an uncertainty-modeling tool, not the claimed novelty.

## Data-driven PV scenarios

The included PVGIS file contains 2019-2020 hourly PV output for Istanbul.
UTC timestamps are converted to UTC+3 and 730 complete daily capacity-factor
profiles are retained. PCA keeps enough components to explain 95% of training
variance. Gaussian sampling in the learned latent space generates bounded,
temporally correlated 24-hour PV profiles.

## Three objectives

1. Risk-adjusted sum of squared net-load slopes (duck-curve severity).
2. Risk-adjusted daily distribution energy loss.
3. Risk-adjusted critical-load service recovery time.

For each quantity `z`, the implemented risk form is `mean(z) + weight *
CVaR(z)`. All designs use the same scenario banks.

## Service recovery

Industrial loads in the Ahmadi et al. reliability table are designated as
critical loads. Recovery time is the earliest elapsed outage time after which
at least the configured fraction (default 95%) of critical load remains served
through physical repair. If sustained service is never achieved, recovery time
equals repair duration.

The optimizer uses a conditional common-random-number bank of main-feeder
outages that affect critical load. The publication reliability assessment uses
the original unconditional component-failure bank. This distinction prevents
uninfluenceable events from diluting the optimization while preserving valid
system-wide reporting.

ENS remains a diagnostic output and does not select the design.

Final design selection is feasibility-first: every archive member is audited
against every stochastic PV profile, and minimum SSS is selected only from the
hard-feasible subset. If that subset is empty, the least-violating fallback is
explicitly labeled. A dominant voltage penalty steers the search toward the
feasible region but never substitutes for the final hard gate.

## Required comparisons

- Deterministic PV versus stochastic PCA PV.
- Risk-neutral versus risk-aware stochastic optimization.
- Proposed three-objective method versus two-objective SSS/loss method.
- PCA scenario distribution versus held-out historical daily profiles.
- Scenario-count, recovery-threshold, and CVaR-weight sensitivity.
- Multiple independent optimizer seeds and equal evaluation budgets.

## Interpretation boundary

The model estimates critical-load service recovery under islanded PV-BESS
support. It does not model repair crews, switching sequences, feeder
reconfiguration, or physical infrastructure restoration and must not be
described as a repair-time optimizer.
