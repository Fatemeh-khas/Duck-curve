# MO-EZOA recovery notes

## Cause of the import failure

`duckcurve/optimizer/ezoa.py` had been overwritten by a copy of the experiment
pipeline. That copied file imported `EZOA` from `duckcurve.optimizer` while
`duckcurve.optimizer.__init__` was still importing `EZOA` from the same file.
This created the reported partially-initialized-module circular import.

The problem was not only an import statement: no `EZOA` or `EZOAResult` class
remained anywhere in the source tree.

## Reconstructed files

- `duckcurve/optimizer/ezoa.py`: restored the MO-EZOA optimizer, external Pareto
  archive guidance, ZOA foraging/defense phases, OBL support, Pareto replacement,
  crowding-aware leader selection, hypervolume history, and result dataclass.
- `duckcurve/optimizer/obl.py`: changed OBL survivor selection from a scale-biased
  raw objective sum to non-dominated sorting plus crowding distance.
- `duckcurve/experiments/ezoa_run.py`: restored the paper pipeline in its correct
  package. The two objectives remain separate; no weighted-sum aggregation is
  used. Default archive capacity is 100, consistent with the report.
- `run_paper_case.py`: added a reproducible command-line runner that saves the
  Pareto archive, decoded placements, dispatch, SOC, trajectories, summaries,
  and figures.

## Commands

From the folder containing `run_paper_case.py`:

```powershell
python -c "from duckcurve.optimizer import EZOA, EZOAResult; print('Import OK')"
```

Quick smoke test:

```powershell
python run_paper_case.py --population 10 --iterations 3 --output results_quick
```

Paper-scale run:

```powershell
python run_paper_case.py --population 80 --iterations 180 --archive 100 --seed 42 --output results
```

A paper-scale metaheuristic run can take substantial time. Do not compare a
small smoke-test run with the values in the final report.

## Reproducibility limitation

The original missing source cannot be recovered byte-for-byte from the report.
This package reconstructs a functional MO-EZOA implementation consistent with
the remaining interfaces and reported methodology. Exact reproduction of the
old numerical archive is not guaranteed because the overwritten optimizer's
precise stochastic update and selection implementation is unavailable.
