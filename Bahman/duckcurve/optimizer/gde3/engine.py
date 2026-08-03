from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive, dominates
from ..common import OptimizerResult
from ..moo import record_archive_history, select_population


def run_gde3_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    differential_weight: float = 0.5,
    crossover_rate: float = 0.9,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    F = np.asarray(evaluate(X), dtype=float)
    archive = ParetoArchive(capacity=archive_capacity)
    archive.add(X, F)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        pool_X = []
        pool_F = []
        for i in range(population_size):
            candidates = [idx for idx in range(population_size) if idx != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
            mutant = np.clip(X[r1] + differential_weight * (X[r2] - X[r3]), lo, hi)
            mask = rng.random(X.shape[1]) < crossover_rate
            mask[rng.integers(0, X.shape[1])] = True
            trial = np.where(mask, mutant, X[i])
            trial = np.clip(trial, lo, hi)
            f_trial = np.asarray(evaluate(trial[None, :]), dtype=float)[0]

            if dominates(f_trial, F[i]):
                pool_X.append(trial)
                pool_F.append(f_trial)
            elif dominates(F[i], f_trial):
                pool_X.append(X[i].copy())
                pool_F.append(F[i].copy())
            else:
                pool_X.extend((X[i].copy(), trial))
                pool_F.extend((F[i].copy(), f_trial))

        X, F, _, _ = select_population(
            np.asarray(pool_X, dtype=float),
            np.asarray(pool_F, dtype=float),
            population_size,
        )
        archive.add(X, F)
        ref_point = record_archive_history(archive.F, history_best, history_hv, ref_point)
        if verbose:
            best = history_best[-1]
            print(
                f"  iter {t + 1:3d}/{iterations}  archive={len(archive):3d}  "
                f"f1(sss)={best[0]:.3e}  f2(loss_kWh/day)={best[1]:.3f}"
            )

    return OptimizerResult(
        archive=archive,
        history_best_per_obj=history_best,
        history_hypervolume=history_hv,
    )