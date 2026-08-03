from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_leaders_by_grid


def run_mogwo_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    grid_count: int = 10,
    grid_inflation: float = 0.10,
    selection_pressure: float = 4.0,
    deletion_pressure: float = 2.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    archive_store_capacity = max(archive_capacity, population_size * (iterations + 2))
    archive = ParetoArchive(capacity=archive_store_capacity)
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(
        archive,
        rng,
        archive_capacity=archive_capacity,
        grid_count=grid_count,
        grid_inflation=grid_inflation,
        deletion_pressure=deletion_pressure,
    )

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        a = 2.0 - 2.0 * (t / max(iterations - 1, 1))
        X_next = np.empty_like(X)

        for i in range(population_size):
            alpha, beta, delta = select_archive_leaders_by_grid(
                archive=archive,
                rng=rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
                n_leaders=3,
            )

            r1 = rng.random(X.shape[1])
            r2 = rng.random(X.shape[1])
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            r1 = rng.random(X.shape[1])
            r2 = rng.random(X.shape[1])
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            r1 = rng.random(X.shape[1])
            r2 = rng.random(X.shape[1])
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            X_next[i] = np.clip((X1 + X2 + X3) / 3.0, lo, hi)

        X = X_next
        F = np.asarray(evaluate(X), dtype=float)
        archive.add(X, F)
        prune_archive_by_grid(
            archive,
            rng,
            archive_capacity=archive_capacity,
            grid_count=grid_count,
            grid_inflation=grid_inflation,
            deletion_pressure=deletion_pressure,
        )
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