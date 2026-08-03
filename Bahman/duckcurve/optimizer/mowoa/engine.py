from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def run_mowoa_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    spiral_constant: float = 1.0,
    grid_count: int = 10,
    grid_inflation: float = 0.10,
    selection_pressure: float = 4.0,
    deletion_pressure: float = 2.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    archive = ParetoArchive(capacity=max(archive_capacity, population_size * (iterations + 2)))
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        a = 2.0 - 2.0 * (t / max(iterations - 1, 1))
        X_next = np.empty_like(X)
        for i in range(population_size):
            leader_idx = select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
            )
            leader = archive.X[leader_idx]
            r1 = rng.random(X.shape[1])
            r2 = rng.random(X.shape[1])
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    D = np.abs(C * leader - X[i])
                    candidate = leader - A * D
                else:
                    rand_pos = X[rng.integers(0, population_size)]
                    D = np.abs(C * rand_pos - X[i])
                    candidate = rand_pos - A * D
            else:
                l = rng.uniform(-1.0, 1.0, size=X.shape[1])
                D = np.abs(leader - X[i])
                candidate = D * np.exp(spiral_constant * l) * np.cos(2.0 * np.pi * l) + leader
            X_next[i] = np.clip(candidate, lo, hi)

        X = X_next
        F = np.asarray(evaluate(X), dtype=float)
        archive.add(X, F)
        prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)
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