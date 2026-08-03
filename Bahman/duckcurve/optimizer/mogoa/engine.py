from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def _s_function(distance: float, attraction: float, length_scale: float) -> float:
    return attraction * np.exp(-distance / length_scale) - np.exp(-distance)


def run_mogoa_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    c_max: float = 1.0,
    c_min: float = 1.0e-4,
    attraction: float = 0.5,
    length_scale: float = 1.5,
    grid_count: int = 10,
    grid_inflation: float = 0.10,
    selection_pressure: float = 4.0,
    deletion_pressure: float = 2.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    span = np.maximum(hi - lo, 1.0e-12)
    archive = ParetoArchive(capacity=max(archive_capacity, population_size * (iterations + 2)))
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        progress = t / max(iterations - 1, 1)
        c = c_max - progress * (c_max - c_min)
        target = archive.X[
            select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
            )
        ]

        X_next = np.empty_like(X)
        scale = 0.5 * span
        for i in range(population_size):
            social = np.zeros(X.shape[1], dtype=float)
            for j in range(population_size):
                if i == j:
                    continue
                diff = X[j] - X[i]
                normalized_distance = np.linalg.norm(diff / span)
                if normalized_distance <= 1.0e-12:
                    continue
                direction = diff / (np.linalg.norm(diff) + 1.0e-12)
                interaction_distance = 2.0 + np.remainder(normalized_distance, 2.0)
                social += scale * _s_function(interaction_distance, attraction, length_scale) * direction
            candidate = c * social + target
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