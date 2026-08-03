from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def _alo_ratio(progress: float) -> float:
    if progress > 0.95:
        return 1.0 + 1.0e6 * progress
    if progress > 0.90:
        return 1.0 + 1.0e5 * progress
    if progress > 0.75:
        return 1.0 + 1.0e4 * progress
    if progress > 0.50:
        return 1.0 + 1.0e3 * progress
    if progress > 0.10:
        return 1.0 + 1.0e2 * progress
    return 1.0 + 1.0e1 * progress


def _random_walk_position(
    center: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    steps: int,
) -> np.ndarray:
    walk = np.cumsum(rng.choice((-1.0, 1.0), size=(steps, center.size)), axis=0)
    walk_min = walk.min(axis=0)
    walk_max = walk.max(axis=0)
    normalized = (walk[-1] - walk_min) / np.maximum(walk_max - walk_min, 1.0e-12)
    radius = (hi - lo) / max(ratio, 1.0)
    lower = np.clip(center - radius, lo, hi)
    upper = np.clip(center + radius, lo, hi)
    return lower + normalized * (upper - lower)


def run_moalo_engine(
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
    archive = ParetoArchive(capacity=max(archive_capacity, population_size * (iterations + 2)))
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        progress = t / max(iterations - 1, 1)
        ratio = _alo_ratio(progress)
        steps = max(4, min(64, t + 4))
        elite_index = select_archive_member_by_grid(
            archive,
            rng,
            grid_count=grid_count,
            grid_inflation=grid_inflation,
            selection_pressure=selection_pressure,
        )
        elite = archive.X[elite_index]
        X_next = np.empty_like(X)
        for i in range(population_size):
            antlion_index = select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
            )
            antlion = archive.X[antlion_index]
            walk_antlion = _random_walk_position(antlion, lo, hi, ratio, rng, steps)
            walk_elite = _random_walk_position(elite, lo, hi, ratio, rng, steps)
            X_next[i] = np.clip(0.5 * (walk_antlion + walk_elite), lo, hi)

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