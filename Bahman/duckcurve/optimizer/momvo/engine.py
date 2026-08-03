from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive, crowding_distance
from ..common import OptimizerResult
from ..moo import non_dominated_sort, prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def _selection_scores(F: np.ndarray) -> np.ndarray:
    fronts, rank = non_dominated_sort(F)
    crowding = np.zeros(F.shape[0], dtype=float)
    for front in fronts:
        crowding[front] = crowding_distance(F[front])
    finite_crowding = np.where(np.isfinite(crowding), crowding, np.max(crowding[np.isfinite(crowding)]) if np.isfinite(crowding).any() else 1.0)
    crowding_scale = finite_crowding / max(np.max(finite_crowding), 1.0e-12)
    scores = 1.0 / (1.0 + rank.astype(float)) + 0.25 * crowding_scale
    scores = np.clip(scores, 1.0e-12, None)
    scores /= scores.sum()
    return scores


def run_momvo_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    wep_min: float = 0.20,
    wep_max: float = 1.0,
    grid_count: int = 10,
    grid_inflation: float = 0.10,
    selection_pressure: float = 4.0,
    deletion_pressure: float = 2.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    span = hi - lo
    archive = ParetoArchive(capacity=max(archive_capacity, population_size * (iterations + 2)))
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        progress = t / max(iterations - 1, 1)
        wep = wep_min + progress * (wep_max - wep_min)
        tdr = 1.0 - progress ** (1.0 / 6.0)
        scores = _selection_scores(F)
        leader = archive.X[
            select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
            )
        ]

        X_next = X.copy()
        elite_index = int(np.argmin(F.sum(axis=1)))
        X_next[elite_index] = X[elite_index]
        for i in range(population_size):
            if i == elite_index:
                continue
            candidate = X[i].copy()
            for j in range(X.shape[1]):
                if rng.random() < scores[i]:
                    donor = X[rng.choice(population_size, p=scores)]
                    candidate[j] = donor[j]
                if rng.random() < wep:
                    shift = tdr * span[j] * rng.random()
                    if rng.random() < 0.5:
                        candidate[j] = leader[j] + shift
                    else:
                        candidate[j] = leader[j] - shift
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