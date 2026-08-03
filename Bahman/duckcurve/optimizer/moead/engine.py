from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import random_weight_vectors, record_archive_history, tchebycheff


def run_moead_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    neighborhood_size: int = 15,
    differential_weight: float = 0.55,
    crossover_rate: float = 0.90,
    neighborhood_mating_prob: float = 0.90,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    F = np.asarray(evaluate(X), dtype=float)
    weights = random_weight_vectors(population_size, F.shape[1], rng)
    neighborhood_size = min(population_size, max(2, int(neighborhood_size)))
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :neighborhood_size]
    ideal = np.min(F, axis=0)
    archive = ParetoArchive(capacity=archive_capacity)
    archive.add(X, F)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        for i in range(population_size):
            pool = neighborhoods[i] if rng.random() < neighborhood_mating_prob else np.arange(population_size)
            replace = len(pool) < 3
            r1, r2, r3 = rng.choice(pool, size=3, replace=replace)
            mutant = np.clip(X[r1] + differential_weight * (X[r2] - X[r3]), lo, hi)
            mask = rng.random(X.shape[1]) < crossover_rate
            mask[rng.integers(0, X.shape[1])] = True
            trial = np.where(mask, mutant, X[i])
            trial = np.clip(trial, lo, hi)
            f_trial = np.asarray(evaluate(trial[None, :]), dtype=float)[0]
            ideal = np.minimum(ideal, f_trial)
            trial_score = tchebycheff(np.tile(f_trial, (len(neighborhoods[i]), 1)), weights[neighborhoods[i]], ideal)
            current_score = tchebycheff(F[neighborhoods[i]], weights[neighborhoods[i]], ideal)
            better = trial_score <= current_score
            if np.any(better):
                targets = neighborhoods[i][better]
                X[targets] = trial
                F[targets] = f_trial
            archive.add(trial[None, :], f_trial[None, :])
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