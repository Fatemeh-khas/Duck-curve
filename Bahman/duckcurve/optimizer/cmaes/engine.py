from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import record_archive_history, select_population


def run_sep_cmaes_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    sigma_init: float = 0.15,
    elite_fraction: float = 0.35,
    covariance_learning_rate: float = 0.30,
    mean_learning_rate: float = 0.70,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    F = np.asarray(evaluate(X), dtype=float)
    archive = ParetoArchive(capacity=archive_capacity)
    archive.add(X, F)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    span = hi - lo
    mu = max(2, min(population_size, int(round(elite_fraction * population_size))))
    mean = np.mean(X[:mu], axis=0)
    diag_std = np.maximum(sigma_init * span, 1.0e-3 * np.maximum(span, 1.0))

    for t in range(iterations):
        elite_X, elite_F, _, _ = select_population(X, F, mu)
        order = np.lexsort(tuple(elite_F[:, col] for col in range(elite_F.shape[1] - 1, -1, -1)))
        elite_X = elite_X[order]
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        target_mean = np.sum(weights[:, None] * elite_X, axis=0)
        mean = (1.0 - mean_learning_rate) * mean + mean_learning_rate * target_mean
        centered = elite_X - mean
        spread = np.sqrt(np.sum(weights[:, None] * centered * centered, axis=0) + 1.0e-12)
        diag_std = np.clip(
            (1.0 - covariance_learning_rate) * diag_std + covariance_learning_rate * spread,
            1.0e-3 * np.maximum(span, 1.0),
            0.50 * np.maximum(span, 1.0),
        )
        X = np.clip(mean + rng.normal(size=(population_size, X.shape[1])) * diag_std, lo, hi)
        keep = min(max(1, mu // 3), population_size)
        X[:keep] = elite_X[:keep]
        F = np.asarray(evaluate(X), dtype=float)
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