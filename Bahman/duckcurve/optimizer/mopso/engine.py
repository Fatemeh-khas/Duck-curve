from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive, crowding_distance, dominates
from ..common import OptimizerResult
from ..moo import record_archive_history


def _select_leader_batch(archive: ParetoArchive, rng: np.random.Generator, count: int) -> np.ndarray:
    if len(archive) == 0:
        raise RuntimeError("archive is empty")
    if len(archive) == 1:
        return np.repeat(archive.X, count, axis=0)
    dist = crowding_distance(archive.F)
    finite = np.isfinite(dist)
    fallback = dist[finite].max() if finite.any() else 1.0
    weights = np.where(finite, dist, fallback)
    weights = weights - weights.min() + 1.0e-6
    probs = weights / weights.sum()
    idx = rng.choice(len(archive), size=count, p=probs)
    return archive.X[idx]


def run_mopso_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    inertia: float = 0.55,
    cognitive_coeff: float = 1.60,
    social_coeff: float = 1.60,
    mutation_prob: float = 0.05,
    velocity_clamp: float = 0.25,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    span = hi - lo
    V = rng.normal(0.0, 0.10 * span, size=X.shape)
    F = np.asarray(evaluate(X), dtype=float)
    pbest_X = X.copy()
    pbest_F = F.copy()
    archive = ParetoArchive(capacity=archive_capacity)
    archive.add(X, F)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        leaders = _select_leader_batch(archive, rng, population_size)
        r1 = rng.random(X.shape)
        r2 = rng.random(X.shape)
        V = inertia * V + cognitive_coeff * r1 * (pbest_X - X) + social_coeff * r2 * (leaders - X)
        vmax = velocity_clamp * span
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        if mutation_prob > 0.0:
            progress = t / max(iterations - 1, 1)
            mask = rng.random(X.shape) < mutation_prob * (1.0 - 0.5 * progress)
            X = np.clip(X + mask * rng.normal(0.0, 0.05 * span, size=X.shape), lo, hi)
        F = np.asarray(evaluate(X), dtype=float)

        for i in range(population_size):
            if dominates(F[i], pbest_F[i]) or (
                not dominates(pbest_F[i], F[i]) and rng.random() < 0.5
            ):
                pbest_X[i] = X[i]
                pbest_F[i] = F[i]

        archive.add(X, F)
        archive.add(pbest_X, pbest_F)
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