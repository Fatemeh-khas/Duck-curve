from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def _levy_step(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    beta = 1.5
    sigma = (
        math.gamma(1.0 + beta)
        * np.sin(np.pi * beta / 2.0)
        / (
            math.gamma((1.0 + beta) / 2.0)
            * beta
            * 2.0 ** ((beta - 1.0) / 2.0)
        )
    ) ** (1.0 / beta)
    u = rng.normal(0.0, sigma, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    return u / (np.abs(v) ** (1.0 / beta) + 1.0e-12)


def run_moda_engine(
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
    neighborhood_scale_min: float = 0.10,
    neighborhood_scale_max: float = 1.00,
    delta_clamp: float = 0.10,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    span = hi - lo
    delta = np.zeros_like(X)
    archive = ParetoArchive(capacity=max(archive_capacity, population_size * (iterations + 2)))
    F = np.asarray(evaluate(X), dtype=float)
    archive.add(X, F)
    prune_archive_by_grid(archive, rng, archive_capacity, grid_count, grid_inflation, deletion_pressure)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        progress = t / max(iterations - 1, 1)
        w = 0.9 - progress * 0.5
        my_c = max(0.0, 0.1 - progress * 0.1 * 2.0)
        s_coeff = 2.0 * rng.random() * my_c
        a_coeff = 2.0 * rng.random() * my_c
        c_coeff = 2.0 * rng.random() * my_c
        f_coeff = 2.0 * rng.random()
        e_coeff = my_c
        radius = span * (neighborhood_scale_min + (neighborhood_scale_max - neighborhood_scale_min) * progress)
        food = archive.X[
            select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=selection_pressure,
            )
        ]
        enemy = archive.X[
            select_archive_member_by_grid(
                archive,
                rng,
                grid_count=grid_count,
                grid_inflation=grid_inflation,
                selection_pressure=-deletion_pressure,
            )
        ]

        X_next = np.empty_like(X)
        for i in range(population_size):
            diff = X - X[i]
            neighbor_mask = np.all(np.abs(diff) <= radius, axis=1)
            neighbor_mask[i] = False
            neighbors = np.flatnonzero(neighbor_mask)
            if neighbors.size == 0:
                step = 0.01 * span * _levy_step(rng, (X.shape[1],))
                delta[i] = 0.5 * delta[i] + step
                X_next[i] = np.clip(X[i] + delta[i], lo, hi)
                continue

            neighbor_diff = diff[neighbors]
            separation = -np.sum(neighbor_diff, axis=0)
            alignment = np.mean(delta[neighbors], axis=0)
            cohesion = np.mean(X[neighbors], axis=0) - X[i]
            food_attraction = food - X[i]
            enemy_distraction = X[i] - enemy
            delta[i] = (
                w * delta[i]
                + s_coeff * separation
                + a_coeff * alignment
                + c_coeff * cohesion
                + f_coeff * food_attraction
                + e_coeff * enemy_distraction
            )
            delta[i] = np.clip(delta[i], -delta_clamp * span, delta_clamp * span)
            X_next[i] = np.clip(X[i] + delta[i], lo, hi)

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