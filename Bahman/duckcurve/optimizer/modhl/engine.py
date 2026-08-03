from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import prune_archive_by_grid, record_archive_history, select_archive_member_by_grid


def _lexicographic_order(F: np.ndarray) -> np.ndarray:
    keys = tuple(F[:, col] for col in range(F.shape[1] - 1, -1, -1))
    return np.lexsort(keys)


def _mirror_bounds(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    Y = np.asarray(X, dtype=float).copy()
    span = hi - lo
    active = span > 1.0e-12
    if np.any(active):
        reflected = np.mod(Y[..., active] - lo[active], 2.0 * span[active])
        Y[..., active] = lo[active] + np.where(
            reflected <= span[active],
            reflected,
            2.0 * span[active] - reflected,
        )
    if np.any(~active):
        Y[..., ~active] = lo[~active]
    return Y


def _initialize_leaders(
    rng: np.random.Generator,
    leader_pool_size: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    leaders = rng.uniform(lo, hi, size=(leader_pool_size, lo.shape[0]))
    if leader_pool_size >= 1:
        leaders[0] = lo.copy()
    if leader_pool_size >= 2:
        leaders[-1] = hi.copy()
    return leaders


def _leader_count_schedule(version: str, iteration: int, iterations: int, leader_pool_size: int) -> int:
    progress_index = float(iteration)
    if version == "V1":
        aa = 1.0 - progress_index / max(iterations / 2.0, 1.0)
        if aa > 0.0:
            return max(1, 3 + round(aa * (leader_pool_size - 3)))
        aa = 1.0 - progress_index / max(float(iterations), 1.0)
        return max(1, 1 + round(aa * 4.0))
    if version == "V2":
        aa = 1.0 - progress_index / max(float(iterations), 1.0)
        return max(1, 1 + round(aa * (leader_pool_size - 1)))
    if version == "V3":
        aa = -(progress_index + 1.0) * 10.0 / max(float(iterations), 1.0)
        return max(1, round(leader_pool_size * np.exp(aa)) + 1)
    return leader_pool_size


def _select_leader_pool(
    archive: ParetoArchive,
    rng: np.random.Generator,
    leader_pool_size: int,
    grid_count: int,
    grid_inflation: float,
    selection_pressure: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(archive) == 0:
        raise RuntimeError("archive is empty")

    indices: list[int] = []
    excluded: set[int] = set()
    while len(indices) < leader_pool_size:
        idx = select_archive_member_by_grid(
            archive,
            rng,
            grid_count=grid_count,
            grid_inflation=grid_inflation,
            selection_pressure=selection_pressure,
            excluded=(excluded if len(excluded) < len(archive) else None),
        )
        indices.append(idx)
        excluded.add(idx)
        if len(excluded) == len(archive):
            excluded.clear()

    idx_array = np.asarray(indices, dtype=int)
    order = _lexicographic_order(archive.F[idx_array])
    idx_array = idx_array[order]
    return archive.X[idx_array].copy(), archive.F[idx_array].copy()


def run_modhl_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    leader_pool_size: int = 10,
    version: str = "V4",
    leader_tol_window_pct: float = 5.0,
    leader_tol_abs: float = 1.0e-5,
    grid_count: int = 10,
    grid_inflation: float = 0.10,
    selection_pressure: float = 4.0,
    deletion_pressure: float = 2.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X0[:population_size], dtype=float).copy()
    archive_store_capacity = max(archive_capacity, population_size * (iterations + 2) + leader_pool_size)
    archive = ParetoArchive(capacity=archive_store_capacity)

    leader_pool_size = max(1, int(leader_pool_size))
    initial_leaders = _initialize_leaders(rng, leader_pool_size, lo, hi)
    archive.add(initial_leaders, np.asarray(evaluate(initial_leaders), dtype=float))

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

    history_best: list[np.ndarray] = []
    history_hv: list[float] = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    tol_window = max(1, round(iterations * leader_tol_window_pct / 100.0))
    active_leader_count = leader_pool_size
    worst_leader_history: list[float] = []

    for t in range(iterations):
        a = 2.0 - t * (2.0 / max(float(iterations), 1.0))
        leader_positions, leader_objectives = _select_leader_pool(
            archive,
            rng,
            leader_pool_size=leader_pool_size,
            grid_count=grid_count,
            grid_inflation=grid_inflation,
            selection_pressure=selection_pressure,
        )

        if version in {"V1", "V2", "V3"}:
            active_leader_count = min(
                leader_pool_size,
                _leader_count_schedule(version, t, iterations, leader_pool_size),
            )
        else:
            worst_leader_history.append(float(leader_objectives[active_leader_count - 1, 0]))
            if t + 1 > tol_window + 1:
                current = worst_leader_history[-1]
                previous = worst_leader_history[-1 - tol_window]
                if current >= previous - leader_tol_abs:
                    active_leader_count -= 1
            active_leader_count = max(1, min(active_leader_count, leader_pool_size))

        active_leaders = leader_positions[:active_leader_count]
        X_next = np.empty_like(X)
        for i in range(population_size):
            proposals = np.empty((active_leader_count, X.shape[1]), dtype=float)
            for leader_idx in range(active_leader_count):
                r1 = rng.random(X.shape[1])
                r2 = rng.random(X.shape[1])
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                distance = np.abs(C1 * active_leaders[leader_idx] - X[i])
                proposals[leader_idx] = active_leaders[leader_idx] - A1 * distance
            X_next[i] = proposals.mean(axis=0)

        X = _mirror_bounds(X_next, lo, hi)
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