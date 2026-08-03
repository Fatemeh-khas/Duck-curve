from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .archive import ParetoArchive, crowding_distance, dominates


def record_archive_history(
    archive_F: np.ndarray,
    history_best: List[np.ndarray],
    history_hv: List[float],
    ref_point: np.ndarray | None,
) -> np.ndarray | None:
    if archive_F.size == 0:
        return ref_point
    history_best.append(np.min(archive_F, axis=0))
    if archive_F.shape[1] == 2:
        if ref_point is None:
            ref_point = archive_F.max(axis=0) * 1.1 + 1.0e-9
        else:
            ref_point = np.maximum(ref_point, archive_F.max(axis=0) * 1.1 + 1.0e-9)
        history_hv.append(hypervolume_2d(archive_F, ref_point))
    return ref_point


def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    if len(F) == 0:
        return 0.0
    pts = F[np.argsort(F[:, 0])]
    hv = 0.0
    prev_f1 = ref[1]
    for f0, f1 in pts:
        if f1 < prev_f1:
            hv += (ref[0] - f0) * (prev_f1 - f1)
            prev_f1 = f1
    return float(max(hv, 0.0))


def non_dominated_sort(F: np.ndarray) -> tuple[List[np.ndarray], np.ndarray]:
    n_points = F.shape[0]
    dominates_set = [set() for _ in range(n_points)]
    dominated_count = np.zeros(n_points, dtype=int)
    fronts: List[List[int]] = [[]]

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dominates(F[i], F[j]):
                dominates_set[i].add(j)
                dominated_count[j] += 1
            elif dominates(F[j], F[i]):
                dominates_set[j].add(i)
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    rank = np.full(n_points, -1, dtype=int)
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: List[int] = []
        for index in fronts[current]:
            rank[index] = current
            for dominated_index in dominates_set[index]:
                dominated_count[dominated_index] -= 1
                if dominated_count[dominated_index] == 0:
                    next_front.append(dominated_index)
        if next_front:
            fronts.append(next_front)
        current += 1

    return [np.asarray(front, dtype=int) for front in fronts if front], rank


def select_population(
    X: np.ndarray,
    F: np.ndarray,
    population_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fronts, rank = non_dominated_sort(F)
    selected: List[int] = []
    crowding = np.zeros(F.shape[0], dtype=float)

    for front in fronts:
        dist = crowding_distance(F[front])
        crowding[front] = dist
        if len(selected) + len(front) <= population_size:
            selected.extend(front.tolist())
            continue
        keep = front[np.argsort(-dist)[: population_size - len(selected)]]
        selected.extend(keep.tolist())
        break

    indices = np.asarray(selected, dtype=int)
    return X[indices].copy(), F[indices].copy(), rank, crowding


def binary_tournament_indices(
    rng: np.random.Generator,
    rank: np.ndarray,
    crowding: np.ndarray,
    count: int,
) -> np.ndarray:
    chosen = np.empty(count, dtype=int)
    size = rank.shape[0]
    for i in range(count):
        a, b = rng.integers(0, size, size=2)
        if rank[a] < rank[b]:
            chosen[i] = a
        elif rank[b] < rank[a]:
            chosen[i] = b
        elif crowding[a] > crowding[b]:
            chosen[i] = a
        elif crowding[b] > crowding[a]:
            chosen[i] = b
        else:
            chosen[i] = a if rng.random() < 0.5 else b
    return chosen


def simulated_binary_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
    crossover_prob: float,
    eta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if rng.random() >= crossover_prob:
        return child_a, child_b

    for j in range(parent_a.shape[0]):
        if rng.random() > 0.5 or abs(parent_a[j] - parent_b[j]) < 1.0e-12:
            continue
        y1 = min(parent_a[j], parent_b[j])
        y2 = max(parent_a[j], parent_b[j])
        lower = lo[j]
        upper = hi[j]
        rand = rng.random()

        beta = 1.0 + 2.0 * (y1 - lower) / max(y2 - y1, 1.0e-12)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        beta = 1.0 + 2.0 * (upper - y2) / max(y2 - y1, 1.0e-12)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        c1 = float(np.clip(c1, lower, upper))
        c2 = float(np.clip(c2, lower, upper))
        if rng.random() <= 0.5:
            child_a[j], child_b[j] = c2, c1
        else:
            child_a[j], child_b[j] = c1, c2
    return child_a, child_b


def polynomial_mutation(
    X: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
    mutation_prob: float,
    eta: float,
) -> np.ndarray:
    Y = np.asarray(X, dtype=float).copy()
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if rng.random() >= mutation_prob:
                continue
            span = hi[j] - lo[j]
            if span <= 1.0e-12:
                continue
            x = Y[i, j]
            delta1 = (x - lo[j]) / span
            delta2 = (hi[j] - x) / span
            rand = rng.random()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - val ** mut_pow
            Y[i, j] = float(np.clip(x + delta_q * span, lo[j], hi[j]))
    return Y


def random_weight_vectors(population_size: int, n_obj: int, rng: np.random.Generator) -> np.ndarray:
    if n_obj == 2:
        grid = np.linspace(0.0, 1.0, population_size)
        weights = np.column_stack((grid, 1.0 - grid))
    else:
        weights = rng.dirichlet(np.ones(n_obj), size=population_size)
    weights = np.clip(weights, 1.0e-6, None)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def tchebycheff(values: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    return np.max(weights * np.abs(values - ideal), axis=-1)


def ensure_even_population(population_size: int) -> int:
    return population_size if population_size % 2 == 0 else population_size + 1


def grid_subindices(
    F: np.ndarray,
    grid_count: int,
    grid_inflation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    span = np.maximum(f_max - f_min, 1.0e-12)
    lower = f_min - grid_inflation * span
    upper = f_max + grid_inflation * span
    scaled = (F - lower) / np.maximum(upper - lower, 1.0e-12)
    scaled = np.clip(scaled, 0.0, 1.0 - 1.0e-12)
    subindices = np.floor(grid_count * scaled).astype(int)
    return subindices, lower, upper


def grid_cell_ids(subindices: np.ndarray) -> np.ndarray:
    return np.asarray(
        ["|".join(str(int(value)) for value in row) for row in subindices],
        dtype=object,
    )


def select_archive_member_by_grid(
    archive: ParetoArchive,
    rng: np.random.Generator,
    grid_count: int,
    grid_inflation: float,
    selection_pressure: float,
    excluded: set[int] | None = None,
) -> int:
    if len(archive) == 0:
        raise RuntimeError("archive is empty")
    excluded = excluded or set()
    candidates = np.asarray([i for i in range(len(archive)) if i not in excluded], dtype=int)
    if candidates.size == 0:
        candidates = np.arange(len(archive), dtype=int)

    subindices, _, _ = grid_subindices(archive.F[candidates], grid_count, grid_inflation)
    cells = grid_cell_ids(subindices)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    probabilities = counts.astype(float) ** (-selection_pressure)
    probabilities /= probabilities.sum()
    chosen_cell = rng.choice(np.arange(counts.size), p=probabilities)
    members = candidates[inverse == chosen_cell]
    return int(rng.choice(members))


def prune_archive_by_grid(
    archive: ParetoArchive,
    rng: np.random.Generator,
    archive_capacity: int,
    grid_count: int,
    grid_inflation: float,
    deletion_pressure: float,
) -> None:
    while len(archive) > archive_capacity:
        subindices, _, _ = grid_subindices(archive.F, grid_count, grid_inflation)
        cells = grid_cell_ids(subindices)
        _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
        probabilities = counts.astype(float) ** deletion_pressure
        probabilities /= probabilities.sum()
        crowded_cell = rng.choice(np.arange(counts.size), p=probabilities)
        members = np.flatnonzero(inverse == crowded_cell)
        remove_index = int(rng.choice(members))
        keep = np.ones(len(archive), dtype=bool)
        keep[remove_index] = False
        archive.X = archive.X[keep]
        archive.F = archive.F[keep]


def select_archive_leaders_by_grid(
    archive: ParetoArchive,
    rng: np.random.Generator,
    grid_count: int,
    grid_inflation: float,
    selection_pressure: float,
    n_leaders: int,
) -> list[np.ndarray]:
    if len(archive) == 0:
        raise RuntimeError("archive is empty")
    if len(archive) == 1:
        return [archive.X[0].copy() for _ in range(n_leaders)]

    excluded: set[int] = set()
    leaders: list[np.ndarray] = []
    for _ in range(n_leaders):
        idx = select_archive_member_by_grid(
            archive,
            rng,
            grid_count=grid_count,
            grid_inflation=grid_inflation,
            selection_pressure=selection_pressure,
            excluded=excluded,
        )
        excluded.add(idx)
        leaders.append(archive.X[idx].copy())
    return leaders