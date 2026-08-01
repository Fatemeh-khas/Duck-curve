"""Opposition-Based Learning initialization for a multi-objective population.

For each candidate ``x`` in ``[lo, hi]``, the opposite is
``x_opposite = lo + hi - x``.  The original and opposite populations are
combined, evaluated, and reduced to the requested population size using
non-dominated sorting followed by crowding distance within the last admitted
front.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .archive import crowding_distance


def opposite(population: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return the element-wise opposition of a population."""
    return lo + hi - population


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_rank(F: np.ndarray) -> np.ndarray:
    """Return Pareto ranks (0 is the first front)."""
    N = F.shape[0]
    ranks = np.full(N, -1, dtype=int)
    domination_count = np.zeros(N, dtype=int)
    dominates_list: list[list[int]] = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            if _dominates(F[i], F[j]):
                dominates_list[i].append(j)
                domination_count[j] += 1
            elif _dominates(F[j], F[i]):
                dominates_list[j].append(i)
                domination_count[i] += 1

    front = [i for i in range(N) if domination_count[i] == 0]
    rank = 0
    while front:
        next_front: list[int] = []
        for i in front:
            ranks[i] = rank
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1

    if np.any(ranks < 0):
        raise RuntimeError("non-dominated sorting failed to rank every candidate")
    return ranks


def apply_obl_initialization(
    initial_population: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the population and its opposite, retaining Pareto-diverse rows."""
    initial_population = np.asarray(initial_population, dtype=float)
    N, _ = initial_population.shape
    combined = np.vstack([initial_population, opposite(initial_population, lo, hi)])
    F = np.asarray(evaluate(combined), dtype=float)
    ranks = _non_dominated_rank(F)

    selected: list[int] = []
    for rank in range(int(ranks.max()) + 1):
        front = np.flatnonzero(ranks == rank)
        remaining = N - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front.tolist())
            continue

        distances = crowding_distance(F[front])
        # Stable random-free ordering gives reproducible initialization.
        order = np.argsort(-distances, kind="mergesort")
        selected.extend(front[order[:remaining]].tolist())
        break

    keep = np.asarray(selected, dtype=int)
    return combined[keep].copy(), F[keep].copy()
