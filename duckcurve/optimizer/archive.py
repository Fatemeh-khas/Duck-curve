from __future__ import annotations

from typing import Tuple

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def crowding_distance(F: np.ndarray) -> np.ndarray:
    N, M = F.shape
    if N <= 2:
        return np.full(N, np.inf)

    dist = np.zeros(N, dtype=float)
    for m in range(M):
        order = np.argsort(F[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        f_min, f_max = F[order[0], m], F[order[-1], m]
        if f_max - f_min < 1e-12:
            continue
        for i in range(1, N - 1):
            dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (f_max - f_min)
    return dist


class ParetoArchive:
    def __init__(self, capacity: int = 80):
        self.capacity = capacity
        self.X: np.ndarray = np.empty((0, 0))
        self.F: np.ndarray = np.empty((0, 0))

    def __len__(self) -> int:
        return self.X.shape[0]

    def add(self, X_new: np.ndarray, F_new: np.ndarray) -> None:
        if self.X.size == 0:
            self.X = np.asarray(X_new, dtype=float).copy()
            self.F = np.asarray(F_new, dtype=float).copy()
        else:
            self.X = np.vstack([self.X, np.asarray(X_new, dtype=float)])
            self.F = np.vstack([self.F, np.asarray(F_new, dtype=float)])

        self._dedupe_and_filter()
        if len(self) > self.capacity:
            self._prune()

    def _dedupe_and_filter(self) -> None:
        if len(self) == 0:
            return

        _, unique_idx = np.unique(np.round(self.X, decimals=6), axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        self.X = self.X[unique_idx]
        self.F = self.F[unique_idx]

        N = len(self)
        keep = np.ones(N, dtype=bool)
        for i in range(N):
            if not keep[i]:
                continue
            for j in range(N):
                if i == j or not keep[j]:
                    continue
                if dominates(self.F[j], self.F[i]):
                    keep[i] = False
                    break

        self.X = self.X[keep]
        self.F = self.F[keep]

    def _prune(self) -> None:
        dist = crowding_distance(self.F)
        order = np.argsort(-dist)
        keep = order[: self.capacity]
        self.X = self.X[keep]
        self.F = self.F[keep]

    def sample_elite(self, rng: np.random.Generator) -> np.ndarray:
        if len(self) == 0:
            raise RuntimeError("archive is empty")
        return self.X[rng.integers(0, len(self))]

    def best_by_objective(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        i = int(np.argmin(self.F[:, m]))
        return self.X[i], self.F[i]