from __future__ import annotations

from typing import Callable

import numpy as np

from ..archive import ParetoArchive
from ..common import OptimizerResult
from ..moo import (
    binary_tournament_indices,
    ensure_even_population,
    polynomial_mutation,
    record_archive_history,
    select_population,
    simulated_binary_crossover,
)


def run_nsga2_engine(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    X0: np.ndarray,
    population_size: int = 80,
    iterations: int = 180,
    archive_capacity: int = 100,
    seed: int = 42,
    verbose: bool = True,
    crossover_prob: float = 0.9,
    mutation_prob: float | None = None,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    population_size = int(population_size)
    mating_size = ensure_even_population(population_size)
    mutation_prob = (1.0 / max(1, lo.shape[0])) if mutation_prob is None else float(mutation_prob)

    X = np.asarray(X0[:population_size], dtype=float).copy()
    F = np.asarray(evaluate(X), dtype=float)
    archive = ParetoArchive(capacity=archive_capacity)
    archive.add(X, F)

    history_best = []
    history_hv = []
    ref_point = record_archive_history(archive.F, history_best, history_hv, None)

    for t in range(iterations):
        _, _, rank, crowding = select_population(X, F, population_size)
        parent_idx = binary_tournament_indices(rng, rank, crowding, mating_size)
        parents = X[parent_idx]

        children = []
        for i in range(0, mating_size, 2):
            child_a, child_b = simulated_binary_crossover(
                parents[i],
                parents[i + 1],
                lo,
                hi,
                rng,
                crossover_prob=crossover_prob,
                eta=eta_crossover,
            )
            children.extend((child_a, child_b))

        X_child = polynomial_mutation(
            np.asarray(children[:population_size], dtype=float),
            lo,
            hi,
            rng,
            mutation_prob=mutation_prob,
            eta=eta_mutation,
        )
        F_child = np.asarray(evaluate(X_child), dtype=float)
        X, F, _, _ = select_population(
            np.vstack((X, X_child)),
            np.vstack((F, F_child)),
            population_size,
        )
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