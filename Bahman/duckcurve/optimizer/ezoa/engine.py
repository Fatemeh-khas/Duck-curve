"""Multi-Objective Enhanced Zebra Optimization Algorithm (MO-EZOA).

This module holds the *optimizer engine* only: the population update rules
(standard ZOA foraging + defense phases), opposition-based learning (OBL)
initialization, and an external Pareto archive used both as the algorithm's
elite memory and as the source of leader guidance (crowding-aware, so search
pressure spreads across the whole front instead of collapsing onto one
solution).

The experiment-specific pipeline (objective wiring, penalty terms, knee-point
selection, etc.) lives in ``duckcurve.optimizer.ezoa.pipeline`` and imports the
``EZOA`` / ``OptimizerResult`` symbols from here.

Convergence-quality notes (2026-07-22 revision)
------------------------------------------------
An earlier version of this file plateaued after ~40-60 of 155 iterations with
an archive of only 4-9 points. Three genuine algorithmic causes were found
and fixed here -- none of them involve steering the search toward a
pre-chosen answer, they just remove artificial barriers to exploration:

1. The defense phase's shrinking random-walk term accidentally applied its
   ``(1 - t/T)`` decay factor twice (once inside ``R`` and again in the
   update itself), so by mid-run the perturbation had collapsed to numerical
   noise. Fixed to apply the decay once.
2. That same term scaled the perturbation by the candidate's *own value*
   (``... * X[i]``), a known weakness of vanilla ZOA: once a decision
   variable (e.g. a BESS dispatch value) approaches zero, it stops being
   perturbed at all. Replaced with perturbation scaled by the variable's
   actual bound range, so every dimension keeps a meaningful step size
   throughout the run.
3. Candidates were only ever accepted if they *strictly dominated* the
   incumbent in both objectives simultaneously. That rejects every move that
   trades a little of one objective for a lot of the other, which is exactly
   how a population should spread out along a Pareto front -- so the search
   was structurally prevented from ever finding more than a small cluster of
   solutions near wherever it first got lucky. Genuine trade-off moves
   (mutually non-dominated) are now accepted with a tunable probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..archive import ParetoArchive, crowding_distance, dominates
from ..obl import apply_obl_initialization


@dataclass
class OptimizerResult:
    archive: ParetoArchive
    history_best_per_obj: List[np.ndarray]
    history_hypervolume: List[float] = field(default_factory=list)


EZOAResult = OptimizerResult


def _hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """2-objective hypervolume against a dominated reference point (both
    objectives minimized). Returns 0 for an empty archive."""
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


class EZOA:
    """Multi-objective Enhanced Zebra Optimization Algorithm.

    Parameters
    ----------
    evaluate:
        Batch objective function mapping an (N, dim) population to an
        (N, n_obj) objective matrix. All objectives are minimized.
    lo, hi:
        Per-dimension lower/upper bounds, shape (dim,).
    population_size, iterations, archive_capacity, obl_init, seed:
        Standard metaheuristic controls.
    defense_step:
        Base random-walk coefficient for the S1 (flee) defense phase,
        relative to each dimension's bound range. Larger values keep more
        exploratory pressure throughout the run.
    tradeoff_accept_prob:
        Probability of accepting a candidate that is mutually
        non-dominated with the incumbent (neither strictly better nor
        worse). This is what lets the population spread along the Pareto
        front instead of collapsing onto whichever point it reaches first.
    """

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], np.ndarray],
        lo: np.ndarray,
        hi: np.ndarray,
        population_size: int = 80,
        iterations: int = 180,
        archive_capacity: int = 100,
        obl_init: bool = True,
        seed: int = 42,
        defense_step: float = 0.15,
        tradeoff_accept_prob: float = 0.35,
        primary_elite_trials: int = 20,
        primary_elite_step: float = 0.10,
    ):
        self.evaluate = evaluate
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.range = np.maximum(self.hi - self.lo, 1e-12)
        self.dim = self.lo.shape[0]
        self.population_size = population_size
        self.iterations = iterations
        self.obl_init = obl_init
        self.defense_step = defense_step
        self.tradeoff_accept_prob = tradeoff_accept_prob
        self.primary_elite_trials = max(0, int(primary_elite_trials))
        self.primary_elite_step = max(0.0, float(primary_elite_step))
        self.rng = np.random.default_rng(seed)

        self.archive = ParetoArchive(capacity=archive_capacity)

        self.X = self.rng.uniform(self.lo, self.hi, size=(population_size, self.dim))
        self.F: Optional[np.ndarray] = None

        if self.obl_init:
            X_sel, F_sel = apply_obl_initialization(self.X, self.lo, self.hi, self.evaluate)
            self.X = X_sel
            self.F = F_sel

    def _select_leader(self) -> np.ndarray:
        """Pick a leader biased towards low-crowding (isolated, diverse)
        archive members, so search pressure spreads across the front rather
        than collapsing onto a single point."""
        if len(self.archive) == 0:
            idx = self.rng.integers(0, self.population_size)
            return self.X[idx].copy()

        dist = crowding_distance(self.archive.F)
        finite = np.isfinite(dist)
        if finite.any():
            fallback = dist[finite].max() if finite.any() else 1.0
            weights = np.where(finite, dist, fallback)
            weights = weights - weights.min() + 1e-6
            probs = weights / weights.sum()
            idx = self.rng.choice(len(self.archive), p=probs)
        else:
            idx = self.rng.integers(0, len(self.archive))
        return self.archive.X[idx].copy()

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo, self.hi)

    def _accept(self, f_new: np.ndarray, f_old: np.ndarray) -> bool:
        """Accept a candidate if it dominates the incumbent outright, or --
        with `tradeoff_accept_prob` -- if it's a genuine trade-off (neither
        dominates the other). Rejects candidates that are simply worse.
        """
        if dominates(f_new, f_old):
            return True
        if dominates(f_old, f_new):
            return False
        return bool(self.rng.random() < self.tradeoff_accept_prob)

    def _polish_primary_elite(self) -> None:
        """Deterministically polish the minimum-first-objective solution.

        Each sweep evaluates positive and negative coordinate moves in one
        batch and keeps the best strict improvement. Failed sweeps halve the
        step. The original MO-EZOA trajectory is left unchanged.
        """
        if self.primary_elite_trials == 0 or len(self.archive) == 0:
            return

        best_idx = int(np.lexsort((self.archive.F[:, 1], self.archive.F[:, 0]))[0])
        centre = self.archive.X[best_idx].copy()
        best_f = self.archive.F[best_idx].copy()
        step = self.primary_elite_step * self.range
        eye = np.eye(self.dim)

        for _ in range(self.primary_elite_trials):
            trials = self._clip(
                np.vstack((centre[None, :] + eye * step, centre[None, :] - eye * step))
            )
            trial_f = np.asarray(self.evaluate(trials), dtype=float)
            candidate_idx = int(np.lexsort((trial_f[:, 1], trial_f[:, 0]))[0])

            if tuple(trial_f[candidate_idx]) < tuple(best_f):
                centre = trials[candidate_idx]
                best_f = trial_f[candidate_idx]
                self.archive.add(centre[None, :], best_f[None, :])
            else:
                step *= 0.5

            if float(np.max(step / self.range)) < 1.0e-4:
                break

    def run(
        self,
        on_iter: Optional[Callable[[int, OptimizerResult], None]] = None,
    ) -> OptimizerResult:
        self.F = np.asarray(self.evaluate(self.X), dtype=float)
        self.archive.add(self.X, self.F)

        history_best: List[np.ndarray] = []
        history_hv: List[float] = []
        ref_point: Optional[np.ndarray] = None

        for t in range(self.iterations):
            progress = t / max(self.iterations - 1, 1)

            for i in range(self.population_size):
                pioneer = self._select_leader()
                I = self.rng.choice([1, 2])
                r = self.rng.random(self.dim)
                foraging = self._clip(self.X[i] + r * (pioneer - I * self.X[i]))
                f_forage = np.asarray(self.evaluate(foraging[None, :]), dtype=float)[0]
                if self._accept(f_forage, self.F[i]):
                    self.X[i] = foraging
                    self.F[i] = f_forage

                if self.rng.random() < 0.5:
                    step = self.defense_step * (1.0 - progress)
                    defense = self.X[i] + step * (2 * self.rng.random(self.dim) - 1) * self.range
                else:
                    other = self._select_leader()
                    I = self.rng.choice([1, 2])
                    r = self.rng.random(self.dim)
                    defense = self.X[i] + r * (other - I * self.X[i])

                defense = self._clip(defense)
                f_defense = np.asarray(self.evaluate(defense[None, :]), dtype=float)[0]
                if self._accept(f_defense, self.F[i]):
                    self.X[i] = defense
                    self.F[i] = f_defense

            self.archive.add(self.X, self.F)

            best_per_obj = np.array([self.archive.F[:, m].min() for m in range(self.archive.F.shape[1])])
            history_best.append(best_per_obj)

            if self.archive.F.shape[1] == 2:
                if ref_point is None:
                    ref_point = self.archive.F.max(axis=0) * 1.1 + 1e-9
                else:
                    ref_point = np.maximum(ref_point, self.archive.F.max(axis=0) * 1.1 + 1e-9)
                history_hv.append(_hypervolume_2d(self.archive.F, ref_point))

            if on_iter is not None:
                on_iter(
                    t,
                    OptimizerResult(
                        archive=self.archive,
                        history_best_per_obj=history_best,
                        history_hypervolume=history_hv,
                    ),
                )

        self._polish_primary_elite()
        return OptimizerResult(
            archive=self.archive,
            history_best_per_obj=history_best,
            history_hypervolume=history_hv,
        )