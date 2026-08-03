"""2D Pareto front and EZOA convergence curves."""
from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front(
    F: np.ndarray,
    highlighted: Optional[int] = None,
    obj_names: Tuple[str, str] = (
        "Duck-curve SSS (kW squared)",
        "Daily feeder energy loss (kWh)",
    ),
) -> plt.Figure:
    """F has shape (N, 2). `highlighted` is the index of the chosen operating point."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    order = np.argsort(F[:, 0])
    F_sorted = F[order]
    ax.plot(F_sorted[:, 0], F_sorted[:, 1], color="#1F4FA0", linewidth=1.4, alpha=0.85)
    ax.scatter(F[:, 0], F[:, 1], s=22, color="#1F4FA0", edgecolor="white", linewidth=0.6,
               zorder=3, label="Pareto archive")
    if highlighted is not None:
        ax.scatter([F[highlighted, 0]], [F[highlighted, 1]], s=110, marker="*",
                   color="#C0392B", edgecolor="white", linewidth=0.8, zorder=4,
                   label="Selected operating point")
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Pareto front — EZOA")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_convergence(history_hv: List[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(history_hv, color="#1F4FA0", linewidth=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hypervolume")
    ax.set_title("EZOA convergence")
    ax.grid(True)
    fig.tight_layout()
    return fig
