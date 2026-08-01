"""Publication-style SOC trajectories for independently optimized BESS units."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_soc(
    soc_mwh: np.ndarray,
    soc_min_mwh: float | None = None,
    soc_max_mwh: float | None = None,
    unit_labels: list[str] | None = None,
    dispatch_mw: np.ndarray | None = None,
) -> plt.Figure:
    """Plot the actual per-unit SOC trajectories in one publication-style panel.

    ``dispatch_mw`` is retained for backward compatibility but is deliberately
    not drawn: the requested paper figure contains SOC trajectories only.
    """
    del dispatch_mw
    soc_mwh = np.asarray(soc_mwh, dtype=float)
    n_bess, n_soc_points = soc_mwh.shape
    palette = ["#1F4FA0", "#C0392B", "#2E8B57", "#D2A106", "#7E57C2", "#0BA5A4"]
    labels = unit_labels if unit_labels is not None else [f"BESS {k + 1}" for k in range(n_bess)]

    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    hours = np.arange(n_soc_points)
    for k in range(n_bess):
        ax.plot(
            hours,
            soc_mwh[k] * 1000.0,
            color=palette[k % len(palette)],
            linewidth=2.7,
            label=labels[k],
        )

    if soc_min_mwh is not None:
        ax.axhline(
            soc_min_mwh * 1000.0,
            color="#777777",
            linestyle="--",
            linewidth=1.6,
            label="SOC min",
        )
    if soc_max_mwh is not None:
        ax.axhline(
            soc_max_mwh * 1000.0,
            color="#777777",
            linestyle=":",
            linewidth=1.6,
            label="SOC max",
        )

    limit_values = [float(np.min(soc_mwh) * 1000.0), float(np.max(soc_mwh) * 1000.0)]
    if soc_min_mwh is not None:
        limit_values.append(float(soc_min_mwh * 1000.0))
    if soc_max_mwh is not None:
        limit_values.append(float(soc_max_mwh * 1000.0))
    span = max(limit_values) - min(limit_values)
    pad = max(70.0, 0.055 * span)

    ax.set_xlim(0, n_soc_points - 1)
    ax.set_ylim(min(limit_values) - pad, max(limit_values) + pad)
    ax.set_xticks(np.arange(0, n_soc_points, 3))
    ax.set_xlabel("Time (h)", fontsize=13)
    ax.set_ylabel("SOC (kWh)", fontsize=13)
    ax.set_title("Hourly SOC of All BESS Units", fontsize=19, pad=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(
        loc="upper center",
        ncol=min(n_bess + 2, 5),
        fontsize=10.5,
        framealpha=0.94,
        edgecolor="#888888",
    )
    fig.tight_layout()
    return fig