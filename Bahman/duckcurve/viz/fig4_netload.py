"""Figure 4 — the canonical before/after net-load comparison.

This is the headline visual deliverable. Original = baseline net load (no BESS
optimization). Optimized = EZOA-selected Pareto operating point.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_figure_4(
    original_kw: np.ndarray,
    optimized_kw: np.ndarray,
    nl_no_pv_kw: Optional[np.ndarray] = None,
    title: str = "Duck Curve Minimization with EZOA-Placed PV + BESS",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Render the Figure 4 comparison.

    `original_kw` and `optimized_kw` are 24-element arrays (kW). If
    `nl_no_pv_kw` is given, the raw (no PV, no BESS) feeder load is drawn
    underneath as a light reference curve.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure

    T = len(original_kw)
    hours = np.arange(T)

    if nl_no_pv_kw is not None:
        ax.plot(hours, nl_no_pv_kw, color="#999999", linestyle=":", linewidth=1.4,
                label="No PV / No BESS")
    ax.plot(hours, original_kw, color="#C0392B", linestyle="--", linewidth=1.8,
            label="Original Net Load")
    ax.plot(hours, optimized_kw, color="#1F4FA0", linestyle="-", linewidth=2.0,
            label="Optimized Net Load")
    ax.axhline(0, color="#777", linewidth=0.6, alpha=0.7)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Net Load (kW)")
    ax.set_title(title)
    ax.set_xlim(0, T - 1)
    ax.set_xticks(np.arange(0, T, 5))
    ax.grid(True)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig
