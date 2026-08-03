"""Shared matplotlib style for the publication figure suite."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.figsize": (6.4, 4.0),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#888",
        "lines.linewidth": 1.8,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig: plt.Figure, out_dir: str, name: str, formats: Iterable[str] = ("pdf", "png")) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    return paths
