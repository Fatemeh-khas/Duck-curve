"""Publication-oriented figures for the paired outage Monte Carlo study."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


_COLORS = ("#8c8c8c", "#d97732", "#1679a7", "#2e8b57")


def plot_mcs_resilience_violins(
    case_rows: Mapping[str, Sequence[dict]],
):
    """Plot event ENS and served-load distributions for all MCS cases."""
    if not case_rows:
        raise ValueError("case_rows must not be empty")
    labels = list(case_rows)
    ens = [
        np.asarray(
            [row["unserved_energy_kwh"] for row in case_rows[label]],
            dtype=float,
        )
        for label in labels
    ]
    served = [
        np.asarray(
            [row["load_served_percent"] for row in case_rows[label]],
            dtype=float,
        )
        for label in labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    for ax, values, ylabel, title in (
        (
            axes[0],
            ens,
            "Unserved energy per outage event (kWh)",
            "Conditional outage consequence",
        ),
        (
            axes[1],
            served,
            "Affected load served (%)",
            "Conditional service survival",
        ),
    ):
        parts = ax.violinplot(
            values,
            positions=np.arange(1, len(labels) + 1),
            showmeans=True,
            showmedians=True,
            showextrema=True,
            widths=0.8,
        )
        for index, body in enumerate(parts["bodies"]):
            body.set_facecolor(_COLORS[index % len(_COLORS)])
            body.set_edgecolor("black")
            body.set_alpha(0.70)
        parts["cmeans"].set_color("#111111")
        parts["cmedians"].set_color("#ffffff")
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=18)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    return fig


def plot_paired_ens_differences(
    reference_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
):
    """Show the candidate-minus-reference paired ENS distribution."""
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise ValueError("paired cases must have equal non-zero lengths")
    differences = np.asarray(
        [
            candidate["unserved_energy_kwh"] - reference["unserved_energy_kwh"]
            for reference, candidate in zip(reference_rows, candidate_rows)
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.hist(differences, bins=45, color="#1679a7", alpha=0.78, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4, label="No effect")
    ax.axvline(
        float(np.mean(differences)),
        color="#d62728",
        linewidth=1.5,
        label="Mean paired effect",
    )
    ax.set_xlabel("Candidate minus reference ENS (kWh/event)")
    ax.set_ylabel("Scenario count")
    ax.set_title("Paired Monte Carlo outage effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig
