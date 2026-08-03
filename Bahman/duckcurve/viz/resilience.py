"""Before/after resilience assessment figures for IEEE-33 line outages."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_resilience_indices(before: dict, after: dict) -> plt.Figure:
    """Five-panel comparison of outage-performance and resilience indices."""
    panels = [
        ("eens_kwh", "Expected energy not served", "kWh / outage", False),
        ("worst_case_ens_kwh", "Worst-case energy not served", "kWh", False),
        ("load_served_percent", "Aggregate load served", "%", True),
        ("resilience_index", "Mean resilience index", "p.u.", True),
        ("saledi", "SALEDI outage surrogate", "index (lower is better)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), constrained_layout=True)
    colors = ["#C0392B", "#1F4FA0"]
    for ax, (key, title, unit, higher_is_better) in zip(axes.ravel(), panels):
        values = [float(before[key]), float(after[key])]
        bars = ax.bar(["Before", "After"], values, color=colors, width=0.60)
        upper = max(values) * 1.23 if max(values) > 0.0 else 1.0
        ax.set_ylim(0.0, upper)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(unit, fontsize=9.5)
        ax.grid(axis="y")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015 * upper,
                    f"{value:.3g}", ha="center", va="bottom", fontsize=8.5)
        if higher_is_better:
            change = (values[1] - values[0]) / max(abs(values[0]), 1.0e-12) * 100.0
            change_text = f"{change:.1f}% higher"
        else:
            change = (values[0] - values[1]) / max(abs(values[0]), 1.0e-12) * 100.0
            change_text = f"{change:.1f}% lower"
        ax.text(0.5, 0.88, change_text, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.0, color="#1F4FA0", fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})

    note_ax = axes.ravel()[-1]
    note_ax.axis("off")
    note_ax.text(
        0.5, 0.55,
        f"{int(after['outage_scenarios'])} line/start-hour scenarios\n"
        f"{int(after['outage_duration_hours'])}-hour outage duration\n"
        "PV serves island load first; BESS support is limited\n"
        "by outage-start SOC, power rating, and usable energy.",
        ha="center", va="center", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F4F6F7", "edgecolor": "#888"},
    )
    fig.suptitle("Four-hour line-outage resilience: before vs after optimization", fontsize=13)
    return fig


def plot_line_outage_resilience(before: dict, after: dict) -> plt.Figure:
    """Expected unserved energy for an outage of each IEEE-33 feeder line."""
    base = np.asarray(before["line_eens_kwh"], dtype=float)
    opt = np.asarray(after["line_eens_kwh"], dtype=float)
    lines = np.arange(1, len(base) + 1)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.plot(lines, base, "o--", color="#C0392B", markersize=3.5,
            label="Before optimization")
    ax.plot(lines, opt, "s-", color="#1F4FA0", markersize=3.5,
            label="After optimization")
    ax.fill_between(lines, opt, base, where=base >= opt, color="#2E8B57", alpha=0.15,
                    label="EENS reduction")
    ax.set_xlabel("Outaged line number (IEEE 33-bus feeder)")
    ax.set_ylabel("Mean 4-hour energy not served (kWh)")
    ax.set_title("Resilience effect by feeder line outage")
    ax.set_xlim(1, len(base))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_resilience_violins(before: dict, after: dict) -> plt.Figure:
    """Violin distributions over every line-outage/start-hour scenario."""
    panels = [
        (
            "scenario_unserved_kwh",
            "Energy not served per outage scenario",
            "Unserved energy (kWh)",
            "lower is better",
        ),
        (
            "scenario_load_served_percent",
            "Load served per outage scenario",
            "Load served (%)",
            "higher is better",
        ),
    ]
    colors = ["#C0392B", "#1F4FA0"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)

    for ax, (key, title, ylabel, direction) in zip(axes, panels):
        datasets = [
            np.asarray(before[key], dtype=float),
            np.asarray(after[key], dtype=float),
        ]
        violins = ax.violinplot(
            datasets,
            positions=[1, 2],
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=True,
            points=200,
        )
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.72)
            body.set_linewidth(0.8)
        for component in ("cbars", "cmins", "cmaxes", "cmedians"):
            violins[component].set_color("#333333")
            violins[component].set_linewidth(1.0)

        means = [float(np.mean(values)) for values in datasets]
        ax.scatter([1, 2], means, marker="D", s=38, color="#F4D03F",
                   edgecolor="#222222", zorder=4, label="Mean")
        for xpos, values, mean in zip([1, 2], datasets, means):
            median = float(np.median(values))
            ax.text(
                xpos,
                float(np.max(values)) * 1.015 if np.max(values) > 0 else 0.5,
                f"mean {mean:.1f}\nmedian {median:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_xticks([1, 2], ["Before", "After"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n({direction})", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.42)
        ax.legend(loc="best", fontsize=8.5)
        if key == "scenario_load_served_percent":
            ax.set_ylim(-4.0, 108.0)

    scenarios = int(after["outage_scenarios"])
    duration = int(after["outage_duration_hours"])
    fig.suptitle(
        f"Resilience distributions across all {scenarios} IEEE-33 outage scenarios "
        f"({duration}-hour duration)",
        fontsize=13.5,
    )
    return fig