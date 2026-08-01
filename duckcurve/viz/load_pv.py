"""Load and PV input figures for deterministic or stochastic PV studies."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_load_and_pv(load_profile: np.ndarray, pv_profiles: np.ndarray) -> plt.Figure:
    """Plot the load and either one PV profile or a stochastic PV ensemble."""
    load_profile = np.asarray(load_profile, dtype=float)
    pv_profiles = np.asarray(pv_profiles, dtype=float)
    if load_profile.shape != (24,):
        raise ValueError("load_profile must contain 24 hourly values")
    if pv_profiles.ndim == 1:
        pv_profiles = pv_profiles[None, :]
    if pv_profiles.ndim != 2 or pv_profiles.shape[1] != 24:
        raise ValueError("pv_profiles must have shape (24,) or (n_scenarios, 24)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    hours = np.arange(len(load_profile))

    ax1.plot(hours, 100 * load_profile, color="#C0392B", linewidth=1.8)
    ax1.set_title("Scaled hourly residential load")
    ax1.set_ylabel("load (%)")
    ax1.set_ylim(35, 105)
    ax1.grid(True)

    if len(pv_profiles) == 1:
        ax2.plot(
            hours, 100 * pv_profiles[0], color="#C0392B", linewidth=1.8,
            label="PV profile",
        )
        ax2.set_title("Scaled PV output")
    else:
        lower, median, upper = np.percentile(pv_profiles, [5, 50, 95], axis=0)
        mean = pv_profiles.mean(axis=0)
        ax2.fill_between(
            hours, 100 * lower, 100 * upper, color="#4C78A8", alpha=0.18,
            label="5th–95th percentile band",
        )
        for i, profile in enumerate(pv_profiles):
            ax2.plot(
                hours, 100 * profile, color="#4C78A8", linewidth=0.65,
                alpha=0.28, label="PCA scenarios" if i == 0 else None,
            )
        ax2.plot(
            hours, 100 * mean, color="#C0392B", linewidth=2.2,
            label="Scenario mean",
        )
        ax2.plot(
            hours, 100 * median, color="#222222", linewidth=1.2,
            linestyle="--", label="Scenario median",
        )
        ax2.set_title(f"Stochastic PV scenario ensemble (n={len(pv_profiles)})")
        ax2.legend(loc="upper right", frameon=True, fontsize=7)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("PV (%)")
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(0, len(hours) - 1)
    ax2.set_xticks(np.arange(0, len(hours), 5))
    ax2.grid(True)

    fig.tight_layout()
    return fig
