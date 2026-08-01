"""Publication-style IEEE feeder voltage assessment figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _validate(base_v_pu: np.ndarray, opt_v_pu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_v_pu, dtype=float)
    opt = np.asarray(opt_v_pu, dtype=float)
    if base.shape != opt.shape or base.ndim != 2:
        raise ValueError("Voltage arrays must have identical (hour, bus) shapes")
    return base, opt


def plot_voltage_profile(
    base_v_pu: np.ndarray,
    opt_v_pu: np.ndarray,
    no_pv_v_pu: np.ndarray | None = None,
    v_min_limit_pu: float = 0.95,
    v_max_limit_pu: float = 1.05,
) -> plt.Figure:
    """Conventional bus-by-bus voltage profile at the critical load hour."""
    base, opt = _validate(base_v_pu, opt_v_pu)
    no_pv = np.asarray(no_pv_v_pu, dtype=float) if no_pv_v_pu is not None else None
    if no_pv is not None and no_pv.shape != base.shape:
        raise ValueError("no_pv_v_pu must have the same shape as the other voltage arrays")

    reference = no_pv if no_pv is not None else base
    if no_pv is not None:
        pv_effect = np.max(np.abs(base - no_pv), axis=1)
        eligible = np.flatnonzero(pv_effect >= 2.0e-3)
        critical_hour = (
            int(eligible[np.argmin(base[eligible].min(axis=1))])
            if len(eligible)
            else int(np.argmin(reference.min(axis=1)))
        )
    else:
        critical_hour = int(np.argmin(reference.min(axis=1)))
    buses = np.arange(1, base.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    if no_pv is not None:
        ax.plot(buses, no_pv[critical_hour], color="#E53935", linestyle="--",
                linewidth=1.8, label="No-PV / no-BESS baseline")
    ax.plot(buses, base[critical_hour], color="#DD7B36", marker="o", markersize=4.0,
            linewidth=1.8, label="PV-only before optimization")
    ax.plot(buses, opt[critical_hour], color="#167A9C", marker="^", markersize=4.5,
            linewidth=2.0, label="Optimized PV + BESS")
    ax.axhline(v_min_limit_pu, color="#222", linestyle=":", linewidth=1.2,
               label="IEEE 0.95/1.05 p.u. limits")
    ax.axhline(v_max_limit_pu, color="#222", linestyle=":", linewidth=1.2)

    data_min = min(float(opt[critical_hour].min()), float(base[critical_hour].min()),
                   float(no_pv[critical_hour].min()) if no_pv is not None else 1.0)
    ax.set_ylim(min(0.90, data_min - 0.005), 1.06)
    ax.set_xlim(1, base.shape[1])
    ax.set_xticks(np.arange(1, base.shape[1] + 1, 4))
    ax.set_xlabel("Bus number")
    ax.set_ylabel("Voltage magnitude (p.u.)")
    ax.set_title(f"IEEE 33-bus voltage profile at DER-active critical hour {critical_hour:02d}:00")
    ax.grid(True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_voltage_heatmaps(
    base_v_pu: np.ndarray,
    opt_v_pu: np.ndarray,
    v_min_limit_pu: float = 0.95,
) -> plt.Figure:
    """Complete 24-hour spatial-temporal voltage assessment."""
    base, opt = _validate(base_v_pu, opt_v_pu)
    n_hours, n_buses = base.shape
    buses = np.arange(1, n_buses + 1)
    hours = np.arange(n_hours)
    color_min = min(float(base.min()), float(opt.min()), v_min_limit_pu)
    color_max = max(float(base.max()), float(opt.max()), 1.01)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(buses, base.min(axis=0), "o--", markersize=3,
                    color="#C0392B", label="Before daily minimum")
    axes[0, 0].plot(buses, opt.min(axis=0), "s-", markersize=3,
                    color="#1F4FA0", label="After daily minimum")
    axes[0, 0].axhline(v_min_limit_pu, color="#333", linestyle=":", label="0.95 p.u. limit")
    axes[0, 0].set_title("(a) Daily minimum voltage at every bus")
    axes[0, 0].set_xlabel("Bus number")
    axes[0, 0].set_ylabel("Voltage magnitude (p.u.)")
    axes[0, 0].set_xlim(1, n_buses)
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc="lower left")

    axes[0, 1].plot(hours, base.min(axis=1), "--", color="#C0392B",
                    label="Before minimum bus")
    axes[0, 1].plot(hours, opt.min(axis=1), "-", color="#1F4FA0",
                    label="After minimum bus")
    axes[0, 1].axhline(v_min_limit_pu, color="#333", linestyle=":", label="0.95 p.u. limit")
    axes[0, 1].set_title("(b) Minimum feeder voltage over 24 hours")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Voltage magnitude (p.u.)")
    axes[0, 1].set_xlim(0, n_hours - 1)
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc="lower left")

    extent = [0.5, n_buses + 0.5, n_hours - 0.5, -0.5]
    image = axes[1, 0].imshow(base, aspect="auto", cmap="viridis",
                              vmin=color_min, vmax=color_max, extent=extent)
    axes[1, 0].set_title("(c) Before optimization: hour-by-bus voltage")
    axes[1, 0].set_xlabel("Bus number")
    axes[1, 0].set_ylabel("Hour")
    axes[1, 0].set_yticks(hours[::3])

    axes[1, 1].imshow(opt, aspect="auto", cmap="viridis",
                      vmin=color_min, vmax=color_max, extent=extent)
    axes[1, 1].set_title("(d) After optimization: hour-by-bus voltage")
    axes[1, 1].set_xlabel("Bus number")
    axes[1, 1].set_ylabel("Hour")
    axes[1, 1].set_yticks(hours[::3])
    cbar = fig.colorbar(image, ax=axes[1, :], orientation="horizontal", shrink=0.80, pad=0.12)
    cbar.set_label("Voltage magnitude (p.u.)")
    return fig