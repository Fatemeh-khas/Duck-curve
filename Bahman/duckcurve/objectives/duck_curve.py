from __future__ import annotations

import numpy as np

DEFAULT_EVENING_START = 16
DEFAULT_EVENING_END = 20


def sum_slope_squared(
    net_load_kw: np.ndarray,
    evening_weight: float = 0.0,
    peak_weight: float = 0.0,
    t_start: int = DEFAULT_EVENING_START,
    t_end: int = DEFAULT_EVENING_END,
) -> float:
    nl = np.asarray(net_load_kw, dtype=float)
    slopes = np.diff(nl)
    j_full = float(np.sum(slopes**2))

    j_evening = 0.0
    if evening_weight > 0.0:
        lo = max(0, t_start - 1)
        hi = min(slopes.shape[0], t_end - 1)
        if hi > lo:
            j_evening = float(np.sum(slopes[lo:hi] ** 2))

    j_peak = 0.0
    if peak_weight > 0.0:
        j_peak = float((nl.max() - nl.mean()) ** 2)

    return j_full + evening_weight * j_evening + peak_weight * j_peak


def duck_curve_metric(
    net_load_kw: np.ndarray,
    evening_weight: float = 0.0,
    peak_weight: float = 0.0,
    t_start: int = DEFAULT_EVENING_START,
    t_end: int = DEFAULT_EVENING_END,
) -> float:
    return sum_slope_squared(
        net_load_kw=net_load_kw,
        evening_weight=evening_weight,
        peak_weight=peak_weight,
        t_start=t_start,
        t_end=t_end,
    )


def duck_curve_variance(net_load_kw: np.ndarray) -> float:
    nl = np.asarray(net_load_kw, dtype=float)
    return float(np.var(nl))


def evening_ramp_rate(
    net_load_kw: np.ndarray,
    t_start: int = DEFAULT_EVENING_START,
    t_end: int = DEFAULT_EVENING_END,
) -> float:
    nl = np.asarray(net_load_kw, dtype=float)
    if t_end <= t_start:
        return 0.0
    return float((nl[t_end] - nl[t_start]) / (t_end - t_start))


def evening_ramp_peak(
    net_load_kw: np.ndarray,
    t_start: int = DEFAULT_EVENING_START,
    t_end: int = DEFAULT_EVENING_END,
) -> float:
    nl = np.asarray(net_load_kw, dtype=float)
    slopes = np.diff(nl)
    lo = max(0, t_start - 1)
    hi = min(slopes.shape[0], t_end - 1)
    if hi <= lo:
        return 0.0
    seg = slopes[lo:hi]
    return float(seg[np.argmax(np.abs(seg))])