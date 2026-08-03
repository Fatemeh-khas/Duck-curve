"""24-hour residential load profile.

Shape mirrors Fig. 2(a) of the prior PSO paper (Final version.pdf): a morning ramp,
a midday dip, and the pronounced evening peak around 19:00 that produces the duck
shape when combined with the PV output curve.

Returned as a 24-element NumPy array of *unit fractions* in [0, 1]; the network
layer multiplies this by each bus's nominal load.
"""
from __future__ import annotations

import numpy as np


# Reverse-engineered from Fig. 2(a) of Final version.pdf — 24 hourly samples (00:00–23:00).
# Units: fraction of peak load. Peak is at hour 19 (=1.0).
_HOURLY = np.array([
    0.55, 0.50, 0.45, 0.42, 0.40, 0.42,   # 00-05 — overnight trough
    0.55, 0.68, 0.72, 0.65, 0.55, 0.50,   # 06-11 — morning then midday dip
    0.55, 0.60, 0.65, 0.70, 0.78, 0.88,   # 12-17 — afternoon ramp
    0.95, 1.00, 0.96, 0.85, 0.72, 0.60,   # 18-23 — evening peak then decline
])


def residential_24h_profile() -> np.ndarray:
    """Return the 24-hour residential load profile (fraction of peak), shape (24,)."""
    return _HOURLY.copy()
