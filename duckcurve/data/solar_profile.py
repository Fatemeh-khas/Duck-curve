"""24-hour PV output profile.

Shape mirrors Fig. 2(b) of the prior PSO paper: zero at night, near-sinusoidal
ramp through the morning, peak around solar noon (hour 12), symmetric ramp down.
A Türkiye-irradiance-flavoured shape is used as a placeholder until the actual
one-year dataset cited in [32] is integrated.

Returned as a 24-element NumPy array of unit fractions in [0, 1].
"""
from __future__ import annotations

import numpy as np


def solar_24h_profile() -> np.ndarray:
    """Return the 24-hour PV output profile (fraction of nameplate), shape (24,)."""
    hours = np.arange(24)
    # Smooth bell centred on hour 12 with non-zero output from ~05:30 to ~18:30.
    # cos^2 lobe between sunrise and sunset gives the shape in Fig. 2(b).
    sunrise = 5.5
    sunset = 18.5
    out = np.zeros(24)
    for h in hours:
        if sunrise <= h <= sunset:
            phase = np.pi * (h - sunrise) / (sunset - sunrise)
            out[h] = np.sin(phase) ** 2
    # Normalise so peak = 1.0 exactly (the discrete sin^2 maximum is < 1.0).
    out = out / out.max()
    return out
