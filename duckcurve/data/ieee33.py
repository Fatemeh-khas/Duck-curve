"""IEEE 33-bus radial distribution system (Baran & Wu, 1989).

Parameters taken from:
    Baran, M. E., & Wu, F. F. (1989). Network reconfiguration in distribution systems
    for loss reduction and load balancing. IEEE Transactions on Power Delivery,
    4(2), 1401-1407.

Nominal voltage: 12.66 kV. Base MVA: 10. Slack at bus 1.

Bus loads are (P_kW, Q_kVAr) per Baran-Wu Table 1. Lines are (from, to, R_ohm, X_ohm).
Bus 1 is the slack; buses 2-33 carry the loads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Bus:
    idx: int          # 1-indexed bus number
    p_kw: float       # active load (kW)
    q_kvar: float     # reactive load (kVAr)


@dataclass(frozen=True)
class Line:
    from_bus: int     # 1-indexed
    to_bus: int       # 1-indexed
    r_ohm: float
    x_ohm: float


# ---- Bus data (Baran-Wu Table 1) ------------------------------------------
# (bus_idx, P_kW, Q_kVAr). Bus 1 is slack with zero load.
_BUS_DATA = [
    (1, 0, 0),     (2, 100, 60),   (3, 90, 40),    (4, 120, 80),
    (5, 60, 30),   (6, 60, 20),    (7, 200, 100),  (8, 200, 100),
    (9, 60, 20),   (10, 60, 20),   (11, 45, 30),   (12, 60, 35),
    (13, 60, 35),  (14, 120, 80),  (15, 60, 10),   (16, 60, 20),
    (17, 60, 20),  (18, 90, 40),   (19, 90, 40),   (20, 90, 40),
    (21, 90, 40),  (22, 90, 40),   (23, 90, 50),   (24, 420, 200),
    (25, 420, 200),(26, 60, 25),   (27, 60, 25),   (28, 60, 20),
    (29, 120, 70), (30, 200, 600), (31, 150, 70),  (32, 210, 100),
    (33, 60, 40),
]

IEEE33_BUSES: List[Bus] = [Bus(idx=i, p_kw=p, q_kvar=q) for (i, p, q) in _BUS_DATA]


# ---- Line data (Baran-Wu Table 2) -----------------------------------------
# (from_bus, to_bus, R_ohm, X_ohm). Radial topology, 32 branches.
_LINE_DATA = [
    (1, 2, 0.0922, 0.0477),   (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),   (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),   (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351),   (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),  (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450), (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),  (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),  (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),  (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]

IEEE33_LINES: List[Line] = [
    Line(from_bus=f, to_bus=t, r_ohm=r, x_ohm=x) for (f, t, r, x) in _LINE_DATA
]


def peak_active_load_mw() -> float:
    """Total nominal active load across the feeder, in MW."""
    return sum(b.p_kw for b in IEEE33_BUSES) / 1000.0


def peak_reactive_load_mvar() -> float:
    """Total nominal reactive load across the feeder, in MVAr."""
    return sum(b.q_kvar for b in IEEE33_BUSES) / 1000.0


# Buses available for DG placement (slack bus 1 excluded).
PLACEABLE_BUSES = list(range(2, 34))
