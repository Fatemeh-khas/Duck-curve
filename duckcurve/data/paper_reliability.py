"""Reliability data reported by Ahmadi et al. (UPEC 2021).

Source:
    B. Ahmadi, S. Younesi, O. Ceylan, and A. Ozdemir,
    "Multi-objective Distributed Energy Resource Integration in Radial
    Distribution Networks," UPEC 2021, doi:10.1109/UPEC50034.2021.9548208.

Only values explicitly tabulated in Table I are encoded here. The paper uses
4 h and 2 h as the mean repair durations of main-feeder and load-lateral
components, respectively. It reports base-case SAIFI = 1.534 interruptions
per customer-year and AENS = 24.48 kWh/customer-year.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


PAPER_DOI = "10.1109/UPEC50034.2021.9548208"
MAIN_REPAIR_MEAN_HOURS = 4.0
LATERAL_REPAIR_MEAN_HOURS = 2.0
PUBLISHED_BASE_SAIFI = 1.534
PUBLISHED_BASE_AENS_KWH_PER_CUSTOMER = 24.48


@dataclass(frozen=True)
class PaperReliabilityRecord:
    branch_number: int
    from_bus: int
    to_bus: int
    main_failure_rate_per_year: float
    lateral_failure_rate_per_year: float
    active_load_kw: float
    load_type: str
    customers: int


PAPER_RELIABILITY_RECORDS: Tuple[PaperReliabilityRecord, ...] = tuple(
    PaperReliabilityRecord(*row)
    for row in (
        (1, 1, 2, 0.06, 0.05, 100, "commercial", 20),
        (2, 2, 3, 0.18, 0.15, 90, "commercial", 18),
        (3, 3, 4, 0.12, 0.10, 120, "commercial", 24),
        (4, 4, 5, 0.12, 0.10, 60, "residential", 12),
        (5, 5, 6, 0.30, 0.25, 60, "residential", 12),
        (6, 6, 7, 0.06, 0.05, 200, "commercial", 40),
        (7, 7, 8, 0.24, 0.20, 200, "commercial", 40),
        (8, 8, 9, 0.36, 0.30, 60, "residential", 12),
        (9, 9, 10, 0.36, 0.30, 60, "residential", 12),
        (10, 10, 11, 0.06, 0.05, 45, "residential", 9),
        (11, 11, 12, 0.12, 0.10, 60, "residential", 12),
        (12, 12, 13, 0.48, 0.40, 60, "residential", 12),
        (13, 13, 14, 0.18, 0.15, 120, "commercial", 24),
        (14, 14, 15, 0.18, 0.15, 60, "residential", 12),
        (15, 15, 16, 0.24, 0.20, 60, "residential", 12),
        (16, 16, 17, 0.42, 0.35, 60, "residential", 12),
        (17, 17, 18, 0.24, 0.20, 90, "commercial", 18),
        (18, 2, 19, 0.06, 0.05, 90, "commercial", 18),
        (19, 19, 20, 0.48, 0.40, 90, "commercial", 18),
        (20, 20, 21, 0.18, 0.15, 90, "commercial", 18),
        (21, 21, 22, 0.24, 0.20, 90, "commercial", 18),
        (22, 3, 23, 0.18, 0.15, 90, "commercial", 18),
        (23, 23, 24, 0.30, 0.25, 420, "industrial", 84),
        (24, 24, 25, 0.30, 0.25, 420, "industrial", 84),
        (25, 6, 26, 0.12, 0.10, 60, "residential", 12),
        (26, 26, 27, 0.12, 0.10, 60, "residential", 12),
        (27, 27, 28, 0.36, 0.30, 60, "residential", 12),
        (28, 28, 29, 0.30, 0.25, 120, "commercial", 24),
        (29, 29, 30, 0.18, 0.15, 200, "commercial", 40),
        (30, 30, 31, 0.30, 0.25, 150, "commercial", 30),
        (31, 31, 32, 0.12, 0.10, 210, "industrial", 42),
        (32, 32, 33, 0.12, 0.10, 60, "residential", 12),
    )
)


def paper_analytic_base_indices() -> dict:
    """Apply the paper's stated radial-series formulas to its Table I data.

    The SAIFI reproduction agrees with the paper after rounding. The direct
    AENS reproduction is intentionally returned without a calibration factor;
    it differs from the published AENS, which is recorded separately.
    """
    by_bus = {record.to_bus: record for record in PAPER_RELIABILITY_RECORDS}
    total_customers = sum(record.customers for record in PAPER_RELIABILITY_RECORDS)
    saifi_numerator = 0.0
    aens_numerator = 0.0

    for record in PAPER_RELIABILITY_RECORDS:
        current_bus = record.to_bus
        upstream_main_rate = 0.0
        while current_bus != 1:
            incoming = by_bus[current_bus]
            upstream_main_rate += incoming.main_failure_rate_per_year
            current_bus = incoming.from_bus

        load_point_rate = upstream_main_rate + record.lateral_failure_rate_per_year
        annual_unavailability_hours = (
            upstream_main_rate * MAIN_REPAIR_MEAN_HOURS
            + record.lateral_failure_rate_per_year * LATERAL_REPAIR_MEAN_HOURS
        )
        saifi_numerator += load_point_rate * record.customers
        aens_numerator += record.active_load_kw * annual_unavailability_hours

    calculated_saifi = saifi_numerator / total_customers
    calculated_aens = aens_numerator / total_customers
    return {
        "total_customers": total_customers,
        "calculated_saifi_interruptions_per_customer_year": calculated_saifi,
        "published_saifi_interruptions_per_customer_year": PUBLISHED_BASE_SAIFI,
        "calculated_aens_kwh_per_customer_year": calculated_aens,
        "published_aens_kwh_per_customer_year": PUBLISHED_BASE_AENS_KWH_PER_CUSTOMER,
        "aens_reproduction_gap_kwh_per_customer_year": (
            calculated_aens - PUBLISHED_BASE_AENS_KWH_PER_CUSTOMER
        ),
    }
