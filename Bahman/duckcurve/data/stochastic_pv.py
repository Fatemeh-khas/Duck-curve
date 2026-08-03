"""Data-driven stochastic 24-hour PV profiles.

The model uses principal-component analysis (PCA) to learn the correlated
intraday variability in historical daily PV profiles.  New scenarios are
sampled in the learned latent space, then clipped to physical [0, 1] limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv
import math

import numpy as np

from .solar_profile import solar_24h_profile


def _load_soda_irradiance(
    path: str | Path, seed: int, augmented_days: int = 365, utc_offset_hours: int = 3
) -> np.ndarray:
    """Convert a SoDa/HelioClim minute file into bootstrapped daily PV profiles.

    Global-horizontal irradiation is used as a normalized PV proxy.  If the
    file contains only one measured day, minute observations are block-sampled
    within adjacent local-time hours to provide training examples for PCA.
    """
    rows = []
    with open(path, encoding="utf-8-sig") as stream:
        for line in stream:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split(";")
            if len(parts) < 5:
                continue
            try:
                date = datetime.strptime(parts[0], "%Y-%m-%d")
                hh, mm = (int(v) for v in parts[1].split(":"))
                # SoDa uses 24:00 for the final minute of the day.
                stamp = date + timedelta(hours=hh, minutes=mm)
                stamp += timedelta(hours=utc_offset_hours)
                ghi = float(parts[2])
                clear = float(parts[3])
            except (ValueError, IndexError):
                continue
            rows.append((stamp, max(0.0, ghi), max(0.0, clear)))
    if len(rows) < 24:
        raise ValueError("SoDa irradiance file contains insufficient numeric observations")

    scale = max(v[1] for v in rows)
    if scale <= 0.0:
        raise ValueError("SoDa irradiance file has no positive Global Horiz values")
    observed = np.zeros(24)
    hourly_samples: list[list[float]] = [[] for _ in range(24)]
    for stamp, ghi, _ in rows:
        hourly_samples[stamp.hour].append(ghi / scale)
    for hour, values in enumerate(hourly_samples):
        observed[hour] = float(np.mean(values)) if values else 0.0

    rng = np.random.default_rng(seed)
    training = np.zeros((max(3, augmented_days), 24))
    daylight = observed > 0.0
    for day in range(training.shape[0]):
        daily_scale = np.clip(rng.normal(1.0, 0.08), 0.72, 1.12)
        for hour in range(24):
            pool = []
            for neighbor in range(max(0, hour - 1), min(24, hour + 2)):
                pool.extend(hourly_samples[neighbor])
            if not pool or not daylight[hour]:
                continue
            sample_size = min(60, len(pool))
            # Bootstrap minute irradiance, preserving the measured distribution.
            boot = rng.choice(pool, size=sample_size, replace=True)
            training[day, hour] = np.clip(daily_scale * np.mean(boot), 0.0, 1.0)
    # Keep the actual hourly measured profile as the first training observation.
    training[0] = observed
    return training


def _load_renewables_ninja(
    path: str | Path, utc_offset_hours: int = 3
) -> np.ndarray:
    """Load hourly Renewables.ninja PV output as local-time daily profiles."""
    records: dict[datetime.date, dict[int, float]] = {}
    with open(path, encoding="utf-8-sig") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        reader = csv.DictReader(lines)
        required = {"time", "electricity"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Renewables.ninja CSV requires time and electricity columns")
        for row in reader:
            try:
                stamp = datetime.strptime(row["time"], "%Y-%m-%d %H:%M")
                stamp += timedelta(hours=utc_offset_hours)
                power = max(0.0, float(row["electricity"]))
            except (ValueError, TypeError):
                continue
            records.setdefault(stamp.date(), {})[stamp.hour] = power
    complete_days = [
        np.array([hours[h] for h in range(24)], dtype=float)
        for _, hours in sorted(records.items())
        if len(hours) == 24
    ]
    if len(complete_days) < 3:
        raise ValueError("Renewables.ninja CSV contains fewer than three complete local days")
    profiles = np.vstack(complete_days)
    # The API normally reports output for the configured nameplate capacity.
    # Normalize only if the file is not already a 0..1 capacity-factor series.
    peak = float(profiles.max())
    if peak > 1.0:
        profiles /= peak
    return np.clip(profiles, 0.0, 1.0)


def _load_pvgis(path: str | Path, utc_offset_hours: int = 3) -> np.ndarray:
    """Load PVGIS hourly PV power and return complete local-time daily profiles."""
    records: dict[object, dict[int, float]] = {}
    with open(path, encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"time", "P"}.issubset(reader.fieldnames):
            raise ValueError("PVGIS CSV requires time and P columns")
        raw = []
        for row in reader:
            try:
                stamp = datetime.fromisoformat(row["time"])
                stamp = stamp.replace(tzinfo=None) + timedelta(hours=utc_offset_hours)
                power = max(0.0, float(row["P"]))
            except (ValueError, TypeError):
                continue
            raw.append((stamp, power))
    if len(raw) < 72:
        raise ValueError("PVGIS CSV contains fewer than three days of observations")

    peak = max(power for _, power in raw)
    if peak <= 0.0:
        raise ValueError("PVGIS CSV has no positive PV power")
    # PVGIS P is commonly exported in W. Infer the round nameplate scale
    # (e.g. 892620 W peak -> 1,000,000 W nameplate), never the sample maximum.
    nameplate = 1.0 if peak <= 1.0 else 10.0 ** math.ceil(math.log10(peak))
    for stamp, power in raw:
        records.setdefault(stamp.date(), {})[stamp.hour] = power / nameplate
    complete_days = [
        np.array([hours[h] for h in range(24)], dtype=float)
        for _, hours in sorted(records.items())
        if len(hours) == 24
    ]
    if len(complete_days) < 3:
        raise ValueError("PVGIS CSV contains fewer than three complete local days")
    return np.clip(np.vstack(complete_days), 0.0, 1.0)


def load_daily_pv_csv(
    path: str | Path, seed: int = 2026, augmented_days: int = 365
) -> np.ndarray:
    """Load daily profiles or a SoDa/HelioClim irradiance export.

    Numeric CSV input is one normalized 24-value day per row. Semicolon SoDa
    files are converted from UTC to UTC+3 and bootstrapped from minute data.
    """
    with open(path, encoding="utf-8-sig") as stream:
        prefix = stream.read(512)
    if prefix.lstrip().startswith("time,P,") or "poa_direct" in prefix:
        return _load_pvgis(path)
    if "Renewables.ninja" in prefix:
        return _load_renewables_ninja(path)
    if "HelioClim" in prefix or "Global Horiz" in prefix or ";" in prefix:
        return _load_soda_irradiance(path, seed=seed, augmented_days=augmented_days)
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    # Tolerate a single header row.
    data = data[~np.isnan(data).all(axis=1)]
    if data.shape[1] != 24 or np.isnan(data).any():
        raise ValueError("PV training CSV must contain 24 numeric hourly values per row")
    if data.shape[0] < 3:
        raise ValueError("PV training CSV must contain at least three daily profiles")
    return np.clip(data, 0.0, 1.0)


def synthetic_training_days(
    n_days: int = 365, seed: int = 2026, variability: float = 0.18
) -> np.ndarray:
    """Create a reproducible fallback training set when measurements are absent."""
    if n_days < 3:
        raise ValueError("n_days must be at least 3")
    rng = np.random.default_rng(seed)
    clear = solar_24h_profile()
    hours = np.arange(24)
    days = np.zeros((n_days, 24))
    for i in range(n_days):
        scale = np.clip(rng.normal(0.88, variability), 0.25, 1.08)
        shift = rng.normal(0.0, 0.45)
        shifted = np.interp(hours - shift, hours, clear, left=0.0, right=0.0)
        # Smooth, temporally correlated cloud attenuation.
        innovations = rng.normal(0.0, variability * 0.55, 24)
        cloud = np.convolve(innovations, np.array([0.2, 0.6, 0.2]), mode="same")
        days[i] = np.clip(scale * shifted * (1.0 + cloud), 0.0, 1.0)
    days[:, clear == 0.0] = 0.0
    return days


@dataclass
class PCAPVModel:
    mean_: np.ndarray
    components_: np.ndarray
    latent_cov_: np.ndarray
    daylight_mask_: np.ndarray

    @classmethod
    def fit(cls, profiles: np.ndarray, variance_retained: float = 0.95) -> "PCAPVModel":
        x = np.asarray(profiles, dtype=float)
        if x.ndim != 2 or x.shape[1] != 24 or x.shape[0] < 3:
            raise ValueError("profiles must have shape (n_days, 24), n_days >= 3")
        if not 0.0 < variance_retained <= 1.0:
            raise ValueError("variance_retained must be in (0, 1]")
        x = np.clip(x, 0.0, 1.0)
        mean = x.mean(axis=0)
        centered = x - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        explained = singular**2
        cumulative = np.cumsum(explained) / max(float(explained.sum()), 1e-12)
        k = max(1, int(np.searchsorted(cumulative, variance_retained) + 1))
        components = vt[:k]
        scores = centered @ components.T
        cov = np.atleast_2d(np.cov(scores, rowvar=False))
        cov += np.eye(k) * 1e-10
        daylight = np.max(x, axis=0) > 1e-8
        return cls(mean, components, cov, daylight)

    def sample(self, n_scenarios: int, seed: int = 42) -> np.ndarray:
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        rng = np.random.default_rng(seed)
        latent = rng.multivariate_normal(
            np.zeros(self.components_.shape[0]), self.latent_cov_, size=n_scenarios
        )
        profiles = self.mean_[None, :] + latent @ self.components_
        profiles = np.clip(profiles, 0.0, 1.0)
        profiles[:, ~self.daylight_mask_] = 0.0
        return profiles


def generate_stochastic_pv_profiles(
    n_scenarios: int,
    seed: int = 42,
    training_csv: str | Path | None = None,
    variance_retained: float = 0.95,
    synthetic_days: int = 365,
    synthetic_variability: float = 0.18,
) -> tuple[np.ndarray, PCAPVModel, str]:
    """Fit the PCA model and return scenarios, model, and training-source label."""
    if training_csv:
        training = load_daily_pv_csv(
            training_csv, seed=seed + 7919, augmented_days=synthetic_days
        )
        source = str(Path(training_csv))
    else:
        training = synthetic_training_days(
            synthetic_days, seed=seed + 7919, variability=synthetic_variability
        )
        source = "synthetic_fallback"
    model = PCAPVModel.fit(training, variance_retained=variance_retained)
    return model.sample(n_scenarios, seed=seed), model, source
