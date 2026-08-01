import numpy as np

from duckcurve.data.stochastic_pv import (
    PCAPVModel,
    generate_stochastic_pv_profiles,
    synthetic_training_days,
)
from duckcurve.experiments.baseline import build_baseline_scenario
from duckcurve.grid.timeseries import run_timeseries
from duckcurve.optimizer.encoding import DecisionSpec


def test_pca_scenarios_are_physical_variable_and_reproducible():
    first, model, source = generate_stochastic_pv_profiles(20, seed=17)
    second, _, _ = generate_stochastic_pv_profiles(20, seed=17)
    assert source == "synthetic_fallback"
    assert first.shape == (20, 24)
    assert np.allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert np.any(np.std(first, axis=0) > 0.0)
    assert isinstance(model, PCAPVModel)


def test_custom_pv_profile_changes_timeseries_output():
    spec = DecisionSpec()
    scenario = build_baseline_scenario(spec)
    sunny = np.zeros(24)
    sunny[12] = 1.0
    cloudy = sunny * 0.25
    a = run_timeseries(scenario, pv_profile=sunny)
    b = run_timeseries(scenario, pv_profile=cloudy)
    assert a.total_pv_kw[12] > b.total_pv_kw[12]
    assert a.net_load_kw[12] < b.net_load_kw[12]


def test_synthetic_training_days_have_expected_shape():
    days = synthetic_training_days(30, seed=4)
    assert days.shape == (30, 24)
    assert np.all(days[:, :5] == 0.0)


def test_included_renewables_ninja_year_loads_as_daily_profiles():
    from pathlib import Path
    from duckcurve.data.stochastic_pv import load_daily_pv_csv

    path = Path(__file__).parents[1] / "duckcurve" / "data" / (
        "ninja_pv_41.0064_28.9759_uncorrected.csv"
    )
    profiles = load_daily_pv_csv(path)
    assert profiles.shape[1] == 24
    assert profiles.shape[0] >= 360
    assert np.all((profiles >= 0.0) & (profiles <= 1.0))


def test_included_pvgis_two_year_file_loads_as_daily_profiles():
    from pathlib import Path
    from duckcurve.data.stochastic_pv import load_daily_pv_csv

    path = Path(__file__).parents[1] / "duckcurve" / "data" / "PVGIS_hourly_PV.csv"
    profiles = load_daily_pv_csv(path)
    assert profiles.shape[1] == 24
    assert profiles.shape[0] >= 725
    assert np.all((profiles >= 0.0) & (profiles <= 1.0))
