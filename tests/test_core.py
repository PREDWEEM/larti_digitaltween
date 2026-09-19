from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from predweem_twin.core import (
    ModelParameters,
    PracticalANNModel,
    apply_cohort_decay,
    run_predweem,
)
from predweem_twin.state import thermal_window_dates


ROOT = Path(__file__).parents[1]

spec = importlib.util.spec_from_file_location(
    "lartigau_original", ROOT / "tests/fixtures/lartigau_original.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


@pytest.mark.parametrize("coverage,wmax,kr", [(75, 18.816, 0), (10, 10, 0), (60, 36, 1), (100, 5, 0)])
def test_extracted_core_matches_original_lartigau(coverage, wmax, kr):
    weather = pd.read_csv(ROOT / "data/calibration/lartigau_2026_weather.csv")
    model = PracticalANNModel.from_directory(ROOT / "models")
    original_model = original.PracticalANNModel(
        model.iw, model.bias_iw, model.lw, model.bias_out,
    )
    expected, _ = original.simulate_emergence(
        weather, original_model, coverage, wmax, kr_exponent=kr,
    )
    result = run_predweem(
        weather, model, ModelParameters(cobertura_pct=coverage, w_max=wmax, exponente_kr=kr),
    )
    for column in (
        "EMERREL_RAW_ANN", "ET0", "W_superficial", "Hydric_Factor",
        "EMERREL", "EMERAC", "EMERAC_NORMALIZADA", "Factor_Decaimiento_15Abr",
        "Techo_EMERREL_15Abr", "TT_DESDE_PICO",
    ):
        if column in expected:
            np.testing.assert_allclose(result[column], expected[column], atol=1e-12, rtol=1e-12)
    assert result.Termoinhibida.equals(expected.Termoinhibida)
    assert result.Primer_Pico_Habilitado.equals(expected.Primer_Pico_Habilitado)


def test_lartigau_april_decay_preserves_early_flow_and_caps_late_flow():
    frame = pd.DataFrame({"Fecha": pd.date_range("2026-04-01", "2026-08-01"), "EMERREL": .8})
    result = apply_cohort_decay(frame, 0, ModelParameters())
    assert result.loc[result.Fecha.lt("2026-04-15"), "EMERREL"].eq(.8).all()
    assert result.loc[result.Fecha.eq("2026-04-15"), "EMERREL"].iloc[0] == pytest.approx(.4)
    late = result.loc[result.Fecha.ge("2026-04-15")]
    assert late.EMERREL.is_monotonic_decreasing
    assert late.EMERREL.gt(.1).all()  # No extinción a 110 días pospico.
    disabled = apply_cohort_decay(frame, 0, ModelParameters(decay_enabled=False))
    assert disabled.EMERREL.equals(frame.EMERREL)


def test_decay_does_not_invent_a_cap_without_pre_april_emergence():
    frame = pd.DataFrame({"Fecha": pd.date_range("2026-04-16", periods=10), "EMERREL": .8})
    result = apply_cohort_decay(frame, 0, ModelParameters())
    assert result.EMERREL.equals(frame.EMERREL)
    assert result.Techo_EMERREL_15Abr.isna().all()


def test_real_lartigau_run_has_expected_invariants():
    weather = pd.read_csv(ROOT / "meteo_daily.csv")
    model = PracticalANNModel.from_directory(ROOT / "models")
    result = run_predweem(weather, model, ModelParameters())
    assert not result.empty
    assert result["Fecha"].is_monotonic_increasing
    assert result["EMERREL"].between(0, 1).all()
    assert result["EMERAC_NORMALIZADA"].between(0, 1).all()
    assert (result["W_superficial"] >= 0).all()
    assert (result["W_superficial"] <= 18.816 + 1e-9).all()
    assert (result.loc[result["Julian_days"] <= 25, "EMERREL"] == 0).all()
    assert "Factor_Decaimiento_15Abr" in result
    assert "Techo_EMERREL_15Abr" in result


def test_weather_validation_rejects_inverted_temperatures():
    weather = pd.DataFrame(
        {"Fecha": ["2026-01-01"], "TMAX": [10], "TMIN": [15], "Prec": [0]}
    )
    model = PracticalANNModel.from_directory(ROOT / "models")
    try:
        run_predweem(weather, model, ModelParameters())
    except ValueError as error:
        assert "TMAX" in str(error)
    else:
        raise AssertionError("Se esperaba ValueError")


def test_run_uses_observed_daily_coverage_series():
    weather = pd.read_csv(ROOT / "meteo_daily.csv").head(5)
    weather_dates = pd.to_datetime(weather["Fecha"])
    coverage = pd.DataFrame(
        {
            "Fecha": [weather_dates.iloc[0], weather_dates.iloc[2]],
            "Cobertura_PCT": [20.0, 80.0],
        }
    )
    model = PracticalANNModel.from_directory(ROOT / "models")

    result = run_predweem(
        weather,
        model,
        ModelParameters(cobertura_pct=40.0),
        coverage_series=coverage,
    )

    assert np.allclose(
        result["Cobertura_Rastrojo"], [20.0, 50.0, 80.0, 80.0, 80.0]
    )
    assert result["Cobertura_Modo"].eq("serie observada interpolada").all()
    assert result["Cobertura_Observada"].tolist() == [True, False, True, False, False]


def test_thermal_window_dates_detects_600_and_800_degree_days():
    trajectory = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-04-01", periods=5, freq="D"),
            "TT_DESDE_PICO": [550.0, 600.0, 710.0, 800.0, 850.0],
        }
    )

    start, end = thermal_window_dates(trajectory)

    assert start == pd.Timestamp("2026-04-02")
    assert end == pd.Timestamp("2026-04-04")
