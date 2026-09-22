"""Intensidad por flujo futuro frente al pico del pool, con horizonte completo."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predweem_twin.charts import trajectory_charts
from predweem_twin.flows import historical_weekly_max
from predweem_twin.seasonal import load_local_seasonal_reference
from predweem_twin.state import build_twin_snapshot, weekly_flow_intensity


ROOT = Path(__file__).parents[1]
CUTOFF = pd.Timestamp("2027-05-05")


@pytest.fixture
def reference():
    return load_local_seasonal_reference(ROOT, as_of=CUTOFF)


def forecast(total=.12, cutoff=CUTOFF):
    dates = pd.date_range(cutoff, periods=10)
    # Hoy y los días posteriores al horizonte no deben influir en la intensidad.
    flows = np.array([.9, *([total / 7] * 7), .8, .8])
    return pd.DataFrame({
        "Fecha": dates,
        "EMERREL_TWIN": flows,
        "EMERAC_TWIN": .4 + np.arange(10) * total / 7,
        "EMERAC_NORMALIZADA": .4 + np.arange(10) * total / 7,
        "W_superficial": 10., "Humedad_Relativa": .5,
        "TT_DESDE_PICO": 400., "Termoinhibida": False,
    })


@pytest.mark.parametrize("ratio,level", [
    (0., "Nula"), (1e-10, "Baja"), (.10, "Baja"), (.249999, "Baja"), (.25, "Media"),
    (.250001, "Media"), (.60, "Media"), (.749999, "Media"), (.75, "Media"),
    (.750001, "Alta"), (.9, "Alta"), (1.3, "Alta"),
])
def test_intensity_thresholds_and_future_flow_sum(reference, ratio, level):
    peak = historical_weekly_max(reference, CUTOFF)
    frame = forecast(peak * ratio)
    # El indicador debe analizar los flujos, sin clasificar el acumulado absoluto.
    frame["EMERAC_TWIN"] = .98
    snapshot = build_twin_snapshot(frame, "test", CUTOFF, "test", seasonal_reference=reference)
    assert snapshot["increment_7d"] == pytest.approx(peak * ratio)
    assert snapshot["intensity_7d_ratio"] == pytest.approx(ratio)
    assert snapshot["intensity_7d"] == level
    assert snapshot["forecast_days_7d"] == 7


@pytest.mark.parametrize("cutoff", ["2026-05-05", "2026-09-22", "2027-05-05", "2028-05-05"])
def test_peak_matches_complete_historical_bars_and_available_pool(cutoff):
    cutoff = pd.Timestamp(cutoff)
    reference = load_local_seasonal_reference(ROOT, as_of=cutoff)
    expected_years = "2008, 2009, 2011, 2012, 2013, 2014, 2023, 2024"
    if cutoff >= pd.Timestamp("2026-08-30"):
        expected_years += ", 2026"
    assert reference.Campanas_Anos.iloc[0] == expected_years
    frame = forecast(cutoff=cutoff)
    weekly, _ = trajectory_charts(frame, None, cutoff, seasonal_reference=reference, flow_frequency="Semanal")
    historical = weekly.data[0]
    complete = [float(value) / 100 for value, pattern in zip(historical.y, historical.marker.pattern.shape) if pattern == ""]
    assert historical_weekly_max(reference, cutoff) == pytest.approx(max(complete))
    result = weekly_flow_intensity(frame, cutoff, reference)
    assert result["intensity_7d_ratio"] == pytest.approx(.12 / max(complete))


@pytest.mark.parametrize("days", [0, 3, 6])
def test_partial_horizon_is_not_classified_as_low(reference, days):
    frame = forecast(total=0.).iloc[:days + 1]
    result = weekly_flow_intensity(frame, CUTOFF, reference)
    assert result["intensity_7d"] == ("Sin pronóstico" if days == 0 else "Pronóstico incompleto")
    assert result["forecast_days_7d"] == days
    assert result["increment_7d"] is None
    assert result["intensity_7d_ratio"] is None


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "infinite", "negative"])
def test_seven_calendar_days_must_be_valid_without_using_later_rows(reference, fault):
    frame = forecast()
    if fault == "missing":
        frame = frame.drop(index=3)
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    else:
        frame.loc[3, "EMERREL_TWIN"] = {"nan": np.nan, "infinite": np.inf, "negative": -.01}[fault]
    result = weekly_flow_intensity(frame, CUTOFF, reference)
    assert result["forecast_days_7d"] == 6
    assert result["intensity_7d"] == "Pronóstico incompleto"
    assert result["intensity_7d_ratio"] is None


def test_partial_historical_weeks_cannot_set_the_peak():
    # 1–3 enero 2027: semana parcial con 90 %; 4–10 enero: completa con 7 %.
    reference = pd.DataFrame({
        "Julian_days": np.arange(1, 15),
        "Progreso_Mediano": np.cumsum([.3] * 3 + [.01] * 7 + [.0075] * 4),
        "Campanas_Anos": "prueba",
    })
    assert historical_weekly_max(reference, CUTOFF) == pytest.approx(.07)


@pytest.mark.parametrize("mode", ["missing", "zero"])
def test_missing_or_zero_historical_peak_is_not_a_low_intensity(reference, mode):
    if mode == "missing":
        reference = None
    else:
        reference = reference.copy()
        reference["Progreso_Mediano"] = 0.
    result = weekly_flow_intensity(forecast(), CUTOFF, reference)
    assert result["increment_7d"] == pytest.approx(.12)
    assert result["historical_weekly_max"] is None
    assert result["intensity_7d"] == "Sin referencia"
    assert result["intensity_7d_ratio"] is None


def test_complete_zero_flow_is_null_even_without_historical_peak():
    result = weekly_flow_intensity(forecast(total=0.), CUTOFF, None)
    assert result["forecast_days_7d"] == 7
    assert result["increment_7d"] == 0.
    assert result["intensity_7d"] == "Nula"
    assert result["intensity_7d_ratio"] is None
