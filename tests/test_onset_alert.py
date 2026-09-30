"""Horizonte preventivo, evidencia disponible y conservación del reloj térmico."""

import numpy as np
import pandas as pd
import pytest

from predweem_twin.onset import onset_alert


CUTOFF = pd.Timestamp("2027-02-09")


def trajectory(onset="2027-02-16", periods=20):
    dates = pd.date_range("2027-02-01", periods=periods)
    active = dates >= pd.Timestamp(onset)
    return pd.DataFrame({
        "Fecha": dates, "Primer_Pico_Habilitado": active,
        "EMERREL": np.where(active, .5, 0.),
        "TT_DESDE_PICO": np.cumsum(np.where(active, 10., 0.)),
        "TIPODATO": np.where(dates > CUTOFF, "Pronostico", "Observado"),
        "EMISION_UTC": "2027-02-09T09:00:00Z",
    })


@pytest.mark.parametrize("days,status", [(0, "started"), (1, "watch"), (7, "watch"), (8, "no_signal")])
def test_inclusive_seven_day_window_without_future_leakage(days, status):
    frame = trajectory(CUTOFF + pd.Timedelta(days=days))
    result = onset_alert(frame, CUTOFF)
    assert result["status"] == status
    if days == 8:
        assert result["model_onset_date"] is None


def test_toggle_and_alert_leave_entire_trajectory_and_thermal_origin_unchanged():
    frame = trajectory()
    original = frame.copy(deep=True)
    assert onset_alert(frame, CUTOFF)["status"] == "watch"
    assert onset_alert(frame, CUTOFF, enabled=False)["status"] == "disabled"
    pd.testing.assert_frame_equal(frame, original)
    assert frame.loc[frame.Fecha <= CUTOFF, "TT_DESDE_PICO"].eq(0).all()
    assert frame.loc[frame.TT_DESDE_PICO.gt(0), "Fecha"].min() == pd.Timestamp("2027-02-16")


@pytest.mark.parametrize("column", ["Flujo_observado_PLM2", "Observado"])
def test_first_positive_is_an_upper_bound_even_if_followed_by_zero(column):
    observations = pd.DataFrame({"Fecha": ["2027-02-05", "2027-02-08"], column: [21.33, 0.]})
    result = onset_alert(trajectory(), CUTOFF, observations)
    assert result["status"] == "observed"
    assert result["first_positive_date"] == "2027-02-05"
    assert "a más tardar" in result["message"]
    assert result["model_onset_date"] == "2027-02-16"


def test_future_and_previous_campaign_counts_cannot_confirm_current_emergence():
    observations = pd.DataFrame({
        "Fecha": ["2026-02-05", "2027-02-10", "2027-02-08"],
        "Flujo_observado_PLM2": [100., 100., 0.],
    })
    assert onset_alert(trajectory(), CUTOFF, observations)["status"] == "watch"


@pytest.mark.parametrize("defect", ["missing", "nan", "duplicate"])
def test_bad_horizon_is_unknown_not_no_signal(defect):
    frame = trajectory(onset="2027-03-01")
    row = frame.Fecha.eq("2027-02-12")
    if defect == "missing":
        frame = frame.loc[~row]
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.loc[row]], ignore_index=True)
    else:
        frame.loc[row, "EMERREL"] = np.nan
    result = onset_alert(frame, CUTOFF)
    assert result["status"] == "unavailable"
    assert result["forecast_days_available"] == 6


def test_partial_positive_forecast_still_warns_and_no_forecast_is_unknown():
    frame = trajectory(onset="2027-02-11")
    result = onset_alert(frame.loc[frame.Fecha.le("2027-02-12")], CUTOFF)
    assert result["status"] == "watch"
    assert result["forecast_days_available"] == 3
    assert onset_alert(frame.loc[frame.Fecha.le(CUTOFF)], CUTOFF)["status"] == "unavailable"


@pytest.mark.parametrize("kind,emission,expected", [
    ("Pronostico", "2027-02-09T09:00:00Z", "Pronóstico disponible al corte"),
    ("Observado", "2027-02-09T09:00:00Z", "Revisión retrospectiva"),
    ("Pronostico_archivado", "2027-02-09T09:00:00Z", "Revisión retrospectiva"),
    ("Pronostico", "2027-02-12T09:00:00Z", "Revisión retrospectiva"),
    ("Pronostico", None, "Pronóstico disponible; emisión no documentada"),
])
def test_retrospective_weather_is_not_presented_as_an_issued_early_warning(kind, emission, expected):
    frame = trajectory()
    frame["TIPODATO"] = kind
    frame["EMISION_UTC"] = emission
    result = onset_alert(frame, CUTOFF)
    assert result["mode"].startswith(expected)
    assert result["status"] == "watch"
