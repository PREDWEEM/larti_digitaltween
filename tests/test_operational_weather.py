import pandas as pd
from pathlib import Path

from predweem_twin.weather import (
    last_observed_weather_date,
    operational_weather_window,
    forecast_mask,
)


def test_meteobahia_archived_forecasts_are_historical_not_future():
    frame = pd.DataFrame({
        "Fecha": pd.date_range("2026-09-17", periods=4),
        "TipoDato": ["Historico_pronostico", "Historico_pronostico", "Pronostico", "Pronostico"],
    })
    assert forecast_mask(frame).tolist() == [False, False, True, True]
    assert last_observed_weather_date(frame) == pd.Timestamp("2026-09-18")
    window, meta = operational_weather_window(frame)
    assert meta["as_of"] == pd.Timestamp("2026-09-18")
    assert meta["forecast_days_available"] == 2
    assert not meta["complete"]


def test_calibration_weather_is_continuous_archived_not_station_observations():
    data = Path(__file__).parents[1] / "data/calibration/lartigau_2026_weather.csv"
    frame = pd.read_csv(data)
    dates = pd.to_datetime(frame.Fecha)
    assert dates.tolist() == pd.date_range("2026-01-01", "2026-08-30").tolist()
    assert frame.TipoDato.eq("Historico_pronostico").all()
    assert frame.Fuente.eq("METEOBAHIA_XML_ARCHIVADO").all()
    assert not forecast_mask(frame).any()


def test_operational_window_uses_last_non_forecast_date_and_seven_days():
    dates = pd.date_range("2026-03-29", periods=10, freq="D")
    weather = pd.DataFrame(
        {
            "Fecha": dates,
            "TMAX": 20.0,
            "TMIN": 10.0,
            "Prec": 0.0,
            "TipoDato": ["Observado", "Provisional", *(["Pronostico"] * 8)],
        }
    )

    cutoff = last_observed_weather_date(weather)
    window, metadata = operational_weather_window(weather, forecast_days=7)

    assert cutoff == pd.Timestamp("2026-03-30")
    assert metadata["as_of"] == pd.Timestamp("2026-03-30")
    assert metadata["forecast_days_available"] == 7
    assert metadata["forecast_end"] == pd.Timestamp("2026-04-06")
    assert metadata["complete"]
    assert window["Fecha"].max() == pd.Timestamp("2026-04-06")


def test_partial_file_without_forecast_is_reported_as_incomplete():
    weather = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", "2026-03-30", freq="D"),
            "TMAX": 20.0,
            "TMIN": 10.0,
            "Prec": 0.0,
        }
    )

    _, metadata = operational_weather_window(weather, forecast_days=7)

    assert metadata["as_of"] == pd.Timestamp("2026-03-30")
    assert metadata["forecast_days_available"] == 0
    assert not metadata["complete"]
