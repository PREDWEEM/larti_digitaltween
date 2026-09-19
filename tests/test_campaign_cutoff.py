"""Cierre de campaña y procedencia del histórico MeteoBahía, sin red."""

from datetime import date

import pandas as pd
import pytest

import update_meteo as updater
from predweem_twin.weather import operational_weather_window


@pytest.mark.parametrize("cutoff,expected", [("2026-09-28", 3), ("2026-10-01", 0), ("2026-10-10", 0)])
def test_twin_horizon_stops_at_campaign_end(cutoff, expected):
    frame = pd.DataFrame({"Fecha": pd.date_range("2026-09-25", "2026-10-10")})
    window, metadata = operational_weather_window(frame, as_of=cutoff)
    assert window.Fecha.max() == pd.Timestamp("2026-10-01")
    assert metadata["forecast_days_expected"] == expected
    assert metadata["forecast_days_available"] == expected
    assert metadata["complete"]


@pytest.mark.parametrize("today", [date(2026, 9, 29), date(2026, 10, 1), date(2026, 10, 8)])
def test_meteobahia_update_keeps_archived_origin_and_closes(monkeypatch, tmp_path, today):
    dates = pd.date_range("2026-01-01", "2026-10-01")
    raw = pd.DataFrame({"Fecha": dates, "TMAX": 20., "TMIN": 10., "Prec": 1.})
    history = updater.normalizar(
        raw, fuente=updater.FUENTE_HISTORICA, tipo=updater.TIPO_HISTORICO,
        calidad=updater.CALIDAD_HISTORICA, emision_utc="No_disponible_archivo",
    )
    future = updater.normalizar(
        raw.loc[raw.Fecha.dt.date.ge(today)], fuente=updater.FUENTE_PRONOSTICO,
        tipo="Pronostico", calidad="Pronostico_deterministico_Coronel_Falcon",
        emision_utc="test",
    )
    monkeypatch.setattr(updater, "hoy_argentina", lambda: today)
    monkeypatch.setattr(updater, "leer_archivo_historico", lambda: history)
    monkeypatch.setattr(updater, "meteobahia", lambda: future)
    for attribute, filename in (
        ("ARCHIVO_MAESTRO", "meteo.csv"), ("ARCHIVO_HISTORICO", "history.csv"),
        ("ARCHIVO_ESTADO", "state.json"), ("DIR_PRONOSTICOS", "forecasts"),
    ):
        monkeypatch.setattr(updater, attribute, tmp_path / filename)
    result = updater.ejecutar()
    assert len(result) == 274
    assert result.Fecha.max() == "2026-10-01"
    historical = pd.to_datetime(result.Fecha).dt.date.lt(today)
    assert result.loc[historical, "TipoDato"].eq("Historico_pronostico").all()
    assert result.loc[historical, "Fuente"].eq("METEOBAHIA_XML_ARCHIVADO").all()
    if today > updater.CAMPANIA_END:
        assert not result.TipoDato.eq("Pronostico").any()


def test_closed_campaign_does_not_fetch_live_forecast(monkeypatch):
    monkeypatch.setattr(updater, "hoy_argentina", lambda: date(2026, 10, 2))
    def unexpected_request(*args, **kwargs):
        raise AssertionError("No se debe consultar el XML después del cierre")
    monkeypatch.setattr(updater, "get", unexpected_request)
    assert updater.meteobahia().empty
