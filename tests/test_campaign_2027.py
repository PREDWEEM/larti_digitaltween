"""Reinicio automático en 2027 con MeteoBahía, sin acceso a la red.

Las series de 2027 de estas pruebas son fixtures sintéticos.
"""

import contextlib
import io
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import campaign
import update_meteo as updater
from predweem_twin import weather

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_GLOBALS = ("CAMPANIA_ANIO", "CAMPANIA_START", "CAMPANIA_END",
                    "ARCHIVO_MAESTRO", "ARCHIVO_HISTORICO", "ARCHIVO_ESTADO")


def forecast(today):
    frame = pd.DataFrame({
        "Fecha": pd.date_range(today, today + timedelta(days=6)).strftime("%Y-%m-%d"),
        "TMAX": 30.0, "TMIN": 15.0, "Prec": 0.0,
    })
    return updater.normalizar(
        frame, fuente=updater.FUENTE_PRONOSTICO, tipo="Pronostico",
        calidad="Pronostico_deterministico_Coronel_Falcon", emision_utc="test",
    )


@pytest.fixture
def campaign_2027(monkeypatch, tmp_path):
    for name in CAMPAIGN_GLOBALS:  # monkeypatch restaura 2026 al terminar.
        monkeypatch.setattr(updater, name, getattr(updater, name))
    monkeypatch.chdir(tmp_path)
    updater.configurar_campania(2027)
    monkeypatch.setattr(updater, "DIR_PRONOSTICOS", tmp_path / "forecasts")
    return tmp_path


def run_day(monkeypatch, today):
    monkeypatch.setattr(updater, "hoy_argentina", lambda: today)
    monkeypatch.setattr(updater, "meteobahia", lambda: forecast(today))
    with contextlib.redirect_stdout(io.StringIO()):
        return updater.ejecutar()


def test_campaign_rolls_over_on_january_first_with_separate_files():
    assert campaign.default_campaign_year(date(2026, 12, 31)) == 2026
    assert campaign.default_campaign_year(date(2027, 1, 1)) == 2027
    assert campaign.campaign_weather_path(2026) == Path("meteo_daily.csv")
    assert campaign.campaign_weather_path(2027) == Path("meteo_daily_2027.csv")
    assert updater.archivo_historico_campania(2026) == updater.ARCHIVO_HISTORICO
    assert (ROOT / updater.ARCHIVO_HISTORICO).exists()


def test_first_days_of_2027_build_their_own_archive(monkeypatch, campaign_2027):
    first = run_day(monkeypatch, date(2027, 1, 1))
    assert first.Fecha.min() == "2027-01-01"
    assert first.TipoDato.eq("Pronostico").all()
    assert (campaign_2027 / "meteo_daily_2027.csv").exists()
    assert (campaign_2027 / "data/meteo_falcon_pronosticos_archivados_2027.csv").exists()
    state = json.loads((campaign_2027 / "data/estado_actualizacion_meteo_2027.json").read_text())
    assert state["campania"] == 2027 and state["fin_campania"] == "2027-10-01"
    assert state["inicio_historico"] is None and state["huecos_finales"] == []

    second = run_day(monkeypatch, date(2027, 1, 2))
    past = second[second.Fecha < "2027-01-02"]
    assert past.Fecha.tolist() == ["2027-01-01"]
    assert past.TipoDato.eq(updater.TIPO_HISTORICO).all()
    assert not (campaign_2027 / "meteo_daily.csv").exists()


def test_missing_archive_after_first_day_is_an_error(monkeypatch, campaign_2027):
    with pytest.raises(FileNotFoundError):
        run_day(monkeypatch, date(2027, 1, 3))


def test_main_refuses_a_campaign_that_has_not_started(monkeypatch, campaign_2027):
    monkeypatch.setattr(updater, "hoy_argentina", lambda: date(2026, 11, 22))

    def forbidden():
        raise AssertionError("No debe consultar MeteoBahía para una campaña futura")

    monkeypatch.setattr(updater, "meteobahia", forbidden)
    with contextlib.redirect_stderr(io.StringIO()):
        assert updater.main(["--year", "2027"]) == 1
    assert not list(campaign_2027.rglob("*.csv"))


def test_app_opens_latest_campaign_with_weather(tmp_path):
    legacy = tmp_path / campaign.campaign_weather_path(2026)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("Fecha\n2026-01-01\n")
    assert campaign.initial_campaign_year(tmp_path, date(2026, 11, 22)) == 2026
    # 1 de enero antes de la primera actualización: sigue mostrando 2026.
    assert campaign.initial_campaign_year(tmp_path, date(2027, 1, 1)) == 2026
    (tmp_path / campaign.campaign_weather_path(2027)).write_text("Fecha\n2027-01-01\n")
    assert campaign.initial_campaign_year(tmp_path, date(2027, 1, 1)) == 2027
    assert campaign.initial_campaign_year(tmp_path, date(2026, 12, 31)) == 2026


def test_app_weather_is_limited_to_the_selected_campaign(tmp_path):
    path = tmp_path / "both.csv"
    pd.DataFrame({"Fecha": ["2026-05-05", "2027-05-05", "2027-10-02"],
                  "TMAX": 20., "TMIN": 10., "Prec": [9., 2., 3.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="varios años"):
        weather.read_weather_file(path)
    assert weather.read_weather_file(path, campaign_year=2027).Prec.tolist() == [2.]
    frame = pd.DataFrame({"Fecha": pd.date_range("2027-09-25", "2027-10-10")})
    window, meta = weather.operational_weather_window(frame, as_of="2027-09-28")
    assert window.Fecha.max() == pd.Timestamp("2027-10-01")
    assert meta["campaign_year"] == 2027


def test_app_runs_2027_campaign_from_its_own_file(tmp_path):
    for pattern in ("*.py", "*.json"):
        for source in ROOT.glob(pattern):
            shutil.copy2(source, tmp_path / source.name)
    shutil.copy2(ROOT / "meteo_daily.csv", tmp_path / "meteo_daily.csv")
    for directory in ("models", "predweem_twin", "data"):
        shutil.copytree(ROOT / directory, tmp_path / directory,
                        ignore=shutil.ignore_patterns("*.db", "*.db-shm", "*.db-wal", "__pycache__",
                                                      "historico_pronosticos"))
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
    assert not app.exception, [error.message for error in app.exception]
    app.selectbox(key="campaign_year").set_value(2027).run()
    assert not app.exception, [error.message for error in app.exception]
    assert any("2027" in info.value for info in app.info)

    history = pd.DataFrame({"Fecha": pd.date_range("2027-01-01", "2027-05-04").strftime("%Y-%m-%d"),
                            "TMAX": 25.0, "TMIN": 12.0, "Prec": 1.0})
    history = updater.normalizar(history, fuente=updater.FUENTE_HISTORICA, tipo=updater.TIPO_HISTORICO,
                                 calidad=updater.CALIDAD_HISTORICA, emision_utc="test")
    pd.concat([history, forecast(date(2027, 5, 5))]).to_csv(tmp_path / "meteo_daily_2027.csv", index=False)
    app.run()
    assert not app.exception, [error.message for error in app.exception]
    assert app.date_input(key="state_date_2027").value.year == 2027
