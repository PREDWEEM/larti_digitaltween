"""Aviso por email de la alerta de inicio (issue de GitHub)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("alerta_inicio", ROOT / "scripts/alerta_inicio.py")
alerta_inicio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alerta_inicio)

WATCH = {
    "status": "watch", "title": "Alerta preventiva: posible inicio en los próximos 7 días",
    "message": "Inicio modelado para el 20/02/2027 (dentro de 3 días).", "horizon_days": 7,
    "model_onset_date": "2027-02-20", "mode": "Pronóstico disponible al corte",
    "campania": 2027, "corte": "2027-02-17",
}


def test_notice_only_for_onset_within_seven_days(monkeypatch, tmp_path):
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(alerta_inicio, "calcular_alerta", lambda: dict(WATCH))
    assert alerta_inicio.main(["--cuerpo", str(tmp_path / "c.md")]) == 0
    lines = output.read_text().splitlines()
    sitio = alerta_inicio.nombre_sitio()
    assert "avisar=true" in lines
    assert f"titulo=Alerta de inicio de emergencia · {sitio} · campaña 2027" in lines
    body = (tmp_path / "c.md").read_text()
    assert body.startswith("@PREDWEEM ") and "2027-02-20" in body and "streamlit.app" in body

    output.write_text("")
    monkeypatch.setattr(alerta_inicio, "calcular_alerta", lambda: {**WATCH, "status": "no_signal"})
    alerta_inicio.main(["--cuerpo", str(tmp_path / "otro.md")])
    assert "avisar=false" in output.read_text().splitlines()
    assert not (tmp_path / "otro.md").exists()


def test_reads_the_alert_the_app_shows():
    alerta = alerta_inicio.calcular_alerta()
    assert alerta["status"] in {"disabled", "observed", "started", "watch", "unavailable", "no_signal"}
    assert alerta["horizon_days"] == 7
    assert alerta["campania"] >= 2026
