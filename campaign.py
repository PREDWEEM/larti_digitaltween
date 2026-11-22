"""Campañas meteorológicas operativas: del 1 de enero al 1 de octubre de cada año.

La campaña activa se elige por el año calendario de Argentina, de modo que el
gemelo pasa solo a 2027 el 1 de enero. 2026 conserva sus archivos originales y
cada campaña nueva escribe archivos propios con el sufijo del año, para no
sobrescribir la serie de 2026.

``CAMPAIGN_YEAR`` (variable de entorno) fuerza una campaña concreta, por ejemplo
para regenerar 2026 o para las pruebas automáticas.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ZONA_HORARIA = "America/Argentina/Buenos_Aires"
CAMPAIGN_YEARS = (2026, 2027)
LEGACY_YEAR = 2026
WEATHER_FILE = Path("meteo_daily.csv")


def campaign_bounds(year: int) -> tuple[date, date]:
    """Inicio y cierre inclusivos de la campaña meteorológica."""
    year = int(year)
    if year not in CAMPAIGN_YEARS:
        habilitadas = ", ".join(str(y) for y in CAMPAIGN_YEARS)
        raise ValueError(f"Campaña {year} no habilitada. Campañas disponibles: {habilitadas}.")
    return date(year, 1, 1), date(year, 10, 1)


def default_campaign_year(today: date | None = None) -> int:
    """Campaña del año calendario argentino, acotada a las habilitadas."""
    today = today or datetime.now(ZoneInfo(ZONA_HORARIA)).date()
    return min(max(today.year, CAMPAIGN_YEARS[0]), CAMPAIGN_YEARS[-1])


def active_campaign_year(today: date | None = None) -> int:
    forced = os.getenv("CAMPAIGN_YEAR", "").strip()
    if forced:
        campaign_bounds(int(forced))
        return int(forced)
    return default_campaign_year(today)


def campaign_file(path: str | Path, year: int) -> Path:
    """2026 usa el archivo original; otras campañas agregan ``_<año>``."""
    campaign_bounds(year)
    path = Path(path)
    if int(year) == LEGACY_YEAR:
        return path
    return path.with_name(f"{path.stem}_{int(year)}{path.suffix}")


def campaign_weather_path(year: int) -> Path:
    return campaign_file(WEATHER_FILE, year)


def initial_campaign_year(base: str | Path = ".", today: date | None = None) -> int:
    """Campaña que la app abre: la actual si ya tiene meteorología; si no, la última con datos.

    Evita una pantalla vacía entre la medianoche del 1 de enero y la primera
    actualización programada de la campaña nueva.
    """
    current = default_campaign_year(today)
    available = [
        year for year in CAMPAIGN_YEARS
        if year <= current and (Path(base) / campaign_weather_path(year)).exists()
    ]
    return available[-1] if available else current


CAMPAIGN_YEAR = active_campaign_year()
CAMPANIA_START, CAMPANIA_END = campaign_bounds(CAMPAIGN_YEAR)
