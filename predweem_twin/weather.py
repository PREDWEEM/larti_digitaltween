"""Fuentes meteorológicas: archivo operativo y Open-Meteo."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from campaign import CAMPANIA_END


def _weather_date_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().lower() in {"fecha", "date", "datetime"}:
            return column
    raise ValueError("La meteorología requiere una columna Fecha.")



def limit_weather_to_campaign(frame: pd.DataFrame) -> pd.DataFrame:
    """Excluye fechas posteriores al cierre, también en archivos del usuario."""
    dates = pd.to_datetime(frame[_weather_date_column(frame)], errors="coerce").dt.tz_localize(None)
    result = frame.loc[dates.dt.date <= CAMPANIA_END].copy()
    if result.empty:
        raise ValueError("No hay datos meteorológicos hasta el 01/10/2026.")
    return result.reset_index(drop=True)

def forecast_mask(frame: pd.DataFrame) -> pd.Series:
    """Identifica pronósticos vigentes; el pronóstico archivado es histórico."""
    type_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower() in {"tipodato", "tipo_dato", "data_type"}
        ),
        None,
    )
    if type_column is None:
        return pd.Series(False, index=frame.index)
    labels = frame[type_column].astype(str).str.lower().str.strip()
    archived = labels.str.contains("historico|histórico|archivad|archived", regex=True, na=False)
    return labels.str.contains("pronost|forecast", regex=True, na=False) & ~archived


def last_observed_weather_date(frame: pd.DataFrame):
    """Última fecha histórica disponible; no implica observación de estación."""
    date_column = _weather_date_column(frame)
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.tz_localize(None)
    observed = dates[~forecast_mask(frame) & dates.notna()]
    if observed.empty:
        valid = dates.dropna()
        if valid.empty:
            raise ValueError("No hay fechas meteorológicas válidas.")
        return valid.max()
    return observed.max()


def operational_weather_window(
    frame: pd.DataFrame,
    as_of=None,
    forecast_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Recorta la meteorología al estado observado más siete días."""
    if int(forecast_days) < 1:
        raise ValueError("El horizonte de pronóstico debe ser al menos un día.")
    date_column = _weather_date_column(frame)
    prepared = limit_weather_to_campaign(frame)
    prepared[date_column] = pd.to_datetime(
        prepared[date_column], errors="coerce"
    ).dt.tz_localize(None)
    prepared = prepared.dropna(subset=[date_column]).sort_values(date_column)
    cutoff = (
        pd.Timestamp(as_of).tz_localize(None).normalize()
        if as_of is not None
        else pd.Timestamp(last_observed_weather_date(prepared)).normalize()
    )
    campaign_end = pd.Timestamp(CAMPANIA_END)
    cutoff = min(cutoff, campaign_end)
    horizon_end = min(cutoff + pd.Timedelta(days=int(forecast_days)), campaign_end)
    expected_dates = pd.date_range(cutoff + pd.Timedelta(days=1), horizon_end, freq="D")
    window = prepared[prepared[date_column] <= horizon_end].copy()
    future_dates = window.loc[window[date_column] > cutoff, date_column].dt.normalize().drop_duplicates()
    available = int(len(future_dates))
    return window.reset_index(drop=True), {
        "as_of": cutoff,
        "forecast_end": future_dates.max() if available else None,
        "campaign_end": campaign_end,
        "campaign_closed": cutoff >= campaign_end,
        "forecast_days_requested": int(forecast_days),
        "forecast_days_expected": len(expected_dates),
        "forecast_days_available": min(available, int(forecast_days)),
        "complete": expected_dates.difference(pd.DatetimeIndex(future_dates)).empty,
    }


def read_weather_file(source) -> pd.DataFrame:
    if hasattr(source, "name"):
        suffix = Path(source.name).suffix.lower()
    else:
        suffix = Path(source).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return limit_weather_to_campaign(pd.read_excel(source))
    return limit_weather_to_campaign(pd.read_csv(source))


def _today_argentina():
    return pd.Timestamp.now(tz="America/Argentina/Buenos_Aires").date()


def fetch_open_meteo(latitude: float, longitude: float, start_date, forecast_days: int = 16) -> pd.DataFrame:
    """Combina archivo histórico y pronóstico hasta el cierre inclusivo."""
    if not 1 <= int(forecast_days) <= 16:
        raise ValueError("El horizonte Open-Meteo debe estar entre 1 y 16 días.")
    start = pd.Timestamp(start_date).date()
    today = _today_argentina()
    end = min(today + timedelta(days=int(forecast_days) - 1), CAMPANIA_END)
    if start > end:
        raise ValueError("La fecha inicial supera el cierre del 01/10/2026.")
    history_end = min(today - timedelta(days=6), end)
    daily = "temperature_2m_max,temperature_2m_min,precipitation_sum"
    payloads = []
    ranges = (
        (start, history_end, "https://archive-api.open-meteo.com/v1/archive", "OPEN_METEO_ERA5", "Historico"),
        (max(start, history_end + timedelta(days=1)), end, "https://api.open-meteo.com/v1/forecast", "OPEN_METEO_FORECAST", "Pronostico"),
    )
    for first, last, url, source, data_type in ranges:
        if first > last:
            continue
        response = requests.get(
            url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": first.isoformat(),
                "end_date": last.isoformat(),
                "daily": daily,
                "timezone": "America/Argentina/Buenos_Aires",
            },
            timeout=45,
        )
        response.raise_for_status()
        payloads.append((response.json(), source, data_type))

    frames = []
    for payload, source, data_type in payloads:
        block = payload["daily"]
        frame = pd.DataFrame(
            {
                "Fecha": block["time"],
                "TMAX": block["temperature_2m_max"],
                "TMIN": block["temperature_2m_min"],
                "Prec": block["precipitation_sum"],
                "Fuente": source,
                "TipoDato": data_type,
            }
        )
        if data_type == "Pronostico":
            frame_dates = pd.to_datetime(frame["Fecha"]).dt.date
            frame["TipoDato"] = [
                "Provisional" if value < today else "Pronostico"
                for value in frame_dates
            ]
        frames.append(frame)
    result = (
        pd.concat(frames, ignore_index=True)
        .assign(Fecha=lambda frame: pd.to_datetime(frame["Fecha"]))
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )

    result = result.loc[result["Fecha"].dt.date >= start]
    return limit_weather_to_campaign(result)


def weather_source_label(df: pd.DataFrame) -> str:
    if "Fuente" not in df.columns:
        return "Archivo aportado"
    sources = [str(value) for value in df["Fuente"].dropna().unique()]
    return " + ".join(sources[:3]) if sources else "Archivo aportado"
