"""Escenarios meteorológicos contrafactuales sin alterar el estado observado."""

from __future__ import annotations

import pandas as pd


def apply_scenario(
    weather: pd.DataFrame,
    as_of,
    rain_mm: float = 0.0,
    rain_days: int = 3,
    temperature_delta: float = 0.0,
) -> pd.DataFrame:
    scenario = weather.copy()
    date_column = "Fecha" if "Fecha" in scenario.columns else "FECHA"
    scenario[date_column] = pd.to_datetime(scenario[date_column], errors="coerce")
    future = scenario[date_column] > pd.Timestamp(as_of)
    future_indices = scenario.index[future].tolist()
    if rain_mm > 0 and future_indices:
        selected = future_indices[: max(1, int(rain_days))]
        precipitation_column = "Prec" if "Prec" in scenario.columns else "PREC"
        scenario.loc[selected, precipitation_column] = (
            pd.to_numeric(scenario.loc[selected, precipitation_column], errors="coerce").fillna(0.0)
            + float(rain_mm) / len(selected)
        )
    if temperature_delta and future_indices:
        for column in ("TMAX", "TMIN"):
            scenario.loc[future_indices, column] = (
                pd.to_numeric(scenario.loc[future_indices, column], errors="coerce")
                + float(temperature_delta)
            )
    return scenario
