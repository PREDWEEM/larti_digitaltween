"""Flujos históricos compartidos por los gráficos y la intensidad semanal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_historical_reference(reference, as_of):
    """Traslada el pool al calendario consultado sin inventar una cola anual.

    Se usa el eje común JD_common del archivo histórico y el acumulado 2026.
    El calendario de visualización trata ese eje como días de un año de 365
    días; en años bisiestos interpola el 29 de febrero. No reconstruye las
    fechas originales de los ocho archivos identificados sólo por año.
    Fuera del eje histórico se deja NaN, no un supuesto de emergencia nula.
    El flujo diario es derivado del acumulado, no un conteo diario observado.
    """
    year = pd.Timestamp(as_of).year
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    frame = pd.DataFrame({"Fecha": dates})
    days = dates.dayofyear.to_numpy(dtype=float)
    days[(dates.is_leap_year) & (dates.month > 2)] -= 1
    days[(dates.month == 2) & (dates.day == 29)] = 59.5
    axis = reference["Julian_days"].to_numpy(float)
    columns = ["Progreso_Mediano"] + [
        column for column in reference.columns
        if column.startswith("Progreso_") and column.removeprefix("Progreso_").isdigit()
    ]
    for column in columns:
        frame[column] = np.interp(
            days, axis, reference[column].to_numpy(float),
            left=np.nan, right=np.nan,
        )
    # El resumen operativo conserva su supuesto de normalización hasta el
    # final del eje; la curva individual 2026 muestra sólo su ventana real.
    source_2026 = reference.attrs.get("source_2026", {})
    if "Progreso_2026" in frame and source_2026.get("end"):
        end_day = pd.Timestamp(source_2026["end"]).dayofyear
        frame.loc[days > end_day, "Progreso_2026"] = np.nan
    frame["Flujo_Diario"] = frame["Progreso_Mediano"].diff().clip(lower=0)
    if axis[0] == 1:
        frame.loc[0, "Flujo_Diario"] = frame.loc[0, "Progreso_Mediano"]
    frame.attrs["campaigns"] = reference["Campanas_Anos"].iloc[0]
    return frame


def weekly_flow_groups(dates, values):
    """Agrupa flujos diarios de lunes a domingo sin completar datos faltantes."""
    frame = pd.DataFrame({"Fecha": pd.to_datetime(dates), "Flujo": values})
    frame["Semana"] = frame["Fecha"] - pd.to_timedelta(frame["Fecha"].dt.dayofweek, unit="D")
    return frame.groupby("Semana", sort=True)


def historical_weekly_max(reference, as_of):
    """Máximo de semanas completas del pool visible (enero–1 de octubre).

    Usa el mismo calendario, resumen histórico y escala fraccional del gráfico.
    No toma el máximo diario ni el máximo de una campaña individual.
    """
    if reference is None or reference.empty:
        return None
    historical = annual_historical_reference(reference, as_of)
    end = pd.Timestamp(pd.Timestamp(as_of).year, 10, 1)
    historical = historical.loc[historical["Fecha"] <= end]
    totals = []
    for _, group in weekly_flow_groups(historical["Fecha"], historical["Flujo_Diario"]):
        valid = np.isfinite(group["Flujo"]) & group["Flujo"].ge(0)
        if group.loc[valid, "Fecha"].nunique() == 7:
            totals.append(float(group.loc[valid, "Flujo"].sum()))
    peak = max(totals, default=0.0)
    return peak if peak > 0 else None
