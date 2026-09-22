"""Construcción del estado operativo diario del gemelo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .flows import historical_weekly_max


@dataclass(frozen=True)
class TwinSnapshot:
    site_id: str
    as_of: str
    emergence: float
    remaining: float
    intensity_7d: str
    increment_7d: float | None
    historical_weekly_max: float | None
    intensity_7d_ratio: float | None
    forecast_days_7d: int
    intensity_7d_reason: str
    soil_water: float
    soil_water_fraction: float
    thermal_time: float
    thermal_control_stage: str
    thermoinhibited: bool
    cohort_exhausted: bool
    next_cohort_start: str | None
    next_cohort_end: str | None
    weather_source: str
    assimilated_observations: int
    emergence_density_plm2: float | None
    seasonal_potential_plm2: float | None
    last_observation_date: str | None
    assimilation_mode: str


def thermal_control_stage(thermal_time: float) -> str:
    """Clasifica el TT desde el primer pico, sin redondear sus límites."""
    if not np.isfinite(thermal_time):
        return "SIN DATOS"
    if thermal_time < 600:
        return "AUN NO CONTROLAR"
    if thermal_time <= 700:
        return "CONTROL A TIEMPO"
    if thermal_time <= 800:
        return "ULTIMO PLAZO"
    return "FUERA DE CONTROL"


def _intensity(ratio: float) -> str:
    if ratio == 0:
        return "Nula"
    # Tolera únicamente el redondeo numérico en los límites inclusivos.
    if ratio < 0.25 and not np.isclose(ratio, 0.25, rtol=0, atol=1e-12):
        return "Baja"
    if ratio <= 0.75 or np.isclose(ratio, 0.75, rtol=0, atol=1e-12):
        return "Media"
    return "Alta"


def weekly_flow_intensity(trajectory, as_of, seasonal_reference=None) -> dict:
    """Compara el flujo de t+1 a t+7 con el pico semanal del pool histórico.

    Ambos flujos son fracciones de sus respectivos totales estacionales.
    Se requieren siete fechas consecutivas con flujo válido. Un flujo nulo
    determina intensidad Nula; para flujos positivos se necesita un pico histórico.
    Un día ausente, duplicado o inválido no se interpreta como flujo cero.
    """
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    dates = pd.to_datetime(trajectory["Fecha"]).dt.tz_localize(None).dt.normalize()
    flows = pd.to_numeric(trajectory["EMERREL_TWIN"], errors="coerce")
    daily = pd.Series(flows.to_numpy(), index=dates)
    daily = daily.where(np.isfinite(daily) & daily.ge(0))
    # Un duplicado hace que esa fecha no sea evaluable, sin sumar dos veces.
    daily = daily.loc[~daily.index.duplicated(keep=False)]
    horizon = pd.date_range(cutoff + pd.Timedelta(days=1), periods=7)
    future = daily.reindex(horizon)
    available = int(future.notna().sum())
    peak = historical_weekly_max(seasonal_reference, cutoff)
    result = {
        "intensity_7d": "Sin pronóstico" if available == 0 else "Pronóstico incompleto",
        "increment_7d": None,
        "historical_weekly_max": peak,
        "intensity_7d_ratio": None,
        "forecast_days_7d": available,
        "intensity_7d_reason": f"Flujo previsto disponible para {available}/7 días; no se asigna nivel de intensidad.",
    }
    if available < 7:
        return result
    total = float(future.sum())
    result["increment_7d"] = total
    if total == 0:
        result.update(
            intensity_7d="Nula",
            intensity_7d_ratio=0.0 if peak is not None else None,
            intensity_7d_reason="Flujo previsto igual a cero en los siete días completos: intensidad Nula.",
        )
        return result
    if peak is None:
        result.update(
            intensity_7d="Sin referencia",
            intensity_7d_reason="No hay un máximo semanal histórico positivo con siete días válidos.",
        )
        return result
    ratio = total / peak
    result.update(
        intensity_7d=_intensity(ratio),
        intensity_7d_ratio=ratio,
        intensity_7d_reason="Flujo previsto en siete días / máximo de semanas completas del pool histórico.",
    )
    return result


def _next_cohort(df: pd.DataFrame, idx: int, threshold: float = 0.01):
    future = df.loc[idx + 1 :].copy()
    active = future[future["EMERREL_TWIN"] >= threshold]
    if active.empty:
        return None, None
    start_idx = active.index[0]
    end_idx = start_idx
    for candidate in range(start_idx + 1, len(df)):
        if float(df.at[candidate, "EMERREL_TWIN"]) < threshold:
            break
        end_idx = candidate
    return df.at[start_idx, "Fecha"], df.at[end_idx, "Fecha"]


def build_twin_snapshot(
    trajectory: pd.DataFrame,
    site_id: str,
    as_of,
    weather_source: str,
    assimilated_observations: int = 0,
    seasonal_reference: pd.DataFrame | None = None,
) -> dict:
    as_of = pd.Timestamp(as_of).tz_localize(None).normalize()
    df = trajectory.sort_values("Fecha").reset_index(drop=True)
    candidates = df.index[df["Fecha"] <= as_of].tolist()
    idx = candidates[-1] if candidates else 0
    current = float(df.at[idx, "EMERAC_TWIN"])
    intensity = weekly_flow_intensity(df, as_of, seasonal_reference)
    start, end = _next_cohort(df, idx)
    potential_value = (
        float(df.at[idx, "POTENCIAL_ESTACIONAL_PLM2"])
        if "POTENCIAL_ESTACIONAL_PLM2" in df
        and pd.notna(df.at[idx, "POTENCIAL_ESTACIONAL_PLM2"])
        else None
    )
    density_value = (
        float(df.at[idx, "EMERAC_TWIN_PLM2"])
        if "EMERAC_TWIN_PLM2" in df
        and pd.notna(df.at[idx, "EMERAC_TWIN_PLM2"])
        else None
    )
    observation_dates = (
        pd.to_datetime(df["ULTIMA_OBSERVACION"], errors="coerce").dropna()
        if "ULTIMA_OBSERVACION" in df
        else pd.Series(dtype="datetime64[ns]")
    )
    last_observation = observation_dates.max() if not observation_dates.empty else None
    assimilation_mode = (
        str(df.at[idx, "MODO_ASIMILACION"])
        if "MODO_ASIMILACION" in df
        else "sin observaciones"
    )
    snapshot = TwinSnapshot(
        site_id=site_id,
        as_of=df.at[idx, "Fecha"].date().isoformat(),
        emergence=current,
        remaining=max(0.0, 1.0 - current),
        **intensity,
        soil_water=float(df.at[idx, "W_superficial"]),
        soil_water_fraction=float(df.at[idx, "Humedad_Relativa"]),
        thermal_time=float(df.at[idx, "TT_DESDE_PICO"]),
        thermal_control_stage=thermal_control_stage(float(df.at[idx, "TT_DESDE_PICO"])),
        thermoinhibited=bool(df.at[idx, "Termoinhibida"]),
        cohort_exhausted=(
            bool(df.at[idx, "Cohorte_Agotada"])
            if "Cohorte_Agotada" in df
            else False
        ),
        next_cohort_start=start.date().isoformat() if start is not None else None,
        next_cohort_end=end.date().isoformat() if end is not None else None,
        weather_source=weather_source,
        assimilated_observations=int(assimilated_observations),
        emergence_density_plm2=density_value,
        seasonal_potential_plm2=potential_value,
        last_observation_date=(
            last_observation.date().isoformat()
            if last_observation is not None and pd.notna(last_observation)
            else None
        ),
        assimilation_mode=assimilation_mode,
    )
    return asdict(snapshot)


def milestone_dates(trajectory: pd.DataFrame) -> dict[str, str | None]:
    milestones = {}
    for threshold in (0.25, 0.50, 0.75, 0.95):
        reached = trajectory[trajectory["EMERAC_TWIN"] >= threshold]
        milestones[f"d{int(threshold * 100)}"] = (
            reached.iloc[0]["Fecha"].date().isoformat() if not reached.empty else None
        )
    return milestones


def thermal_window_dates(
    trajectory: pd.DataFrame,
    lower_tt: float = 600.0,
    upper_tt: float = 800.0,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Devuelve las fechas de ingreso y salida de una ventana térmica."""
    if upper_tt <= lower_tt:
        raise ValueError("El límite térmico superior debe ser mayor que el inferior.")
    if "TT_DESDE_PICO" not in trajectory or "Fecha" not in trajectory:
        return None, None
    frame = trajectory[["Fecha", "TT_DESDE_PICO"]].copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce")
    frame["TT_DESDE_PICO"] = pd.to_numeric(
        frame["TT_DESDE_PICO"], errors="coerce"
    )
    frame = frame.dropna().sort_values("Fecha")
    lower = frame[frame["TT_DESDE_PICO"] >= float(lower_tt)]
    if lower.empty:
        return None, None
    upper = frame[frame["TT_DESDE_PICO"] >= float(upper_tt)]
    start = pd.Timestamp(lower.iloc[0]["Fecha"])
    end = pd.Timestamp(upper.iloc[0]["Fecha"]) if not upper.empty else None
    return start, end
