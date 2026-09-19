"""Construcción del estado operativo diario del gemelo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TwinSnapshot:
    site_id: str
    as_of: str
    emergence: float
    remaining: float
    risk_7d: str
    increment_7d: float
    soil_water: float
    soil_water_fraction: float
    thermal_time: float
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


def _risk(increment: float) -> str:
    if increment <= 0.01:
        return "Nulo"
    if increment <= 0.05:
        return "Bajo"
    if increment <= 0.15:
        return "Medio"
    return "Alto"


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
) -> dict:
    as_of = pd.Timestamp(as_of).tz_localize(None).normalize()
    df = trajectory.sort_values("Fecha").reset_index(drop=True)
    candidates = df.index[df["Fecha"] <= as_of].tolist()
    idx = candidates[-1] if candidates else 0
    future_idx = min(idx + 7, len(df) - 1)
    current = float(df.at[idx, "EMERAC_TWIN"])
    future = float(df.at[future_idx, "EMERAC_TWIN"])
    increment = max(0.0, future - current)
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
        risk_7d=_risk(increment),
        increment_7d=increment,
        soil_water=float(df.at[idx, "W_superficial"]),
        soil_water_fraction=float(df.at[idx, "Humedad_Relativa"]),
        thermal_time=float(df.at[idx, "TT_DESDE_PICO"]),
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
