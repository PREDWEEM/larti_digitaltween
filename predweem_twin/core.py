"""Núcleo biofísico de PREDWEEM Lartigau vK4.9.15 Adaptada.

Conserva las entradas ANN, los filtros y el techo decreciente desde el 15 de
abril de LOLIUM_LARTIGAU-2026. La normalización parcial y la cobertura diaria
son extensiones del gemelo, separadas de la calibración externa por sitio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .seasonal import partial_season_normalization, reference_progress


@dataclass(frozen=True)
class ModelParameters:
    cobertura_pct: float = 75.0
    w_max: float = 18.816
    umbral_termoinhibicion: float = 24.0
    umbral_choque_hidrico: float = 45.0
    exponente_kr: float = 0.0
    latitud: float = -38.6166
    longitud: float = -61.7000
    latencia_jd: int = 25
    techo_choque: float = 1.0
    calentamiento_suelo: float = 0.0
    umbral_primer_pico: float = 0.20
    t_base: float = 2.0
    t_opt: float = 20.0
    t_crit: float = 30.0
    tt_control: float = 600.0
    tt_limite: float = 800.0
    decay_enabled: bool = True
    decay_tau_days: float = 60.0
    decay_beta: float = 1.0
    decay_intensity: float = 0.75
    decay_cap_fraction: float = 0.50


class PracticalANNModel:
    """Red neuronal original de cuatro entradas, sin dependencias de UI."""

    def __init__(self, iw: np.ndarray, bias_iw: np.ndarray, lw: np.ndarray, bias_out: np.ndarray):
        self.iw = iw
        self.bias_iw = bias_iw
        self.lw = lw
        self.bias_out = bias_out
        self.input_min = np.array([1.0, 0.0, -7.0, 0.0])
        self.input_max = np.array([300.0, 41.0, 25.5, 84.0])

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "PracticalANNModel":
        path = Path(model_dir)
        return cls(
            np.load(path / "IW.npy"),
            np.load(path / "bias_IW.npy"),
            np.load(path / "LW.npy"),
            np.load(path / "bias_out.npy"),
        )

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = 2 * (inputs - self.input_min) / (self.input_max - self.input_min) - 1
        hidden = np.tanh(normalized @ self.iw + self.bias_iw)
        relative = (np.tanh((hidden @ self.lw.T).flatten() + self.bias_out) + 1) / 2
        return relative, np.cumsum(relative)


def calculate_tt(temperature: float, t_base: float, t_opt: float, t_crit: float) -> float:
    if temperature <= t_base:
        return 0.0
    if temperature <= t_opt:
        return float(temperature - t_base)
    if temperature < t_crit:
        return float((temperature - t_base) * ((t_crit - temperature) / (t_crit - t_opt)))
    return 0.0


def calculate_et0_hargreaves(jday, tmax, tmin, latitude=-38.6166):
    lat_rad = np.radians(latitude)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(dec), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec)
        + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)


def surface_parameters(coverage_pct):
    """Convierte cobertura escalar o diaria en parámetros de superficie."""
    coverage = np.clip(np.asarray(coverage_pct, dtype=float), 0.0, 100.0)
    points = [0.0, 30.0, 70.0, 100.0]
    ke_soil = np.interp(coverage, points, [0.85, 0.50, 0.25, 0.10])
    thermal_modulator = np.interp(
        coverage, points, [0.95, 0.90, 0.85, 0.80]
    )
    if coverage.ndim == 0:
        return float(ke_soil), float(thermal_modulator)
    return ke_soil, thermal_modulator


def daily_coverage(
    dates: pd.Series,
    default_coverage_pct: float,
    coverage_series: pd.DataFrame | None = None,
) -> np.ndarray:
    """Interpola cobertura entre mediciones y conserva el último valor.

    Antes de la primera medición se usa el valor constante configurado. Luego
    de la última se mantiene la última cobertura observada.
    """
    model_dates = pd.to_datetime(dates, errors="coerce").dt.tz_localize(None)
    default = float(np.clip(default_coverage_pct, 0.0, 100.0))
    if coverage_series is None or coverage_series.empty:
        return np.full(len(model_dates), default, dtype=float)

    observed = coverage_series.copy()
    if not {"Fecha", "Cobertura_PCT"}.issubset(observed.columns):
        raise ValueError("La cobertura observada requiere Fecha y Cobertura_PCT.")
    observed["Fecha"] = pd.to_datetime(
        observed["Fecha"], errors="coerce"
    ).dt.tz_localize(None)
    observed["Cobertura_PCT"] = pd.to_numeric(
        observed["Cobertura_PCT"], errors="coerce"
    )
    observed = (
        observed.dropna(subset=["Fecha", "Cobertura_PCT"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
    )
    if observed.empty:
        return np.full(len(model_dates), default, dtype=float)
    if not observed["Cobertura_PCT"].between(0.0, 100.0).all():
        raise ValueError("Cobertura_PCT debe estar comprendida entre 0 y 100.")

    model_axis = model_dates.astype("int64").to_numpy(dtype=float)
    observed_axis = observed["Fecha"].astype("int64").to_numpy(dtype=float)
    return np.interp(
        model_axis,
        observed_axis,
        observed["Cobertura_PCT"].to_numpy(float),
        left=default,
        right=float(observed["Cobertura_PCT"].iloc[-1]),
    )


def surface_water_balance(prec, et0, w_max=20.0, ke_soil=0.4, kr_exponent=0.0):
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    daily_ke = np.asarray(ke_soil, dtype=float)
    if daily_ke.ndim == 0:
        daily_ke = np.full(len(prec), float(daily_ke), dtype=float)
    if len(daily_ke) != len(prec):
        raise ValueError("Ke_Suelo debe ser escalar o tener un valor por día.")
    if w_max <= 0:
        raise ValueError("Wmax debe ser mayor que cero.")
    water = np.zeros(len(prec), dtype=float)
    daily_kr = np.ones(len(prec), dtype=float)
    if not len(prec):
        return water, daily_kr
    water[0] = float(w_max) / 2.0
    exponent = max(float(kr_exponent), 0.0)
    for i in range(1, len(prec)):
        water_fraction = float(np.clip(water[i - 1] / float(w_max), 0.0, 1.0))
        kr = 1.0 if exponent == 0.0 else water_fraction**exponent
        daily_kr[i] = kr
        water[i] = np.clip(
            water[i - 1] + prec[i] - et0[i] * daily_ke[i] * kr,
            0.0,
            float(w_max),
        )
    return water, daily_kr


def apply_cohort_decay(
    trajectory: pd.DataFrame,
    first_peak_index: int | None,
    params: ModelParameters,
) -> pd.DataFrame:
    """Conserva el techo original de Lartigau desde el 15/04 de cada año.

    Hasta el 14/04 no cambia el flujo. Después, el techo parte del 50 % del
    máximo previo y decae con tau=60, beta=1 e intensidad=0.75. Si no existe
    señal positiva previa a abril, el motor original no impone un techo.
    """
    df = trajectory.copy()
    df["EMERREL_ANTES_DECAIMIENTO"] = df["EMERREL"].copy()
    df["Dias_Desde_15Abr"] = 0.0
    df["Factor_Decaimiento_15Abr"] = 1.0
    df["Techo_EMERREL_15Abr"] = np.nan
    df["Tau_Decaimiento_15Abr_d"] = params.decay_tau_days
    df["Beta_Decaimiento_15Abr"] = params.decay_beta
    df["Intensidad_Decaimiento_15Abr"] = params.decay_intensity
    df["Fraccion_Maxima_15Abr"] = params.decay_cap_fraction

    if not params.decay_enabled:
        return df

    tau = max(float(params.decay_tau_days), 1e-9)
    beta = max(float(params.decay_beta), 1e-9)
    intensity = float(np.clip(params.decay_intensity, 0.0, 1.0))
    dates = pd.to_datetime(df["Fecha"])
    for year in dates.dt.year.unique():
        start = pd.Timestamp(year=int(year), month=4, day=15)
        same_year = dates.dt.year.eq(year)
        before = same_year & dates.lt(start)
        after = same_year & dates.ge(start)
        days = (dates.loc[after] - start).dt.days.astype(float)
        factor = (1.0 - intensity) + intensity * np.exp(-((days / tau) ** beta))
        df.loc[after, "Dias_Desde_15Abr"] = days
        df.loc[after, "Factor_Decaimiento_15Abr"] = factor
        previous_max = float(df.loc[before, "EMERREL"].clip(lower=0).max()) if before.any() else 0.0
        if previous_max > 0.0:
            cap = float(params.decay_cap_fraction) * previous_max * factor
            df.loc[after, "Techo_EMERREL_15Abr"] = cap
            df.loc[after, "EMERREL"] = np.minimum(
                df.loc[after, "EMERREL"].clip(lower=0), cap,
            )
    return df


def _clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    df = weather.copy()
    df.columns = [str(column).upper().strip() for column in df.columns]
    df = df.rename(
        columns={
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "DATETIME": "Fecha",
            "PREC": "Prec",
            "PRECIPITACION": "Prec",
            "PRECIPITACIÓN": "Prec",
            "LLUVIA": "Prec",
        }
    )
    required = ["Fecha", "TMAX", "TMIN", "Prec"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Faltan columnas meteorológicas: " + ", ".join(missing))
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.tz_localize(None)
    for column in ("TMAX", "TMIN", "Prec"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = (
        df.dropna(subset=required)
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No hay datos meteorológicos válidos.")
    if (df["TMAX"] < df["TMIN"]).any():
        raise ValueError("Se detectó al menos un día con TMAX menor que TMIN.")
    df["Prec"] = df["Prec"].clip(lower=0.0)
    return df


def run_predweem(
    weather: pd.DataFrame,
    model: PracticalANNModel,
    params: ModelParameters,
    coverage_series: pd.DataFrame | None = None,
    normalization_as_of=None,
    seasonal_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ejecuta PREDWEEM y devuelve una trayectoria diaria auditable."""
    df = _clean_weather(weather)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Cobertura_Rastrojo"] = daily_coverage(
        df["Fecha"], params.cobertura_pct, coverage_series
    )
    ke_soil, thermal_modulator = surface_parameters(df["Cobertura_Rastrojo"])
    df["Ke_Suelo"] = ke_soil
    df["Modulador_Termico_Cobertura"] = thermal_modulator
    df["Cobertura_Modo"] = (
        "serie observada interpolada"
        if coverage_series is not None and not coverage_series.empty
        else "constante"
    )
    if coverage_series is not None and not coverage_series.empty:
        observed_coverage_dates = pd.to_datetime(
            coverage_series["Fecha"], errors="coerce"
        ).dt.tz_localize(None)
        df["Cobertura_Observada"] = df["Fecha"].isin(observed_coverage_dates)
    else:
        df["Cobertura_Observada"] = False
    df["Exponente_Kr"] = params.exponente_kr

    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    amplitude = (df["TMAX"] - df["TMIN"]) / 2.0
    df["TMAX_suelo"] = df["Tmedia_aire"] + amplitude * thermal_modulator
    df["TMIN_suelo"] = df["Tmedia_aire"] - amplitude * thermal_modulator
    if params.calentamiento_suelo:
        winter = df["Julian_days"].between(152, 264)
        df.loc[winter, ["TMAX_suelo", "TMIN_suelo"]] += params.calentamiento_suelo
    df["Tmedia"] = df["Tmedia_aire"]

    ann_input = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = model.predict(ann_input)
    df["EMERREL_RAW_ANN"] = np.clip(emerrel_raw, 0.0, 1.0)
    df["EMERREL"] = df["EMERREL_RAW_ANN"].copy()

    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    hydric_shock = (
        (df["Julian_days"] > params.latencia_jd)
        & (df["Julian_days"] <= 110)
        & (df["Prec_3d"] >= params.umbral_choque_hidrico)
    )
    df.loc[hydric_shock, "EMERREL"] = np.maximum(
        df.loc[hydric_shock, "EMERREL"], params.techo_choque
    )
    df["Choque_Hidrico"] = hydric_shock

    df["ET0"] = calculate_et0_hargreaves(
        df["Julian_days"].to_numpy(),
        df["TMAX"].to_numpy(),
        df["TMIN"].to_numpy(),
        params.latitud,
    )
    water, daily_kr = surface_water_balance(
        df["Prec"].to_numpy(),
        df["ET0"].to_numpy(),
        params.w_max,
        ke_soil,
        params.exponente_kr,
    )
    df["W_superficial"] = water
    df["Kr_Diario"] = daily_kr
    df["Humedad_Relativa"] = df["W_superficial"] / max(params.w_max, 1e-12)
    df["Hydric_Factor"] = 1.0 / (1.0 + np.exp(-10.0 * (df["Humedad_Relativa"] - 0.30)))
    df["EMERREL"] *= df["Hydric_Factor"]
    df.loc[df["Humedad_Relativa"] < 0.20, "EMERREL"] = 0.0

    df["Lluvia_Recarga"] = (df["Prec"] >= params.w_max).cummax()
    df.loc[~df["Lluvia_Recarga"], "EMERREL"] = 0.0
    df["Tmedia_5d"] = df["Tmedia_aire"].rolling(window=5, min_periods=1).mean()
    df["Termoinhibida"] = df["Tmedia_5d"] >= params.umbral_termoinhibicion
    df.loc[df["Termoinhibida"], "EMERREL"] = 0.0
    df.loc[df["Julian_days"] <= params.latencia_jd, "EMERREL"] = 0.0
    df["EMERREL"] = np.clip(df["EMERREL"], 0.0, 1.0)

    candidates = df.index[df["EMERREL"] > params.umbral_primer_pico].tolist()
    first_peak_index = candidates[0] if candidates else None
    df["Primer_Pico_Habilitado"] = False
    if first_peak_index is None:
        df["EMERREL"] = 0.0
    else:
        df.loc[first_peak_index:, "Primer_Pico_Habilitado"] = True
        df.loc[: first_peak_index - 1, "EMERREL"] = 0.0

    df = apply_cohort_decay(df, first_peak_index, params)

    df["EMERAC"] = df["EMERREL"].cumsum()
    available_total = float(df["EMERREL"].sum())
    total = available_total
    normalization_mode = "total del período disponible"
    normalization_metadata = {}
    if seasonal_reference is not None and normalization_as_of is not None:
        seasonal_total, normalization_metadata = partial_season_normalization(
            df, normalization_as_of, seasonal_reference
        )
        if seasonal_total is not None:
            total = seasonal_total
            normalization_mode = normalization_metadata["mode"]
        p10, median, p90 = reference_progress(
            seasonal_reference, df["Julian_days"].to_numpy(float)
        )
        df["Progreso_Estacional_P10"] = p10
        df["Progreso_Estacional_Referencia"] = median
        df["Progreso_Estacional_P90"] = p90
    else:
        df["Progreso_Estacional_P10"] = np.nan
        df["Progreso_Estacional_Referencia"] = np.nan
        df["Progreso_Estacional_P90"] = np.nan
    df["EMERAC_NORMALIZADA"] = df["EMERAC"] / total if total > 0 else 0.0
    df["EMERAC_NORMALIZADA"] = df["EMERAC_NORMALIZADA"].clip(0.0, 1.0)
    df["Normalizacion_Modo"] = normalization_mode
    df["Total_EMERREL_Referencia"] = total
    df["Fecha_Ancla_Normalizacion"] = normalization_metadata.get(
        "anchor_date", pd.NaT
    )
    df["DG"] = df["Tmedia"].apply(
        lambda value: calculate_tt(value, params.t_base, params.t_opt, params.t_crit)
    )
    if first_peak_index is None:
        df["TT_DESDE_PICO"] = 0.0
    else:
        df["TT_DESDE_PICO"] = 0.0
        df.loc[first_peak_index:, "TT_DESDE_PICO"] = df.loc[first_peak_index:, "DG"].cumsum()
    return df
