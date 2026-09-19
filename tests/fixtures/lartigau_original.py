"""Funciones científicas extraídas del original afc421c52695a3a721e3327bcde2ef72ebaef9e0.
Incluye el parche operativo de decaimiento desde 15/04, sin interfaz.
Sólo se usa como referencia independiente en pruebas de equivalencia.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

APP_VERSION = "vK4.9.15 Adaptada"

LATITUD_LARTIGAU = -38.6166

LATENCIA_JD = 25

VENTANA_TERMICA_DIAS = 5

UMBRAL_TERMINHIBICION = 24.0

VENTANA_LLUVIA_DIAS = 3

UMBRAL_CHOQUE_HIDRICO_MM = 45.0

FIN_CHOQUE_HIDRICO_JD = 110

TECHO_CHOQUE_HIDRICO = 1.0

UMBRAL_PRIMER_PICO = 0.20

PERSISTENCIA_PRIMER_PICO_DIAS = 1

WMAX_PREDETERMINADO = 18.816

COBERTURA_PREDETERMINADA = 75

EXPONENTE_KR_PREDETERMINADO = 0.0

P50_HIDRICO = 0.30

PENDIENTE_HIDRICA = 10.0

CORTE_HIDRICO = 0.20

T_BASE_PREDETERMINADA = 2.0

T_OPTIMA_PREDETERMINADA = 20.0

T_CRITICA_PREDETERMINADA = 30.0

TT_CONTROL_PREDETERMINADO = 600

TT_LIMITE_PREDETERMINADO = 800

class PracticalANNModel:
    def __init__(
        self,
        input_weights: np.ndarray,
        input_bias: np.ndarray,
        output_weights: np.ndarray,
        output_bias: np.ndarray,
    ) -> None:
        self.IW = np.asarray(input_weights, dtype=float)
        self.bIW = np.asarray(input_bias, dtype=float)
        self.LW = np.asarray(output_weights, dtype=float)
        self.bLW = np.asarray(output_bias, dtype=float).reshape(-1)

        self.input_min = np.array([1.0, 0.0, -7.0, 0.0])
        self.input_max = np.array([300.0, 41.0, 25.5, 84.0])

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        denominator = self.input_max - self.input_min
        return 2.0 * (values - self.input_min) / denominator - 1.0

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.normalize(values)
        hidden = np.tanh(normalized @ self.IW + self.bIW)
        linear_output = (hidden @ self.LW.T).reshape(-1)
        bias = float(self.bLW[0]) if self.bLW.size else 0.0
        emergence = (np.tanh(linear_output + bias) + 1.0) / 2.0
        emergence = np.clip(emergence, 0.0, 1.0)
        return emergence, np.cumsum(emergence)

def calculate_et0_hargreaves(
    julian_day: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    latitude: float = LATITUD_LARTIGAU,
) -> np.ndarray:
    julian_day = np.asarray(julian_day, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)

    latitude_radians = np.radians(latitude)
    inverse_distance = (
        1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * julian_day)
    )
    declination = 0.409 * np.sin(
        2.0 * np.pi / 365.0 * julian_day - 1.39
    )
    sunset_angle = np.arccos(
        np.clip(
            -np.tan(latitude_radians) * np.tan(declination),
            -1.0,
            1.0,
        )
    )
    extraterrestrial_radiation = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * inverse_distance
        * (
            sunset_angle
            * np.sin(latitude_radians)
            * np.sin(declination)
            + np.cos(latitude_radians)
            * np.cos(declination)
            * np.sin(sunset_angle)
        )
    )
    radiation_mm = extraterrestrial_radiation / 2.45
    mean_temperature = (tmax + tmin) / 2.0
    thermal_range = np.maximum(tmax - tmin, 0.0)

    return np.maximum(
        0.0023
        * radiation_mm
        * (mean_temperature + 17.8)
        * np.sqrt(thermal_range),
        0.0,
    )

def surface_parameters(coverage_percent: float) -> tuple[float, float]:
    coverage = float(np.clip(coverage_percent, 0.0, 100.0))
    reference_coverage = [0.0, 30.0, 70.0, 100.0]

    ke_value = float(
        np.interp(
            coverage,
            reference_coverage,
            [0.85, 0.50, 0.25, 0.10],
        )
    )
    thermal_modulator = float(
        np.interp(
            coverage,
            reference_coverage,
            [0.95, 0.90, 0.85, 0.80],
        )
    )
    return ke_value, thermal_modulator

def surface_water_balance(
    precipitation: np.ndarray,
    et0: np.ndarray,
    w_max: float,
    ke_soil: float,
    kr_exponent: float = EXPONENTE_KR_PREDETERMINADO,
    return_kr: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Balance común con Kr configurable.

    kr_exponent=0 conserva exactamente ET0 × Ke constante.
    kr_exponent=1 reproduce la adaptación de Tres Arroyos.
    """
    precipitation = np.asarray(precipitation, dtype=float)
    et0 = np.asarray(et0, dtype=float)

    water = np.zeros(len(precipitation), dtype=float)
    kr_daily = np.ones(len(precipitation), dtype=float)
    if len(water) == 0:
        return (water, kr_daily) if return_kr else water
    if float(w_max) <= 0.0:
        raise ValueError("Wmax debe ser mayor que cero.")

    exponent = max(float(kr_exponent), 0.0)
    water[0] = float(w_max) / 2.0
    for index in range(1, len(water)):
        relative_previous_water = float(
            np.clip(water[index - 1] / float(w_max), 0.0, 1.0)
        )
        kr = (
            1.0
            if exponent == 0.0
            else relative_previous_water ** exponent
        )
        kr_daily[index] = kr
        actual_evaporation = et0[index] * float(ke_soil) * kr
        water[index] = np.clip(
            water[index - 1]
            + precipitation[index]
            - actual_evaporation,
            0.0,
            float(w_max),
        )
    return (water, kr_daily) if return_kr else water

def apply_first_peak_filter(
    dataframe: pd.DataFrame,
    threshold: float = UMBRAL_PRIMER_PICO,
) -> tuple[pd.DataFrame, int | None]:
    """
    Habilita la campaña en el primer día con EMERREL > threshold.

    No utiliza fecha objetivo, lag, interpolación temporal ni información
    futura de campo.
    """
    dataframe = dataframe.copy()
    dataframe["EMERREL_ANTES_FILTRO_PRIMER_PICO"] = dataframe[
        "EMERREL"
    ].copy()

    above_threshold = dataframe["EMERREL"].gt(float(threshold))
    candidates = dataframe.index[above_threshold].tolist()

    if candidates:
        first_peak_index = int(candidates[0])
        dataframe["Primer_Pico_Habilitado"] = (
            dataframe.index >= first_peak_index
        )
        dataframe.loc[
            dataframe.index < first_peak_index,
            "EMERREL",
        ] = 0.0
    else:
        first_peak_index = None
        dataframe["Primer_Pico_Habilitado"] = False
        dataframe["EMERREL"] = 0.0

    dataframe["Supera_Umbral_Primer_Pico"] = above_threshold
    dataframe["Persistencia_Primer_Pico_Dias"] = (
        PERSISTENCIA_PRIMER_PICO_DIAS
    )
    return dataframe, first_peak_index

def canonicalize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("El archivo meteorológico está vacío.")

    data = raw.copy()
    data.columns = [str(column).strip().upper() for column in data.columns]
    data = data.rename(
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
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Faltan columnas meteorológicas: " + ", ".join(missing)
        )

    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce")
    for column in ("TMAX", "TMIN", "Prec"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=required)
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    data["Prec"] = data["Prec"].clip(lower=0.0)

    if len(data) < 30:
        raise ValueError(
            "Se requieren al menos 30 días meteorológicos válidos."
        )
    return data

def simulate_emergence(
    raw_weather: pd.DataFrame,
    ann_model: PracticalANNModel,
    coverage_percent: int,
    w_max: float,
    thermoinhibition_threshold: float = UMBRAL_TERMINHIBICION,
    hydric_shock_threshold: float = UMBRAL_CHOQUE_HIDRICO_MM,
    kr_exponent: float = EXPONENTE_KR_PREDETERMINADO,
) -> tuple[pd.DataFrame, int | None]:
    data = canonicalize_weather(raw_weather)

    data["Julian_days"] = data["Fecha"].dt.dayofyear
    data["Tmedia_aire"] = (data["TMAX"] + data["TMIN"]) / 2.0
    thermal_amplitude = (data["TMAX"] - data["TMIN"]) / 2.0

    ke_value, thermal_modulator = surface_parameters(coverage_percent)
    data["Cobertura_Rastrojo"] = int(coverage_percent)
    data["Ke_Suelo"] = ke_value
    data["Exponente_Kr"] = float(kr_exponent)
    data["Modulador_Termico_Diagnostico"] = thermal_modulator

    # Sólo diagnóstico de microclima; no son entradas de la ANN.
    data["TMAX_suelo_diagnostica"] = (
        data["Tmedia_aire"]
        + thermal_amplitude * thermal_modulator
    )
    data["TMIN_suelo_diagnostica"] = (
        data["Tmedia_aire"]
        - thermal_amplitude * thermal_modulator
    )

    # ANN desacoplada de la cobertura.
    neural_inputs = data[
        ["Julian_days", "TMAX", "TMIN", "Prec"]
    ].to_numpy(float)
    raw_emergence, _ = ann_model.predict(neural_inputs)
    data["EMERREL_RAW_ANN"] = np.clip(raw_emergence, 0.0, 1.0)
    data["EMERREL"] = data["EMERREL_RAW_ANN"].copy()

    # Choque hídrico conservado.
    data["Prec_3d"] = data["Prec"].rolling(
        window=VENTANA_LLUVIA_DIAS,
        min_periods=1,
    ).sum()
    hydric_shock = (
        (data["Julian_days"] > LATENCIA_JD)
        & (data["Julian_days"] <= FIN_CHOQUE_HIDRICO_JD)
        & (data["Prec_3d"] >= float(hydric_shock_threshold))
    )
    data.loc[hydric_shock, "EMERREL"] = np.maximum(
        data.loc[hydric_shock, "EMERREL"],
        TECHO_CHOQUE_HIDRICO,
    )
    data["Choque_Hidrico"] = hydric_shock

    # Balance hídrico controlado por cobertura mediante Ke.
    data["ET0"] = calculate_et0_hargreaves(
        data["Julian_days"].to_numpy(),
        data["TMAX"].to_numpy(),
        data["TMIN"].to_numpy(),
        LATITUD_LARTIGAU,
    )
    water, kr_daily = surface_water_balance(
        data["Prec"].to_numpy(),
        data["ET0"].to_numpy(),
        float(w_max),
        ke_value,
        kr_exponent=float(kr_exponent),
        return_kr=True,
    )
    data["W_superficial"] = water
    data["Kr_Diario"] = kr_daily
    relative_water = data["W_superficial"] / max(float(w_max), 1e-12)
    data["Humedad_Relativa"] = relative_water

    hydric_exponent = np.clip(
        -PENDIENTE_HIDRICA * (relative_water - P50_HIDRICO),
        -60.0,
        60.0,
    )
    data["Hydric_Factor"] = 1.0 / (1.0 + np.exp(hydric_exponent))
    data["EMERREL"] *= data["Hydric_Factor"]

    data.loc[
        relative_water < CORTE_HIDRICO,
        "EMERREL",
    ] = 0.0

    # Conserva el criterio operativo original de recarga por lluvia diaria.
    data["Lluvia_Recarga"] = (
        data["Prec"] >= float(w_max)
    ).cummax()
    data.loc[~data["Lluvia_Recarga"], "EMERREL"] = 0.0

    # Termoinhibición fija con temperatura media del aire.
    data["Tmedia_5d"] = data["Tmedia_aire"].rolling(
        window=VENTANA_TERMICA_DIAS,
        min_periods=1,
    ).mean()
    data["Termoinhibida"] = (
        data["Tmedia_5d"] >= float(thermoinhibition_threshold)
    )
    data.loc[data["Termoinhibida"], "EMERREL"] = 0.0

    # Latencia fija al final del conjunto de filtros.
    data.loc[
        data["Julian_days"] <= LATENCIA_JD,
        "EMERREL",
    ] = 0.0

    data["EMERREL"] = np.clip(data["EMERREL"], 0.0, 1.0)
    data, first_peak_index = apply_first_peak_filter(
        data,
        threshold=UMBRAL_PRIMER_PICO,
    )

    # Decaimiento tardío Lartigau: hasta 14-abr EMERREL queda intacto.
    data["EMERREL_ANTES_DECAIMIENTO_15ABR"] = data["EMERREL"].copy()
    data["Dias_Desde_15Abr"] = 0.0
    data["Factor_Decaimiento_15Abr"] = 1.0
    data["Techo_EMERREL_15Abr"] = np.nan

    tau_d = 60.0
    beta_d = 1.0
    intensidad_d = 0.75
    fraccion_max_d = 0.50

    inicio_decaimiento = pd.to_datetime({
        "year": data["Fecha"].dt.year,
        "month": np.full(len(data), 4),
        "day": np.full(len(data), 15),
    })
    mascara_decay = data["Fecha"] >= inicio_decaimiento
    dias_decay = (data["Fecha"] - inicio_decaimiento).dt.days.clip(lower=0).astype(float)

    max_pre_por_anio = {}
    for anio in sorted(data["Fecha"].dt.year.dropna().unique()):
        fecha_inicio_anio = pd.Timestamp(year=int(anio), month=4, day=15)
        mascara_pre = (data["Fecha"].dt.year == anio) & (data["Fecha"] < fecha_inicio_anio)
        max_pre = (
            float(data.loc[mascara_pre, "EMERREL"].clip(lower=0.0).max())
            if mascara_pre.any()
            else 0.0
        )
        max_pre_por_anio[int(anio)] = max_pre

    factor_decay = np.ones(len(data), dtype=float)
    factor_decay[mascara_decay.to_numpy()] = (
        (1.0 - intensidad_d)
        + intensidad_d * np.exp(
            -((dias_decay[mascara_decay].to_numpy() / tau_d) ** beta_d)
        )
    )

    techo_decay = np.full(len(data), np.nan, dtype=float)
    for pos, (_, fila) in enumerate(data.iterrows()):
        if bool(mascara_decay.iloc[pos]):
            max_pre = max_pre_por_anio.get(int(fila["Fecha"].year), 0.0)
            if max_pre > 0.0:
                techo_decay[pos] = fraccion_max_d * max_pre * factor_decay[pos]

    data["Dias_Desde_15Abr"] = dias_decay
    data["Factor_Decaimiento_15Abr"] = np.clip(factor_decay, 0.0, 1.0)
    data["Techo_EMERREL_15Abr"] = techo_decay

    mascara_con_techo = mascara_decay & data["Techo_EMERREL_15Abr"].notna()
    if mascara_con_techo.any():
        valores_originales = data.loc[mascara_con_techo, "EMERREL"].clip(lower=0.0).to_numpy()
        techos_activos = data.loc[mascara_con_techo, "Techo_EMERREL_15Abr"].to_numpy()
        data.loc[mascara_con_techo, "EMERREL"] = np.minimum(
            valores_originales,
            techos_activos,
        )

    data["Tau_Decaimiento_15Abr_d"] = tau_d
    data["Beta_Decaimiento_15Abr"] = beta_d
    data["Intensidad_Decaimiento_15Abr"] = intensidad_d
    data["Fraccion_Maxima_15Abr"] = fraccion_max_d

    data["EMERAC"] = data["EMERREL"].cumsum()
    total_emergence = float(data["EMERREL"].sum())
    data["EMERAC_NORMALIZADA"] = (
        data["EMERAC"] / total_emergence
        if total_emergence > 0.0
        else 0.0
    )
    return data, first_peak_index
