"""Lectura y normalización de observaciones de emergencia de campo."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from .assimilation import estimate_flow_potential


DATE_ALIASES = {"fecha", "date", "datetime", "fecha_muestreo"}
FLOW_ALIASES = {
    "plm2",
    "pl_m2",
    "plantas_m2",
    "plantas_por_m2",
    "flujo",
    "flujo_observado",
    "conteo",
    "emergencia_intervalo",
    "media_sr_m2",
    "media_m2",
    "media_plm2",
    "promedio_m2",
    "promedio_plm2",
    "mean_plm2",
}
CUMULATIVE_ALIASES = {
    "observado",
    "emerac",
    "emerac_normalizada",
    "campo_normalizado",
    "emergencia_acumulada",
    "emergencia_acumulada_pct",
    "porcentaje_acumulado",
    "acumulada",
}


def _normalized_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _replicate_number(column) -> int | None:
    """Reconoce 1/2/3, R1, REP_1, REPETICION_1 y variantes."""
    name = _normalized_name(column)
    match = re.fullmatch(
        r"(?:(?:r|rep|replica|repeticion|repeticion_campo)_?)?(\d+)", name
    )
    return int(match.group(1)) if match else None


def replicate_columns(frame: pd.DataFrame) -> list:
    """Devuelve columnas de repeticiones ordenadas por su número."""
    detected = []
    for column in frame.columns:
        number = _replicate_number(column)
        if number is not None:
            detected.append((number, column))
    return [column for _, column in sorted(detected)]


def _file_bytes(source) -> tuple[bytes, str]:
    if hasattr(source, "getvalue"):
        return source.getvalue(), getattr(source, "name", "observaciones.xlsx")
    path = Path(source)
    return path.read_bytes(), path.name


def read_observation_file(source) -> tuple[pd.DataFrame, dict]:
    """Lee CSV/XLS/XLSX y elige la primera hoja con columnas reconocibles."""
    payload, filename = _file_bytes(source)
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(BytesIO(payload), sheet_name=None)
    elif suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else None
        try:
            frame = pd.read_csv(BytesIO(payload), sep=separator, engine="python")
        except UnicodeDecodeError:
            frame = pd.read_csv(
                StringIO(payload.decode("latin-1")), sep=separator, engine="python"
            )
        sheets = {"CSV": frame}
    else:
        raise ValueError("Formato no admitido. Use CSV, TSV, XLS o XLSX.")

    available = []
    for sheet_name, frame in sheets.items():
        names = {_normalized_name(column) for column in frame.columns}
        available.append(sheet_name)
        has_values = bool(names.intersection(FLOW_ALIASES | CUMULATIVE_ALIASES))
        has_replicates = len(replicate_columns(frame)) >= 2
        if names.intersection(DATE_ALIASES) and (has_values or has_replicates):
            return frame, {"archivo": filename, "hoja": sheet_name}
    raise ValueError(
        "No se encontró una hoja con fecha y emergencia observada. "
        f"Hojas revisadas: {', '.join(available)}."
    )


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {_normalized_name(column): column for column in frame.columns}


def _first_column(mapping: dict[str, str], aliases: set[str]):
    return next((mapping[name] for name in aliases if name in mapping), None)


def _model_cumulative_at(trajectory: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    model = trajectory[["Fecha", "EMERAC_NORMALIZADA"]].copy()
    model["Fecha"] = pd.to_datetime(model["Fecha"], errors="coerce").dt.tz_localize(None)
    query = pd.DataFrame({"Fecha": pd.to_datetime(dates).dt.tz_localize(None)})
    matched = pd.merge_asof(
        query.sort_values("Fecha"),
        model.sort_values("Fecha"),
        on="Fecha",
        direction="backward",
    )
    if matched["EMERAC_NORMALIZADA"].isna().any():
        raise ValueError("Hay observaciones anteriores al inicio de la serie meteorológica.")
    return matched["EMERAC_NORMALIZADA"].to_numpy(float)


def prepare_observations(
    raw: pd.DataFrame,
    trajectory: pd.DataFrame,
    mode: str = "auto",
    uncertainty: float = 0.08,
    replicate_uncertainty_floor: float = 0.05,
    seasonal_potential_prior: float | None = None,
    source_name: str = "Archivo cargado",
) -> tuple[pd.DataFrame, dict]:
    """Convierte flujos PLM2 o acumulados porcentuales al estado 0–1.

    Cuando hay repeticiones, verifica la media aportada, infiere el factor de
    conversión a m² y estima la incertidumbre mediante el error estándar del
    flujo de cada intervalo. Para archivos sin repeticiones usa la
    incertidumbre indicada.
    """
    if not 0 < float(uncertainty) <= 1:
        raise ValueError("La incertidumbre debe expresarse entre 0 y 1.")
    if not 0 < float(replicate_uncertainty_floor) <= 1:
        raise ValueError("El mínimo de incertidumbre debe expresarse entre 0 y 1.")

    mapping = _column_map(raw)
    date_column = _first_column(mapping, DATE_ALIASES)
    if date_column is None:
        raise ValueError("Falta una columna de fecha (por ejemplo, FECHA).")

    flow_column = _first_column(mapping, FLOW_ALIASES)
    cumulative_column = _first_column(mapping, CUMULATIVE_ALIASES)
    repetition_columns = replicate_columns(raw)
    normalized_mode = _normalized_name(mode)
    if normalized_mode in {"auto", "detectar_automaticamente"}:
        selected_mode = (
            "flujo" if flow_column is not None or len(repetition_columns) >= 2
            else "acumulado"
        )
    elif normalized_mode in {"flujo", "flujo_por_intervalo_plm2", "plm2"}:
        selected_mode = "flujo"
    elif normalized_mode in {"acumulado", "acumulada", "acumulada_pct"}:
        selected_mode = "acumulado"
    else:
        raise ValueError(f"Modo de observación desconocido: {mode}")

    value_column = flow_column if selected_mode == "flujo" else cumulative_column
    if selected_mode == "flujo" and value_column is None and len(repetition_columns) >= 2:
        value_column = "media calculada de repeticiones"
    if value_column is None:
        raise ValueError(
            "No se encontró la columna requerida. Para flujos use PLM2, una "
            "columna media por m² o al menos dos repeticiones; para acumulados "
            "use EMERGENCIA_ACUMULADA u OBSERVADO."
        )

    selected_columns = [date_column]
    if selected_mode == "flujo" and flow_column is not None:
        selected_columns.append(flow_column)
    elif selected_mode == "acumulado" and cumulative_column is not None:
        selected_columns.append(cumulative_column)
    if selected_mode == "flujo":
        selected_columns.extend(
            column for column in repetition_columns if column not in selected_columns
        )

    prepared = raw[selected_columns].copy().rename(columns={date_column: "Fecha"})
    prepared["Fecha"] = pd.to_datetime(
        prepared["Fecha"], errors="coerce"
    ).dt.tz_localize(None)

    repetition_output_columns = []
    if selected_mode == "flujo" and len(repetition_columns) >= 2:
        for index, column in enumerate(repetition_columns, start=1):
            output_column = f"Repeticion_{index}_original"
            prepared[output_column] = pd.to_numeric(prepared.pop(column), errors="coerce")
            repetition_output_columns.append(output_column)

    if selected_mode == "flujo" and flow_column is not None:
        prepared["Valor_original"] = pd.to_numeric(
            prepared.pop(flow_column), errors="coerce"
        )
    elif selected_mode == "flujo":
        prepared["Valor_original"] = prepared[repetition_output_columns].mean(axis=1)
    else:
        prepared["Valor_original"] = pd.to_numeric(
            prepared.pop(cumulative_column), errors="coerce"
        )

    required_values = ["Fecha", "Valor_original", *repetition_output_columns]
    invalid = prepared[required_values].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"El archivo contiene {int(invalid.sum())} filas con fecha o valor inválido."
        )
    numeric_columns = ["Valor_original", *repetition_output_columns]
    if (prepared[numeric_columns] < 0).any().any():
        raise ValueError("La emergencia observada no puede contener valores negativos.")

    repetition_scale = None
    inferred_quadrat_area = None
    if selected_mode == "flujo" and repetition_output_columns:
        repetition_mean = prepared[repetition_output_columns].mean(axis=1)
        if flow_column is not None:
            positive = repetition_mean > 0
            scale_candidates = (
                prepared.loc[positive, "Valor_original"] / repetition_mean[positive]
            ).replace([np.inf, -np.inf], np.nan).dropna()
            repetition_scale = (
                float(scale_candidates.median()) if not scale_candidates.empty else 1.0
            )
            expected_mean = repetition_mean * repetition_scale
            tolerance = np.maximum(0.5, prepared["Valor_original"].abs() * 0.02)
            if ((prepared["Valor_original"] - expected_mean).abs() > tolerance).any():
                raise ValueError(
                    "La columna media no coincide de forma consistente con las repeticiones."
                )
        else:
            repetition_scale = 1.0
        if repetition_scale > 0:
            inferred_quadrat_area = 1.0 / repetition_scale

    if selected_mode == "flujo":
        aggregation_columns = ["Valor_original", *repetition_output_columns]
        prepared = prepared.groupby("Fecha", as_index=False)[aggregation_columns].sum()
        prepared = prepared.sort_values("Fecha").reset_index(drop=True)
        if repetition_output_columns:
            converted = prepared[repetition_output_columns] * repetition_scale
            prepared["DE_repeticiones_PLM2"] = converted.std(axis=1, ddof=1)
            prepared["EE_repeticiones_PLM2"] = (
                prepared["DE_repeticiones_PLM2"]
                / np.sqrt(len(repetition_output_columns))
            )
            prepared["Factor_conversion_repeticiones"] = repetition_scale
    else:
        prepared = (
            prepared.drop_duplicates("Fecha", keep="last")
            .sort_values("Fecha")
            .reset_index(drop=True)
        )

    model_min = pd.Timestamp(trajectory["Fecha"].min())
    model_max = pd.Timestamp(trajectory["Fecha"].max())
    if prepared["Fecha"].min() < model_min or prepared["Fecha"].max() > model_max:
        raise ValueError(
            "Las observaciones deben estar dentro del período meteorológico "
            f"{model_min.date().isoformat()} a {model_max.date().isoformat()}."
        )

    metadata = {
        "modo": selected_mode,
        "columna_fecha": str(date_column),
        "columna_valor": str(value_column),
        "filas": len(prepared),
    }
    if repetition_output_columns:
        metadata.update(
            {
                "n_repeticiones": len(repetition_output_columns),
                "columnas_repeticiones": [str(column) for column in repetition_columns],
                "factor_conversion_repeticiones": repetition_scale,
                "area_cuadrante_inferida_m2": inferred_quadrat_area,
            }
        )

    if selected_mode == "flujo":
        model_cumulative = _model_cumulative_at(trajectory, prepared["Fecha"])
        model_intervals = np.diff(np.r_[0.0, model_cumulative]).clip(min=1e-6)
        observed_flows = prepared["Valor_original"].to_numpy(float)
        observed_total = float(observed_flows.sum())
        if observed_total <= 0:
            raise ValueError("La suma de PLM2 debe ser mayor que cero.")

        potential_info = estimate_flow_potential(
            model_intervals,
            observed_flows,
            float(model_cumulative[-1]),
            seasonal_potential_prior=seasonal_potential_prior,
        )
        seasonal_total = potential_info["potential"]
        method = "potencial dinámico estimado desde los flujos por intervalo"

        prepared["Flujo_observado_PLM2"] = prepared["Valor_original"]
        prepared["Acumulado_PLM2"] = prepared["Valor_original"].cumsum()
        prepared["Observado"] = (
            prepared["Acumulado_PLM2"] / seasonal_total
        ).clip(0, 1)
        prepared["Unidad_original"] = "plantas/m² por intervalo"
        metadata.update(
            {
                "total_observado_plm2": observed_total,
                "potencial_estacional_plm2": seasonal_total,
                "metodo_normalizacion": method,
                "progreso_modelo_ultima_fecha": float(model_cumulative[-1]),
                "calidad_ajuste_flujos": potential_info["fit_quality"],
                "cv_potencial_estimado": potential_info["potential_cv"],
            }
        )
        if repetition_output_columns:
            estimated_uncertainty = (
                prepared["EE_repeticiones_PLM2"].to_numpy(float) / seasonal_total
            )
            prepared["Incertidumbre"] = np.clip(
                np.maximum(
                    estimated_uncertainty, float(replicate_uncertainty_floor)
                ),
                replicate_uncertainty_floor,
                0.30,
            )
            metadata.update(
                {
                    "metodo_incertidumbre": (
                        "error estándar del flujo entre repeticiones, "
                        f"con mínimo de {replicate_uncertainty_floor:.0%}"
                    ),
                    "incertidumbre_minima": float(prepared["Incertidumbre"].min()),
                    "incertidumbre_maxima": float(prepared["Incertidumbre"].max()),
                }
            )
        else:
            prepared["Incertidumbre"] = float(uncertainty)
            metadata["metodo_incertidumbre"] = "valor fijo indicado por el usuario"
    else:
        values = prepared["Valor_original"].to_numpy(float)
        if values.max(initial=0.0) > 1.0:
            if values.max() > 100.0:
                raise ValueError("Los acumulados deben expresarse entre 0–1 o 0–100 %.")
            values = values / 100.0
            original_unit = "% acumulado"
        else:
            original_unit = "fracción acumulada"
        if np.any(np.diff(values) < -1e-9):
            raise ValueError("La emergencia acumulada debe ser monótona creciente.")
        prepared["Observado"] = np.clip(values, 0.0, 1.0)
        prepared["Unidad_original"] = original_unit
        prepared["Incertidumbre"] = float(uncertainty)
        metadata["metodo_normalizacion"] = "acumulado aportado por el usuario"
        metadata["metodo_incertidumbre"] = "valor fijo indicado por el usuario"

    prepared["Fuente"] = source_name
    prepared["Nota"] = "Carga masiva; " + str(metadata["metodo_normalizacion"])
    return prepared, metadata
