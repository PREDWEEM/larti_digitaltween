"""Lectura y validación de series observadas de cobertura de rastrojo."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import re
import unicodedata

import pandas as pd


DATE_ALIASES = {"fecha", "date", "datetime", "fecha_medicion"}
COVERAGE_ALIASES = {
    "cobertura_pct",
    "cobertura",
    "cobertura_porcentaje",
    "cobertura_rastrojo",
    "cobertura_rastrojo_pct",
    "coverage_pct",
}


def _normalized_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _file_bytes(source) -> tuple[bytes, str]:
    if hasattr(source, "getvalue"):
        return source.getvalue(), getattr(source, "name", "cobertura.xlsx")
    path = Path(source)
    return path.read_bytes(), path.name


def has_coverage_columns(frame: pd.DataFrame) -> bool:
    """Indica si una tabla contiene fecha y cobertura reconocibles."""
    names = {_normalized_name(column) for column in frame.columns}
    return bool(
        names.intersection(DATE_ALIASES)
        and names.intersection(COVERAGE_ALIASES)
    )


def has_coverage_data(frame: pd.DataFrame) -> bool:
    """Indica si, además de la columna, existe al menos una medición."""
    mapping = {_normalized_name(column): column for column in frame.columns}
    coverage_column = next(
        (mapping[name] for name in COVERAGE_ALIASES if name in mapping), None
    )
    if coverage_column is None:
        return False
    values = frame[coverage_column]
    return bool((values.notna() & values.astype(str).str.strip().ne("")).any())


def read_coverage_file(source) -> tuple[pd.DataFrame, dict]:
    """Lee CSV/XLS/XLSX y busca una hoja con fecha y cobertura porcentual."""
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

    reviewed = []
    for sheet_name, frame in sheets.items():
        reviewed.append(sheet_name)
        if has_coverage_columns(frame):
            return frame, {"archivo": filename, "hoja": sheet_name}
    raise ValueError(
        "No se encontró una hoja con FECHA y COBERTURA_PCT. "
        f"Hojas revisadas: {', '.join(reviewed)}."
    )


def prepare_coverage_series(
    raw: pd.DataFrame,
    minimum_date=None,
    maximum_date=None,
    source_name: str = "Archivo cargado",
) -> tuple[pd.DataFrame, dict]:
    """Normaliza una serie a Fecha, Cobertura_PCT y Fuente."""
    mapping = {_normalized_name(column): column for column in raw.columns}
    date_column = next(
        (mapping[name] for name in DATE_ALIASES if name in mapping), None
    )
    coverage_column = next(
        (mapping[name] for name in COVERAGE_ALIASES if name in mapping), None
    )
    if date_column is None or coverage_column is None:
        raise ValueError("La serie requiere las columnas FECHA y COBERTURA_PCT.")

    prepared = raw[[date_column, coverage_column]].copy().rename(
        columns={date_column: "Fecha", coverage_column: "Cobertura_PCT"}
    )
    supplied = (
        prepared["Cobertura_PCT"].notna()
        & prepared["Cobertura_PCT"].astype(str).str.strip().ne("")
    )
    prepared = prepared.loc[supplied].copy()
    if prepared.empty:
        raise ValueError("La columna de cobertura no contiene mediciones.")
    prepared["Fecha"] = pd.to_datetime(
        prepared["Fecha"], errors="coerce"
    ).dt.tz_localize(None)
    prepared["Cobertura_PCT"] = pd.to_numeric(
        prepared["Cobertura_PCT"], errors="coerce"
    )
    invalid = prepared[["Fecha", "Cobertura_PCT"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"La serie contiene {int(invalid.sum())} filas con fecha o cobertura inválida."
        )
    if not prepared["Cobertura_PCT"].between(0.0, 100.0).all():
        raise ValueError("COBERTURA_PCT debe estar comprendida entre 0 y 100.")

    prepared = (
        prepared.sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    if minimum_date is not None and prepared["Fecha"].min() < pd.Timestamp(minimum_date):
        raise ValueError("Hay mediciones anteriores al inicio de la serie meteorológica.")
    if maximum_date is not None and prepared["Fecha"].max() > pd.Timestamp(maximum_date):
        raise ValueError("Hay mediciones posteriores al final de la serie meteorológica.")

    prepared["Fuente"] = source_name
    metadata = {
        "filas": len(prepared),
        "cobertura_minima": float(prepared["Cobertura_PCT"].min()),
        "cobertura_maxima": float(prepared["Cobertura_PCT"].max()),
        "fecha_inicial": prepared["Fecha"].min(),
        "fecha_final": prepared["Fecha"].max(),
    }
    return prepared, metadata
