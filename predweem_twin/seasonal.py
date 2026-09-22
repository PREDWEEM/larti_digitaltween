"""Referencia estacional para normalizar ejecuciones meteorológicas parciales."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import pickle

import numpy as np
import pandas as pd


EXCLUDED_YEARS = ("2010", "2015")
EXCLUDED_SITES = ("balcarce", "san pedro")
HISTORICAL_YEARS = ("2008", "2009", "2011", "2012", "2013", "2014", "2023", "2024")
EXCLUDED_CAMPAIGNS = ("test -emerel tresas 2025.xlsx",)


def load_seasonal_reference(
    source: str | Path,
    excluded_years: tuple[str, ...] = EXCLUDED_YEARS,
    excluded_sites: tuple[str, ...] = EXCLUDED_SITES,
) -> pd.DataFrame:
    """Selecciona las ocho curvas identificadas únicamente por año.

    La lista explícita impide incorporar otra localidad o un año nuevo sin
    revisar su procedencia. Los nombres originales no acreditan que las ocho
    curvas sean de Lartigau. Cada curva se normaliza por su propio total.
    """
    with Path(source).open("rb") as handle:
        payload = pickle.load(handle)

    julian_days = np.asarray(
        payload.get("JD_common", payload.get("JD_COMMON")), dtype=float
    )
    curves = np.asarray(
        payload.get("curves_interp", payload.get("curves")), dtype=float
    )
    names = [str(value) for value in payload.get("names", payload.get("files", []))]
    if curves.ndim != 2 or len(julian_days) != curves.shape[1]:
        raise ValueError("La referencia histórica no contiene curvas compatibles.")
    if len(names) != len(curves) or any(not name.strip() for name in names):
        raise ValueError("La referencia requiere un nombre por curva para filtrar años y localidades.")
    if (not np.isfinite(julian_days).all()
            or len(julian_days) < 2 or not (np.diff(julian_days) > 0).all()):
        raise ValueError("El eje histórico debe ser finito y estrictamente creciente.")
    sites = tuple(" ".join(site.casefold().split()) for site in excluded_sites)
    allowed = {f"{year}.xlsx" for year in HISTORICAL_YEARS}
    keep = np.array([
        name.strip().casefold() in allowed
        and not any(year in name for year in excluded_years)
        and not any(site in " ".join(name.casefold().split()) for site in sites)
        and name.strip().casefold() not in EXCLUDED_CAMPAIGNS
        for name in names
    ], dtype=bool)
    excluded_names = [name for name, selected in zip(names, keep) if not selected]
    names = [name for name, selected in zip(names, keep) if selected]
    curves = curves[keep]
    if len(names) != len(allowed) or {name.strip().casefold() for name in names} != allowed:
        raise ValueError("Se requieren exactamente las ocho series históricas identificadas por año.")
    if not np.isfinite(curves).all():
        raise ValueError("Las curvas históricas seleccionadas contienen valores no finitos.")
    curves = np.clip(curves, 0.0, None)
    totals = curves.sum(axis=1, keepdims=True)
    if not (totals[:, 0] > 1e-12).all():
        raise ValueError("Cada campaña seleccionada debe contener flujos positivos.")
    progress = np.cumsum(curves, axis=1) / totals
    reference = pd.DataFrame(
        {
            "Julian_days": julian_days,
            "Progreso_P10": np.quantile(progress, 0.10, axis=0),
            "Progreso_Mediano": np.median(progress, axis=0),
            "Progreso_P90": np.quantile(progress, 0.90, axis=0),
            "N_Campanas": len(names),
            "N_Campanas_Dia": len(names),
            "Campanas_Anos": ", ".join(HISTORICAL_YEARS),
            "Campanas": ", ".join(names),
            "Campanas_Excluidas": ", ".join(excluded_names),
        }
    )
    for name, curve in zip(names, progress):
        reference[f"Progreso_{Path(name.strip()).stem}"] = curve
    return reference


def load_local_seasonal_reference(root: str | Path, as_of=None) -> pd.DataFrame:
    """Pool de ocho curvas históricas y Lartigau 2026, según disponibilidad.

    El total 2026 se habilita desde el último conteo; nunca se usa para un
    corte anterior. Se interpola su acumulado, no se inventan conteos diarios.
    Los percentiles reúnen las nueve campañas con igual peso individual.
    """
    root = Path(root)
    reference = load_seasonal_reference(root / "models/modelo_clusters_k3.pkl")
    counts_path = root / "data/calibration/lartigau_2026_counts.csv"
    counts = pd.read_csv(counts_path)
    if not {"FECHA", "PLM2"}.issubset(counts.columns) or len(counts) < 2:
        raise ValueError("La referencia 2026 requiere FECHA, PLM2 y al menos dos visitas.")
    dates = pd.to_datetime(counts["FECHA"], errors="raise").dt.normalize()
    flows = pd.to_numeric(counts["PLM2"], errors="raise").to_numpy(float)
    if (dates.isna().any() or dates.duplicated().any()
            or not dates.is_monotonic_increasing or not dates.dt.year.eq(2026).all()
            or not np.isfinite(flows).all() or (flows < 0).any()
            or flows.sum() <= 0 or flows[0] != 0):
        raise ValueError("Conteos 2026 inválidos o sin cero inicial delimitador.")
    available_from = dates.iloc[-1]
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else None
    if cutoff is not None and pd.isna(cutoff):
        raise ValueError("Fecha de corte de la referencia inválida.")
    use_2026 = cutoff is None or cutoff >= available_from
    reference["Referencia_2026_Desde"] = available_from.date().isoformat()
    reference.attrs["source_2026"] = {
        "path": "data/calibration/lartigau_2026_counts.csv",
        "sha256": sha256(counts_path.read_bytes()).hexdigest(),
        "start": dates.iloc[0].date().isoformat(),
        "end": available_from.date().isoformat(),
        "sample_count": len(counts),
        "window_total_plm2": float(flows.sum()),
        "used": use_2026,
        "processing": "acumulado / total registrado; interpolación lineal entre visitas",
        "scope": "ventana registrada; no certifica el cierre biológico de la campaña",
    }
    if not use_2026:
        reference["Campanas_Excluidas"] += (
            f", {counts_path.name} (disponible desde {available_from:%d/%m/%Y})"
        )
        return reference

    reference["Progreso_2026"] = np.interp(
        reference["Julian_days"], dates.dt.dayofyear, np.cumsum(flows) / flows.sum(),
        left=np.nan, right=1.0,
    )
    campaigns = reference[[f"Progreso_{year}" for year in (*HISTORICAL_YEARS, "2026")]]
    reference["N_Campanas_Dia"] = campaigns.notna().sum(axis=1)
    for q, column in [(0.10, "Progreso_P10"), (0.50, "Progreso_Mediano"), (0.90, "Progreso_P90")]:
        empirical = campaigns.quantile(q, axis=1)
        reference[column + "_Empirico"] = empirical
        # La incorporación de la ventana 2026 cambia la composición del pool.
        # Conservar el avance evita flujos negativos; los cuantiles sin esta
        # regularización y las nueve curvas quedan disponibles para auditoría.
        reference[column] = empirical.cummax()
    reference["N_Campanas"] = 9
    reference["Campanas_Anos"] += ", 2026"
    reference["Campanas"] += f", {counts_path.name}"
    return reference


def reference_progress(
    reference: pd.DataFrame, julian_days
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola P10, mediana y P90 para uno o varios días julianos."""
    days = np.asarray(julian_days, dtype=float)
    axis = reference["Julian_days"].to_numpy(float)
    values = []
    for column in ("Progreso_P10", "Progreso_Mediano", "Progreso_P90"):
        values.append(
            np.interp(
                days,
                axis,
                reference[column].to_numpy(float),
                left=0.0,
                right=1.0,
            )
        )
    return tuple(values)


def partial_season_normalization(
    trajectory: pd.DataFrame,
    as_of,
    reference: pd.DataFrame,
) -> tuple[float | None, dict]:
    """Estima el total de señal estacional sin usar el fin del pronóstico.

    La señal acumulada de PREDWEEM se ancla, en la fecha del estado, al progreso
    mediano de campañas históricas. Si aún no existe señal positiva, utiliza el
    último día disponible como ancla provisional.
    """
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    candidates = trajectory.index[trajectory["Fecha"] <= cutoff].tolist()
    anchor_idx = candidates[-1] if candidates else 0
    p10, median, p90 = reference_progress(
        reference, trajectory["Julian_days"].to_numpy(float)
    )
    raw_cumulative = trajectory["EMERAC"].to_numpy(float)

    if raw_cumulative[anchor_idx] <= 1e-12 or median[anchor_idx] <= 0.01:
        valid = np.flatnonzero((raw_cumulative > 1e-12) & (median > 0.01))
        if not len(valid):
            return None, {
                "mode": "sin señal suficiente",
                "anchor_date": trajectory.at[anchor_idx, "Fecha"],
                "reference_progress": float(median[anchor_idx]),
            }
        anchor_idx = int(valid[0])

    seasonal_total = float(raw_cumulative[anchor_idx] / median[anchor_idx])
    if not np.isfinite(seasonal_total) or seasonal_total <= 1e-12:
        return None, {"mode": "sin señal suficiente"}
    return seasonal_total, {
        "mode": "referencia estacional histórica",
        "anchor_date": trajectory.at[anchor_idx, "Fecha"],
        "reference_progress": float(median[anchor_idx]),
        "reference_p10": float(p10[anchor_idx]),
        "reference_p90": float(p90[anchor_idx]),
        "seasonal_signal_total": seasonal_total,
    }
