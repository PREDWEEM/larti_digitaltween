"""Referencia estacional para normalizar ejecuciones meteorológicas parciales."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def load_seasonal_reference(
    source: str | Path,
    excluded_years: tuple[str, ...] = ("2010", "2015"),
    include_patterns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Construye percentiles de progreso y permite una referencia local.

    ``include_patterns`` filtra por nombre de campaña sin distinguir
    mayúsculas. El clasificador original no incluye una campaña identificada
    como Lartigau. Por defecto se utiliza la referencia compartida, excluyendo
    los años 2010 y 2015; no se presenta como una referencia local validada.
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
    if names and len(names) == len(curves):
        patterns = tuple(
            str(pattern).lower() for pattern in (include_patterns or ())
        )
        keep = np.array([
            not any(year in name for year in excluded_years)
            and (
                not patterns
                or any(pattern in name.lower() for pattern in patterns)
            )
            for name in names
        ])
        curves = curves[keep]
        names = [name for name, selected in zip(names, keep) if selected]
    if include_patterns and not len(curves):
        raise ValueError(
            "La referencia histórica no contiene campañas para: "
            + ", ".join(include_patterns)
        )
    curves = np.clip(curves, 0.0, None)
    totals = curves.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 1e-12
    if not valid.any():
        raise ValueError("La referencia histórica no contiene flujos positivos.")
    progress = np.cumsum(curves[valid], axis=1) / totals[valid]
    return pd.DataFrame(
        {
            "Julian_days": julian_days,
            "Progreso_P10": np.quantile(progress, 0.10, axis=0),
            "Progreso_Mediano": np.median(progress, axis=0),
            "Progreso_P90": np.quantile(progress, 0.90, axis=0),
            "N_Campanas": int(valid.sum()),
            "Campanas": ", ".join(
                name for name, selected in zip(names, valid) if selected
            ) if names else "",
        }
    )


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
