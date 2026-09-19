"""Calibración externa por localidad con ANN y filtros biofísicos fijos.

G(F) = logistic(offset + slope * logit(F)). La transformación es monótona,
conserva 0 y 1 y no crea flujo en días donde la curva base no avanza.
Se ajustan dos parámetros sobre flujos por intervalos, condicionados a la
ventana realmente muestreada; no se supone que el último conteo sea el 100 %.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "monotone_logit_v1"
OFFSET_BOUNDS = (-1.5, 1.5)
SLOPE_BOUNDS = (0.6, 1.6)
REGULARIZATION = 0.05


def model_fingerprint(root: str | Path) -> str:
    """Vincula el perfil con los pesos, filtros y referencia usados al ajustar."""
    root = Path(root)
    paths = [
        "models/IW.npy", "models/LW.npy", "models/bias_IW.npy",
        "models/bias_out.npy", "models/modelo_clusters_k3.pkl",
        "predweem_twin/core.py", "predweem_twin/seasonal.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((root / path).read_bytes())
    return digest.hexdigest()


def calibrated_progress(progress, offset=0.0, slope=1.0):
    """Transforma progreso sin extrapolar fuera de [0, 1]."""
    if not np.isfinite([offset, slope]).all() or slope <= 0:
        raise ValueError("Los parámetros deben ser finitos y la pendiente positiva.")
    values = np.asarray(progress, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("El progreso debe ser finito y estar entre 0 y 1.")
    if offset == 0 and slope == 1:
        return values.copy()
    inner = np.clip(values, 1e-12, 1.0 - 1e-12)
    logits = offset + slope * (np.log(inner) - np.log1p(-inner))
    mapped = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    return np.where(values <= 0, 0.0, np.where(values >= 1, 1.0, mapped))


def validate_profile(profile: dict) -> dict:
    if profile.get("schema_version") != 1 or profile.get("method") != METHOD:
        raise ValueError("Versión de calibración no compatible.")
    for key in ("profile_id", "site", "training_year", "training_through", "available_from", "parameters"):
        if not profile.get(key):
            raise ValueError(f"Falta {key} en el perfil de calibración.")
    end = pd.Timestamp(profile["training_through"])
    available = pd.Timestamp(profile["available_from"])
    if pd.isna(end) or pd.isna(available) or available < end:
        raise ValueError("Fechas de calibración inválidas.")
    if int(profile["training_year"]) != end.year:
        raise ValueError("La campaña del perfil no coincide con sus fechas.")
    offset = float(profile["parameters"]["offset"])
    slope = float(profile["parameters"]["slope"])
    if not np.isfinite([offset, slope]).all():
        raise ValueError("Parámetros de calibración no finitos.")
    if not OFFSET_BOUNDS[0] <= offset <= OFFSET_BOUNDS[1]:
        raise ValueError("Desplazamiento fuera del rango de calibración.")
    if not SLOPE_BOUNDS[0] <= slope <= SLOPE_BOUNDS[1]:
        raise ValueError("Pendiente fuera del rango de calibración.")
    return profile


def load_site_profile(path: str | Path) -> dict | None:
    source = Path(path)
    if not source.exists():
        return None
    return validate_profile(json.loads(source.read_text(encoding="utf-8")))


def _validated_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    frame = trajectory.copy().reset_index(drop=True)
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="raise").dt.normalize()
    if frame.empty or frame["Fecha"].isna().any():
        raise ValueError("La trayectoria está vacía o tiene fechas inválidas.")
    if not frame["Fecha"].is_monotonic_increasing or frame["Fecha"].duplicated().any():
        raise ValueError("La trayectoria debe tener fechas ordenadas y únicas.")
    expected = pd.date_range(frame["Fecha"].min(), frame["Fecha"].max(), freq="D")
    if len(expected) != len(frame):
        raise ValueError("La trayectoria requiere meteorología diaria sin huecos.")
    values = frame["EMERAC_NORMALIZADA"].to_numpy(float)
    calibrated_progress(values)
    if (np.diff(values) < -1e-10).any():
        raise ValueError("La trayectoria acumulada debe ser monótona.")
    return frame


def calibration_intervals(trajectory: pd.DataFrame, observations: pd.DataFrame):
    """Suma sobre (muestreo previo, muestreo actual]; excluye el inicio desconocido."""
    frame = _validated_trajectory(trajectory)
    obs = observations.copy()
    obs["Fecha"] = pd.to_datetime(obs["Fecha"], errors="raise").dt.normalize()
    if obs["Fecha"].isna().any() or obs["Fecha"].duplicated().any():
        raise ValueError("Los muestreos deben tener fechas válidas y únicas.")
    obs = obs.sort_values("Fecha").reset_index(drop=True)
    if len(obs) < 7:
        raise ValueError("Se requieren al menos siete muestreos (seis intervalos).")
    if obs["Fecha"].dt.year.nunique() != 1:
        raise ValueError("Ajuste cada campaña por separado.")
    values = obs["Flujo_observado_PLM2"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Los flujos deben ser finitos y no negativos.")
    progress = frame.set_index("Fecha")["EMERAC_NORMALIZADA"].reindex(obs["Fecha"])
    if progress.isna().any():
        raise ValueError("Falta meteorología para alguna fecha de muestreo.")
    observed = values[1:]
    if observed.sum() <= 0 or np.diff(progress).sum() <= 1e-10:
        raise ValueError("No hay señal suficiente para calibrar esta campaña.")
    se = (
        obs["EE_repeticiones_PLM2"].to_numpy(float)[1:]
        if "EE_repeticiones_PLM2" in obs
        else np.zeros(len(observed))
    )
    if not np.isfinite(se).all() or (se < 0).any():
        raise ValueError("El error estándar debe ser finito y no negativo.")
    # Evita que un cero con tres repeticiones tenga peso infinito.
    sigma = np.maximum(se, max(1.0, 0.10 * observed.max()))
    intervals = pd.DataFrame({
        "Inicio_exclusivo": obs["Fecha"].iloc[:-1].to_numpy(),
        "Fecha": obs["Fecha"].iloc[1:].to_numpy(),
        "Dias_intervalo": obs["Fecha"].diff().dt.days.iloc[1:].to_numpy(int),
        "Observado_PLM2": observed,
        "EE_repeticiones_PLM2": se if "EE_repeticiones_PLM2" in obs else np.nan,
        "Sigma_ajuste_PLM2": sigma,
    })
    return progress.to_numpy(float), intervals


def _window_predictions(progress, observed, offset, slope):
    mass = np.maximum(np.diff(calibrated_progress(progress, offset, slope)), 0.0)
    if mass.sum() <= 1e-12:
        return np.zeros_like(observed), 0.0
    # Esta escala es un parámetro auxiliar del ajuste, NO un potencial del lote.
    scale = float(observed.sum() / mass.sum())
    return mass * scale, scale


def fit_site_calibration(trajectory, observations, *, site: str) -> tuple[dict, pd.DataFrame]:
    """Ajuste determinista, regularizado hacia identidad, sin optimizadores externos."""
    if not site.strip():
        raise ValueError("Indique la localidad de calibración.")
    progress, comparison = calibration_intervals(trajectory, observations)
    observed = comparison["Observado_PLM2"].to_numpy(float)
    sigma = comparison["Sigma_ajuste_PLM2"].to_numpy(float)

    def evaluate(offset, slope):
        predicted, scale = _window_predictions(progress, observed, offset, slope)
        loss = float(np.mean(((predicted - observed) / sigma) ** 2))
        penalty = REGULARIZATION * (offset**2 + np.log(slope)**2)
        return loss + penalty, predicted, scale, loss

    best_offset, best_slope = 0.0, 1.0
    best_score, best_predicted, best_scale, best_loss = evaluate(0.0, 1.0)
    for offset in np.linspace(*OFFSET_BOUNDS, 61):
        for slope in np.linspace(*SLOPE_BOUNDS, 41):
            score, predicted, scale, loss = evaluate(offset, slope)
            if score < best_score - 1e-12:
                best_score, best_predicted, best_scale, best_loss = score, predicted, scale, loss
                best_offset, best_slope = float(offset), float(slope)
    _, baseline, baseline_scale, baseline_loss = evaluate(0.0, 1.0)
    comparison["Base_PLM2_ajuste"] = baseline
    comparison["Calibrado_PLM2_ajuste"] = best_predicted
    comparison["Fraccion_base_intervalo"] = np.diff(progress)
    comparison["Fraccion_calibrada_intervalo"] = np.diff(
        calibrated_progress(progress, best_offset, best_slope)
    )
    last_date = pd.Timestamp(comparison["Fecha"].max()).date().isoformat()
    profile = {
        "schema_version": 1,
        "method": METHOD,
        "profile_id": f"{site.lower()}-{last_date}-v1",
        "site": site,
        "status": "experimental_una_campana",
        "training_year": int(pd.Timestamp(last_date).year),
        "training_through": last_date,
        "available_from": last_date,
        "season_complete": False,
        "first_interval_excluded": True,
        "parameters": {"offset": best_offset, "slope": best_slope},
        "fit": {
            "n_observations": int(len(observations)),
            "n_intervals": int(len(comparison)),
            "observed_window_total_plm2": float(observed.sum()),
            "rmse_base_plm2": float(np.sqrt(np.mean((baseline - observed)**2))),
            "rmse_calibrated_plm2": float(np.sqrt(np.mean((best_predicted - observed)**2))),
            "weighted_mse_base": baseline_loss,
            "weighted_mse_calibrated": best_loss,
            "regularization": REGULARIZATION,
            "nuisance_scale_base": baseline_scale,
            "nuisance_scale_calibrated": best_scale,
            "parameter_at_bound": bool(
                np.isclose(best_offset, OFFSET_BOUNDS).any()
                or np.isclose(best_slope, SLOPE_BOUNDS).any()
            ),
        },
    }
    return validate_profile(profile), comparison


def apply_site_calibration(
    trajectory, profile, *, site: str, as_of, enabled=True, observations=None,
    model_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aplica memoria local antes de asimilar, con control temporal y de localidad.

    Si se asimilan observaciones de la campaña que generó el perfil, se utiliza
    el modelo original: esos datos no se reutilizan como calibración y evidencia
    nueva a la vez. En una campaña posterior sí se combinan memoria y conteos.
    """
    frame = _validated_trajectory(trajectory)
    original = frame["EMERAC_NORMALIZADA"].to_numpy(float)
    frame["EMERAC_BASE_SIN_CALIBRAR"] = original
    audit = {"applied": False, "profile_id": None, "reason": "Sin perfil de calibración."}
    if not enabled:
        audit["reason"] = "Calibración desactivada."
    elif profile is not None:
        validate_profile(profile)
        audit["profile_id"] = profile["profile_id"]
        cutoff = pd.Timestamp(as_of).normalize()
        if site.strip().casefold() != profile["site"].strip().casefold():
            audit["reason"] = "El perfil pertenece a otra localidad."
        elif cutoff < pd.Timestamp(profile["available_from"]):
            audit["reason"] = "El perfil usa observaciones posteriores a la fecha del estado."
        elif (frame["Fecha"].dt.year != cutoff.year).any():
            audit["reason"] = "Seleccione una sola campaña para aplicar la calibración."
        elif (
            profile.get("model_fingerprint") is not None
            and model_fingerprint != profile["model_fingerprint"]
        ):
            audit["reason"] = "La versión del modelo no coincide con la usada para calibrar."
        else:
            reuse = False
            if observations is not None and not observations.empty:
                dates = pd.to_datetime(observations["Fecha"], errors="raise")
                reuse = bool((
                    dates.dt.year.eq(int(profile["training_year"]))
                    & dates.le(cutoff)
                ).any())
            if reuse:
                audit["reason"] = (
                    "Se asimilan conteos de la campaña de calibración. Se usa la base "
                    "original para evitar reutilizar esa evidencia; el perfil queda "
                    "disponible para campañas posteriores."
                )
            else:
                frame["EMERAC_NORMALIZADA"] = calibrated_progress(
                    original, **profile["parameters"]
                )
                audit.update(applied=True, reason="Calibración local experimental activa.")
    frame["EMERAC_CALIBRADA"] = frame["EMERAC_NORMALIZADA"]
    frame["EMERREL_CALIBRADA"] = frame["EMERAC_CALIBRADA"].diff().fillna(
        frame["EMERAC_CALIBRADA"]
    ).clip(lower=0.0)
    frame["Calibracion_Perfil"] = audit["profile_id"] or ""
    frame["Calibracion_Aplicada"] = audit["applied"]
    frame["Calibracion_Motivo"] = audit["reason"]
    return frame, audit
