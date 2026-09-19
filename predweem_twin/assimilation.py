"""Asimilación secuencial de flujos de emergencia y estados acumulados."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_gain(model_uncertainty: float, observation_uncertainty: float) -> float:
    """Ganancia escalar para magnitudes expresadas como fracción 0–1."""
    p = max(float(model_uncertainty), 1e-6) ** 2
    r = max(float(observation_uncertainty), 1e-6) ** 2
    return p / (p + r)


def estimate_flow_potential(
    model_intervals,
    observed_flows,
    model_progress: float,
    seasonal_potential_prior: float | None = None,
    potential_prior_cv: float = 0.30,
) -> dict:
    """Estima el potencial estacional sin tratar la serie parcial como completa.

    Combina una estimación acumulada estructural con el ajuste de los flujos por
    intervalo. El ajuste por intervalos recibe peso únicamente cuando reproduce
    razonablemente la forma observada y todavía queda emergencia por ocurrir.
    Un potencial histórico opcional domina al inicio y pierde peso a medida que
    avanza la campaña.
    """
    intervals = np.clip(np.asarray(model_intervals, dtype=float), 0.0, None)
    flows = np.clip(np.asarray(observed_flows, dtype=float), 0.0, None)
    if len(intervals) != len(flows) or len(flows) == 0:
        raise ValueError("Los flujos y los intervalos simulados deben tener igual longitud.")

    progress = float(np.clip(model_progress, 0.0, 1.0))
    observed_total = float(flows.sum())
    structural_total = observed_total / max(progress, 0.05)

    denominator = float(np.dot(intervals, intervals))
    flow_total = (
        float(np.dot(intervals, flows) / denominator)
        if denominator > 1e-12
        else structural_total
    )
    fitted = intervals * flow_total
    centered_total = float(np.square(flows - flows.mean()).sum())
    if len(flows) >= 3 and centered_total > 1e-12:
        fit_quality = float(
            np.clip(1.0 - np.square(flows - fitted).sum() / centered_total, 0.0, 1.0)
        )
    else:
        fit_quality = 0.0

    prior = None
    if seasonal_potential_prior is not None and float(seasonal_potential_prior) > 0:
        prior = float(seasonal_potential_prior)
        evidence_weight = float(np.clip(progress / 0.95, 0.0, 1.0))
        structural_total = (
            (1.0 - evidence_weight) * prior + evidence_weight * structural_total
        )

    flow_weight = fit_quality * max(0.0, 1.0 - progress)
    potential = max(
        observed_total,
        (1.0 - flow_weight) * structural_total + flow_weight * flow_total,
    )
    potential_cv = float(
        np.clip(
            0.05
            + float(potential_prior_cv)
            * (1.0 - progress)
            * (1.0 - fit_quality),
            0.05,
            0.60,
        )
    )
    return {
        "potential": float(potential),
        "potential_cv": potential_cv,
        "observed_total": observed_total,
        "structural_total": float(structural_total),
        "flow_total": float(flow_total),
        "fit_quality": fit_quality,
        "flow_weight": float(flow_weight),
        "prior": prior,
    }


def _base_frame(trajectory: pd.DataFrame) -> pd.DataFrame:
    df = trajectory.copy().sort_values("Fecha").reset_index(drop=True)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.tz_localize(None)
    df["EMERAC_TWIN"] = pd.to_numeric(
        df["EMERAC_NORMALIZADA"], errors="coerce"
    ).fillna(0.0)
    df["EMERREL_TWIN"] = (
        df["EMERAC_TWIN"].diff().fillna(df["EMERAC_TWIN"]).clip(lower=0.0)
    )
    df["POTENCIAL_ESTACIONAL_PLM2"] = np.nan
    df["EMERAC_TWIN_PLM2"] = np.nan
    df["EMERREL_TWIN_PLM2"] = np.nan
    df["MODO_ASIMILACION"] = "sin observaciones"
    df["ULTIMA_OBSERVACION"] = pd.NaT
    return df


def _reanchor_future(df: pd.DataFrame, idx: int, posterior: float) -> None:
    base_anchor = float(df.at[idx, "EMERAC_NORMALIZADA"])
    base_future = df.loc[idx:, "EMERAC_NORMALIZADA"].to_numpy(float)
    if base_anchor < 1.0 - 1e-9:
        progress = np.clip(
            (base_future - base_anchor) / (1.0 - base_anchor), 0.0, 1.0
        )
        adjusted = posterior + (1.0 - posterior) * progress
    else:
        adjusted = np.full(len(base_future), posterior)
    df.loc[idx:, "EMERAC_TWIN"] = np.maximum.accumulate(
        np.clip(adjusted, posterior, 1.0)
    )


def _adjust_interval(
    df: pd.DataFrame,
    previous_idx: int,
    idx: int,
    previous_state: float,
    prior_interval: float,
    posterior_interval: float,
) -> None:
    start = previous_idx + 1
    path = df.loc[start:idx, "EMERAC_TWIN"].to_numpy(float)
    previous_path = np.r_[previous_state, path[:-1]]
    increments = np.clip(path - previous_path, 0.0, None)
    if prior_interval > 1e-12:
        adjusted_increments = increments * (posterior_interval / prior_interval)
    else:
        base_daily = (
            df.loc[start:idx, "EMERAC_NORMALIZADA"]
            .diff()
            .fillna(
                float(df.at[start, "EMERAC_NORMALIZADA"])
                - (
                    float(df.at[previous_idx, "EMERAC_NORMALIZADA"])
                    if previous_idx >= 0
                    else 0.0
                )
            )
            .clip(lower=0.0)
            .to_numpy(float)
        )
        if base_daily.sum() > 1e-12:
            adjusted_increments = posterior_interval * base_daily / base_daily.sum()
        else:
            adjusted_increments = np.zeros(len(path), dtype=float)
            if len(adjusted_increments):
                adjusted_increments[-1] = posterior_interval
    df.loc[start:idx, "EMERAC_TWIN"] = previous_state + np.cumsum(
        adjusted_increments
    )


def _flow_column(observations: pd.DataFrame) -> pd.Series | None:
    for column in ("Flujo_observado_PLM2", "flow_plm2"):
        if column in observations:
            values = pd.to_numeric(observations[column], errors="coerce")
            if values.notna().any():
                return values
    if {"Valor_original", "Unidad_original"}.issubset(observations.columns):
        units = observations["Unidad_original"].astype(str).str.lower()
        mask = units.str.contains("por intervalo", na=False)
        values = pd.to_numeric(observations["Valor_original"], errors="coerce")
        return values.where(mask)
    return None


def _assimilate_interval_flows(
    df: pd.DataFrame,
    observations: pd.DataFrame,
    flow_values: pd.Series,
    model_uncertainty: float,
    default_observation_uncertainty: float,
    seasonal_potential_prior: float | None,
    potential_prior_cv: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = observations.copy()
    obs["Fecha"] = pd.to_datetime(obs["Fecha"], errors="coerce").dt.tz_localize(None)
    obs["Flujo_observado_PLM2"] = pd.to_numeric(flow_values, errors="coerce")
    if "Incertidumbre" not in obs:
        obs["Incertidumbre"] = default_observation_uncertainty
    obs["Incertidumbre"] = pd.to_numeric(
        obs["Incertidumbre"], errors="coerce"
    ).fillna(default_observation_uncertainty)
    if "EE_repeticiones_PLM2" not in obs:
        obs["EE_repeticiones_PLM2"] = np.nan
    obs["EE_repeticiones_PLM2"] = pd.to_numeric(
        obs["EE_repeticiones_PLM2"], errors="coerce"
    )
    obs = (
        obs.dropna(subset=["Fecha", "Flujo_observado_PLM2"])
        .sort_values("Fecha")
        .reset_index(drop=True)
    )
    obs = obs[obs["Flujo_observado_PLM2"] >= 0].reset_index(drop=True)
    if obs.empty:
        return df, pd.DataFrame()

    base_cumulative = df["EMERAC_NORMALIZADA"].to_numpy(float)
    model_intervals: list[float] = []
    observed_flows: list[float] = []
    previous_base_state = 0.0
    previous_idx = -1
    records = []

    for row in obs.itertuples(index=False):
        valid_indices = df.index[df["Fecha"] <= row.Fecha].tolist()
        if not valid_indices:
            continue
        idx = valid_indices[-1]
        if idx <= previous_idx:
            continue

        base_state = float(base_cumulative[idx])
        base_interval = max(0.0, base_state - previous_base_state)
        model_intervals.append(base_interval)
        observed_flows.append(float(row.Flujo_observado_PLM2))
        records.append((row, idx, base_interval))
        previous_idx = idx
        previous_base_state = base_state

    if not records:
        return df, pd.DataFrame()

    final_base_state = float(base_cumulative[records[-1][1]])
    potential_info = estimate_flow_potential(
        model_intervals,
        observed_flows,
        final_base_state,
        seasonal_potential_prior=seasonal_potential_prior,
        potential_prior_cv=potential_prior_cv,
    )
    final_potential = potential_info["potential"]
    previous_idx = -1
    previous_posterior = 0.0
    cumulative_observed = 0.0
    audit = []

    for row, idx, base_interval in records:
        potential = final_potential
        observed_flow_fraction = float(row.Flujo_observado_PLM2) / potential

        prior_state = float(df.at[idx, "EMERAC_TWIN"])
        prior_interval = max(0.0, prior_state - previous_posterior)
        replicate_se = getattr(row, "EE_repeticiones_PLM2", np.nan)
        replicate_uncertainty = (
            float(replicate_se) / potential
            if pd.notna(replicate_se) and potential > 0
            else 0.0
        )
        potential_component = (
            observed_flow_fraction * potential_info["potential_cv"]
        )
        observation_uncertainty = max(
            float(row.Incertidumbre),
            float(np.hypot(replicate_uncertainty, potential_component)),
            1e-6,
        )
        gain = kalman_gain(model_uncertainty, observation_uncertainty)
        posterior_interval = float(
            np.clip(
                prior_interval
                + gain * (observed_flow_fraction - prior_interval),
                0.0,
                1.0 - previous_posterior,
            )
        )
        posterior = previous_posterior + posterior_interval
        _adjust_interval(
            df,
            previous_idx,
            idx,
            previous_posterior,
            prior_interval,
            posterior_interval,
        )
        _reanchor_future(df, idx, posterior)

        cumulative_observed += float(row.Flujo_observado_PLM2)
        field_state = float(np.clip(cumulative_observed / potential, 0.0, 1.0))
        audit.append(
            {
                "Modo_asimilacion": "flujo por intervalo",
                "Fecha_observacion": row.Fecha,
                "Fecha_asimilada": df.at[idx, "Fecha"],
                "Flujo_observado_PLM2": float(row.Flujo_observado_PLM2),
                "Flujo_modelo_fraccion": base_interval,
                "Flujo_previo_fraccion": prior_interval,
                "Flujo_observado_fraccion": observed_flow_fraction,
                "Acumulado_observado_PLM2": cumulative_observed,
                "Potencial_estacional_PLM2": potential,
                "CV_potencial": potential_info["potential_cv"],
                "Calidad_ajuste_flujos": potential_info["fit_quality"],
                "Estado_campo_estimado": field_state,
                "Pronostico_previo": prior_state,
                "Incertidumbre_observacion": observation_uncertainty,
                "Ganancia": gain,
                "Estado_posterior": posterior,
                "Innovacion_flujo": observed_flow_fraction - prior_interval,
            }
        )
        previous_idx = idx
        previous_posterior = posterior

    df["EMERAC_TWIN"] = np.maximum.accumulate(
        np.clip(df["EMERAC_TWIN"].to_numpy(float), 0.0, 1.0)
    )
    df["EMERREL_TWIN"] = (
        df["EMERAC_TWIN"].diff().fillna(df["EMERAC_TWIN"]).clip(lower=0.0)
    )
    if np.isfinite(final_potential):
        df["POTENCIAL_ESTACIONAL_PLM2"] = final_potential
        df["EMERAC_TWIN_PLM2"] = df["EMERAC_TWIN"] * final_potential
        df["EMERREL_TWIN_PLM2"] = df["EMERREL_TWIN"] * final_potential
    df["MODO_ASIMILACION"] = "flujos por intervalo"
    df["ULTIMA_OBSERVACION"] = max(row.Fecha for row, _, _ in records)
    return df, pd.DataFrame(audit)


def _assimilate_cumulative(
    df: pd.DataFrame,
    observations: pd.DataFrame,
    model_uncertainty: float,
    default_observation_uncertainty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = observations.copy()
    rename = {
        "fecha": "Fecha",
        "emergencia_acumulada": "Observado",
        "observado": "Observado",
        "incertidumbre": "Incertidumbre",
    }
    obs = obs.rename(
        columns={
            column: rename.get(str(column).strip().lower(), column)
            for column in obs.columns
        }
    )
    required = {"Fecha", "Observado"}
    if not required.issubset(obs.columns):
        raise ValueError("Las observaciones requieren las columnas Fecha y Observado.")
    obs["Fecha"] = pd.to_datetime(obs["Fecha"], errors="coerce").dt.tz_localize(None)
    obs["Observado"] = pd.to_numeric(obs["Observado"], errors="coerce")
    if "Incertidumbre" not in obs:
        obs["Incertidumbre"] = default_observation_uncertainty
    obs["Incertidumbre"] = pd.to_numeric(
        obs["Incertidumbre"], errors="coerce"
    ).fillna(default_observation_uncertainty)
    obs = obs.dropna(subset=["Fecha", "Observado"]).sort_values("Fecha")
    obs["Observado"] = obs["Observado"].clip(0.0, 1.0)

    audit = []
    for row in obs.itertuples(index=False):
        valid_indices = df.index[df["Fecha"] <= row.Fecha].tolist()
        if not valid_indices:
            continue
        idx = valid_indices[-1]
        prior = float(df.at[idx, "EMERAC_TWIN"])
        gain = kalman_gain(model_uncertainty, row.Incertidumbre)
        posterior = float(
            np.clip(prior + gain * (row.Observado - prior), 0.0, 1.0)
        )

        before = df.loc[:idx, "EMERAC_TWIN"].to_numpy(copy=True)
        if prior > 1e-9:
            before = before * (posterior / prior)
        else:
            before[-1] = posterior
        df.loc[:idx, "EMERAC_TWIN"] = np.maximum.accumulate(
            np.clip(before, 0.0, posterior)
        )
        _reanchor_future(df, idx, posterior)

        audit.append(
            {
                "Modo_asimilacion": "acumulado normalizado",
                "Fecha_observacion": row.Fecha,
                "Fecha_asimilada": df.at[idx, "Fecha"],
                "Pronostico_previo": prior,
                "Observado": float(row.Observado),
                "Estado_campo_estimado": float(row.Observado),
                "Incertidumbre_observacion": float(row.Incertidumbre),
                "Ganancia": gain,
                "Estado_posterior": posterior,
                "Innovacion": float(row.Observado - prior),
            }
        )

    df["EMERAC_TWIN"] = np.maximum.accumulate(
        df["EMERAC_TWIN"].clip(0.0, 1.0)
    )
    df["EMERREL_TWIN"] = (
        df["EMERAC_TWIN"].diff().fillna(df["EMERAC_TWIN"]).clip(lower=0.0)
    )
    df["MODO_ASIMILACION"] = "acumulado normalizado"
    if not obs.empty:
        df["ULTIMA_OBSERVACION"] = obs["Fecha"].max()
    return df, pd.DataFrame(audit)


def assimilate_observations(
    trajectory: pd.DataFrame,
    observations: pd.DataFrame | None,
    model_uncertainty: float = 0.12,
    default_observation_uncertainty: float = 0.08,
    seasonal_potential_prior: float | None = None,
    potential_prior_cv: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actualiza el gemelo con flujos por intervalo o acumulados normalizados.

    Los archivos de plantas/m² se asimilan directamente como sumas del flujo
    diario entre muestreos. Los acumulados 0–1/0–100 mantienen el método escalar
    anterior para conservar compatibilidad.
    """
    df = _base_frame(trajectory)
    if observations is None or observations.empty:
        return df, pd.DataFrame()

    obs = observations.copy()
    if "Fecha" not in obs:
        obs = obs.rename(
            columns={
                column: "Fecha"
                for column in obs.columns
                if str(column).strip().lower() == "fecha"
            }
        )
    flow_values = _flow_column(obs)
    if flow_values is not None and flow_values.notna().all():
        return _assimilate_interval_flows(
            df,
            obs,
            flow_values,
            model_uncertainty,
            default_observation_uncertainty,
            seasonal_potential_prior,
            potential_prior_cv,
        )
    return _assimilate_cumulative(
        df,
        obs,
        model_uncertainty,
        default_observation_uncertainty,
    )
