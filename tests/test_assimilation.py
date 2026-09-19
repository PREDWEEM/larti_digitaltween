import numpy as np
import pandas as pd

from predweem_twin.assimilation import (
    assimilate_observations,
    estimate_flow_potential,
    kalman_gain,
)


def sample_trajectory():
    dates = pd.date_range("2026-03-01", periods=10)
    accumulated = np.linspace(0, 1, 10)
    return pd.DataFrame({"Fecha": dates, "EMERAC_NORMALIZADA": accumulated})


def test_kalman_gain_responds_to_uncertainty():
    assert kalman_gain(0.12, 0.03) > kalman_gain(0.12, 0.20)


def test_assimilation_is_bounded_monotonic_and_audited():
    observations = pd.DataFrame(
        {"Fecha": ["2026-03-05"], "Observado": [0.30], "Incertidumbre": [0.05]}
    )
    adjusted, audit = assimilate_observations(sample_trajectory(), observations)
    assert adjusted["EMERAC_TWIN"].between(0, 1).all()
    assert (adjusted["EMERAC_TWIN"].diff().fillna(0) >= -1e-12).all()
    assert len(audit) == 1
    assert audit.iloc[0]["Estado_posterior"] < audit.iloc[0]["Pronostico_previo"]
    assert np.isclose(adjusted.iloc[-1]["EMERAC_TWIN"], 1.0)


def test_flow_potential_does_not_treat_partial_series_as_complete():
    result = estimate_flow_potential(
        model_intervals=[0.20, 0.10, 0.15],
        observed_flows=[200.0, 100.0, 150.0],
        model_progress=0.45,
    )
    assert result["potential"] > result["observed_total"]
    assert np.isclose(result["potential"], 1000.0)
    assert result["potential_cv"] >= 0.05


def test_interval_flows_are_assimilated_against_daily_model_sum():
    trajectory = sample_trajectory()
    observations = pd.DataFrame(
        {
            "Fecha": ["2026-03-03", "2026-03-06"],
            "Flujo_observado_PLM2": [20.0, 35.0],
            "Incertidumbre": [0.05, 0.05],
            "EE_repeticiones_PLM2": [2.0, 3.0],
        }
    )
    adjusted, audit = assimilate_observations(trajectory, observations)
    assert audit["Modo_asimilacion"].eq("flujo por intervalo").all()
    assert np.isclose(audit.iloc[0]["Flujo_modelo_fraccion"], 2 / 9)
    assert np.isclose(audit.iloc[1]["Flujo_modelo_fraccion"], 3 / 9)
    assert adjusted["EMERAC_TWIN"].is_monotonic_increasing
    assert adjusted["EMERAC_TWIN"].between(0, 1).all()
    assert adjusted["POTENCIAL_ESTACIONAL_PLM2"].notna().all()
    assert adjusted["EMERREL_TWIN_PLM2"].notna().all()
    assert audit["Potencial_estacional_PLM2"].nunique() == 1
    assert audit["Acumulado_observado_PLM2"].tolist() == [20.0, 55.0]
    assert np.isclose(adjusted.iloc[-1]["EMERAC_TWIN"], 1.0)
