from copy import deepcopy
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from predweem_twin.assimilation import assimilate_observations
from predweem_twin.calibration import (
    apply_site_calibration, calibrated_progress, calibration_intervals,
    fit_site_calibration, load_site_profile, model_fingerprint,
)
from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem
from predweem_twin.observations import prepare_observations
from predweem_twin.seasonal import load_local_seasonal_reference


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/calibration"


def test_original_attachment_and_frozen_inputs_match_provenance():
    source = saved_profile()["source"]
    original = DATA / source["original_file"]
    assert sha256(original.read_bytes()).hexdigest() == source["uploaded_sha256"]
    excel = pd.read_excel(original)
    csv = pd.read_csv(DATA / source["observations_file"], parse_dates=["FECHA"])
    pd.testing.assert_frame_equal(excel, csv, check_dtype=False)
    for stem in ("observations", "weather"):
        path = DATA / source[stem + "_file"]
        assert sha256(path.read_bytes()).hexdigest() == source[stem + "_sha256"]


def example_trajectory(year=2027):
    return pd.DataFrame({
        "Fecha": pd.date_range(f"{year}-01-01", periods=12),
        "EMERAC_NORMALIZADA": [0, 0, .02, .07, .07, .2, .4, .4, .6, .7, .7, .8],
        "EMERREL": [0, 0, 2, 5, 0, 13, 20, 0, 20, 10, 0, 10],
        "TT_DESDE_PICO": np.arange(12) * 10.,
    })


def saved_profile():
    return load_site_profile(DATA / "lartigau_2026.json")


def apply(frame=None, **kwargs):
    arguments = {
        "site": "Lartigau", "as_of": "2027-01-08",
        "model_fingerprint": model_fingerprint(ROOT),
    }
    arguments.update(kwargs)
    return apply_site_calibration(
        example_trajectory() if frame is None else frame,
        saved_profile(), **arguments,
    )


def test_calibration_preserves_biophysical_gates_partial_tail_and_inputs():
    original = example_trajectory()
    before = original.copy(deep=True)
    result, audit = apply(original)
    assert audit["applied"]
    assert result.EMERAC_CALIBRADA.between(0, 1).all()
    assert result.EMERAC_CALIBRADA.is_monotonic_increasing
    assert result.EMERAC_CALIBRADA.iloc[-1] < 1  # partial season is not completed
    no_flux = before.EMERAC_NORMALIZADA.diff().fillna(0).eq(0)
    assert result.loc[no_flux, "EMERREL_CALIBRADA"].eq(0).all()
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_series_equal(result.EMERREL, before.EMERREL)
    pd.testing.assert_series_equal(result.TT_DESDE_PICO, before.TT_DESDE_PICO)
    assert np.array_equal(calibrated_progress([0, 1], .8, 1.3), [0, 1])


@pytest.mark.parametrize("arguments", [
    {"enabled": False}, {"site": "Bordenave"},
    {"as_of": "2026-04-01"}, {"model_fingerprint": "other-model"},
])
def test_inapplicable_profile_returns_exact_original(arguments):
    original = example_trajectory()
    result, audit = apply(original, **arguments)
    assert not audit["applied"]
    np.testing.assert_array_equal(result.EMERAC_NORMALIZADA, original.EMERAC_NORMALIZADA)


def test_same_campaign_observations_are_not_reused_for_calibration_and_assimilation():
    frame = example_trajectory(2026)
    obs = pd.DataFrame({"Fecha": ["2026-01-04"], "Observado": [.1]})
    result, audit = apply(frame, as_of="2026-09-18", observations=obs)
    assert not audit["applied"]
    assert "reutilizar" in audit["reason"]
    original_twin, _ = assimilate_observations(frame, obs)
    actual_twin, _ = assimilate_observations(result, obs)
    np.testing.assert_array_equal(actual_twin.EMERAC_TWIN, original_twin.EMERAC_TWIN)


def test_new_season_observations_combine_with_persistent_profile():
    obs = pd.DataFrame({"Fecha": ["2027-01-06"], "Observado": [.25]})
    result, audit = apply(observations=obs)
    assert audit["applied"]
    twin, assimilation_audit = assimilate_observations(result, obs)
    assert len(assimilation_audit) == 1
    assert twin.EMERAC_TWIN.is_monotonic_increasing
    assert twin.EMERAC_TWIN.between(0, 1).all()
    assert twin.EMERAC_BASE_SIN_CALIBRAR.equals(result.EMERAC_BASE_SIN_CALIBRAR)


def test_invalid_profile_cannot_change_outputs():
    profile = deepcopy(saved_profile())
    profile["parameters"]["slope"] = float("nan")
    with pytest.raises(ValueError, match="finitos"):
        apply_site_calibration(example_trajectory(), profile, site="Lartigau", as_of="2027-01-08")


def test_known_transformation_can_be_learned_without_seasonal_total():
    dates = pd.date_range("2026-01-01", periods=100)
    progress = np.linspace(.01, .75, 100)
    frame = pd.DataFrame({"Fecha": dates, "EMERAC_NORMALIZADA": progress})
    sample = np.arange(2, 100, 7)
    truth = calibrated_progress(progress, offset=.8, slope=1.3)
    flows = np.diff(np.r_[0., truth[sample]]) * 1500
    obs = pd.DataFrame({"Fecha": dates[sample], "Flujo_observado_PLM2": flows})
    profile, comparison = fit_site_calibration(frame, obs, site="Example")
    assert profile["fit"]["rmse_calibrated_plm2"] < profile["fit"]["rmse_base_plm2"] * .5
    assert len(comparison) == len(obs) - 1
    assert profile["season_complete"] is False
    altered = obs.copy()
    altered.loc[0, "Flujo_observado_PLM2"] *= 100
    other, _ = fit_site_calibration(frame, altered, site="Example")
    assert other["parameters"] == profile["parameters"]  # unknown first interval excluded


@pytest.fixture(scope="module")
def real_data():
    weather = pd.read_csv(DATA / "lartigau_2026_weather.csv")
    model = PracticalANNModel.from_directory(ROOT / "models")
    reference = load_local_seasonal_reference(ROOT, as_of="2026-08-30")
    trajectory = run_predweem(
        weather, model, ModelParameters(),
        normalization_as_of="2026-08-30", seasonal_reference=reference,
    )
    raw = pd.read_csv(DATA / "lartigau_2026_counts.csv")
    prepared, metadata = prepare_observations(raw, trajectory)
    return trajectory, raw, prepared, metadata


def test_source_units_missing_replicates_and_irregular_intervals(real_data):
    trajectory, raw, prepared, metadata = real_data
    assert len(raw) == 15 and "n_repeticiones" not in metadata
    assert raw.columns.tolist() == ["FECHA", "PLM2"]
    assert raw.PLM2.sum() == pytest.approx(3933.5)
    _, intervals = calibration_intervals(trajectory, prepared)
    assert intervals.iloc[0]["Inicio_exclusivo"] == pd.Timestamp("2026-02-01")
    assert intervals.iloc[0]["Fecha"] == pd.Timestamp("2026-02-19")
    assert intervals.iloc[0]["Dias_intervalo"] == 18
    assert intervals.iloc[0]["Observado_PLM2"] == pytest.approx(600.)
    assert intervals.loc[intervals.Fecha.eq("2026-07-03"), "Dias_intervalo"].iloc[0] == 16
    assert intervals.Dias_intervalo.min() == 5
    assert intervals.Dias_intervalo.max() == 21
    assert len(intervals) == 14
    assert intervals.EE_repeticiones_PLM2.isna().all()
    assert intervals.Sigma_ajuste_PLM2.eq(170.8).all()
    assert intervals.Observado_PLM2.sum() == pytest.approx(raw.iloc[:, -1].sum())


def test_calibration_keeps_lartigau_decay_and_shared_reference(real_data):
    trajectory, _, _, _ = real_data
    result, audit = apply(trajectory, as_of="2026-08-30")
    assert audit["applied"]
    for column in ("EMERREL", "Factor_Decaimiento_15Abr", "Techo_EMERREL_15Abr", "TT_DESDE_PICO"):
        pd.testing.assert_series_equal(trajectory[column], result[column])
    blocked = trajectory.EMERAC_NORMALIZADA.diff().fillna(0).eq(0)
    assert result.loc[blocked, "EMERREL_CALIBRADA"].eq(0).all()
    saved = saved_profile()
    assert saved["model_parameters"]["cobertura_pct"] == 75.
    assert saved["model_parameters"]["w_max"] == 18.816
    assert saved["model_parameters"]["decay_tau_days"] == 60.
    assert saved["model_parameters"]["decay_cap_fraction"] == .5
    assert "lartigau_2026_counts.csv" in saved["seasonal_reference"]["campaigns"]
    assert "tresas" not in saved["seasonal_reference"]["campaigns"].lower()
    assert saved["seasonal_reference"]["excluded_years"] == ["2010", "2015"]
    assert saved["seasonal_reference"]["excluded_sites"] == ["balcarce", "san pedro"]
    assert "balcarce" not in saved["seasonal_reference"]["campaigns"].lower()
    assert "san pedro" not in saved["seasonal_reference"]["campaigns"].lower()
    assert saved["seasonal_reference"]["n_campaigns"] == 9


def test_fit_matches_persisted_profile_and_does_not_mutate_network(real_data):
    trajectory, _, prepared, _ = real_data
    fingerprint = model_fingerprint(ROOT)
    profile, comparison = fit_site_calibration(trajectory, prepared, site="Lartigau")
    saved = saved_profile()
    assert profile["parameters"] == saved["parameters"]
    assert profile["fit"]["rmse_base_plm2"] == pytest.approx(saved["fit"]["rmse_base_plm2"])
    assert profile["fit"]["rmse_calibrated_plm2"] == pytest.approx(saved["fit"]["rmse_calibrated_plm2"])
    assert model_fingerprint(ROOT) == fingerprint == saved["model_fingerprint"]


@pytest.mark.parametrize("fault", ["negative", "duplicate", "missing_weather", "nan"])
def test_invalid_training_data_are_rejected(real_data, fault):
    trajectory, _, prepared, _ = real_data
    trajectory, obs = trajectory.copy(), prepared.copy()
    if fault == "negative":
        obs.loc[3, "Flujo_observado_PLM2"] = -1
    elif fault == "duplicate":
        obs.loc[3, "Fecha"] = obs.loc[2, "Fecha"]
    elif fault == "missing_weather":
        trajectory = trajectory.drop(index=80)
    else:
        obs.loc[3, "Flujo_observado_PLM2"] = np.nan
    with pytest.raises(ValueError):
        fit_site_calibration(trajectory, obs, site="Lartigau")
