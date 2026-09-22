"""Composición, peso por campaña y disponibilidad del pool de Lartigau."""

from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from predweem_twin.seasonal import load_local_seasonal_reference, load_seasonal_reference
from predweem_twin.flows import historical_weekly_max


ROOT = Path(__file__).parents[1]
YEARS = (2008, 2009, 2011, 2012, 2013, 2014, 2023, 2024)
COUNTS = Path("data/calibration/lartigau_2026_counts.csv")
MODEL = Path("models/modelo_clusters_k3.pkl")


@pytest.fixture
def local_files(tmp_path):
    for path in (COUNTS, MODEL):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    return tmp_path


def test_nine_campaign_percentiles_have_equal_individual_weight():
    reference = load_local_seasonal_reference(ROOT, "2027-05-05")
    with (ROOT / MODEL).open("rb") as handle:
        payload = pickle.load(handle)
    expected = []
    for year in YEARS:
        raw = np.asarray(payload["curves_interp"])[payload["names"].index(f"{year}.xlsx")]
        expected.append(raw.cumsum() / raw.sum())
    counts = pd.read_csv(ROOT / COUNTS)
    flows = counts["PLM2"].to_numpy()
    dates = pd.to_datetime(counts.FECHA)
    expected.append(np.interp(
        reference.Julian_days, dates.dt.dayofyear, flows.cumsum() / flows.sum(),
        left=np.nan, right=1.,
    ))
    expected = np.asarray(expected)
    assert reference.N_Campanas.eq(9).all()
    assert reference.Campanas_Anos.eq(", ".join(map(str, (*YEARS, 2026)))).all()
    for q, column in [(0.1, "Progreso_P10"), (.5, "Progreso_Mediano"), (.9, "Progreso_P90")]:
        quantile = np.nanquantile(expected, q, axis=0)
        np.testing.assert_allclose(reference[column + "_Empirico"], quantile)
        np.testing.assert_allclose(reference[column], np.maximum.accumulate(quantile))
        assert reference[column].between(0, 1 + 1e-12).all()
    # Evitar dar 50 % del peso a 2026 y 50 % al resumen de las otras ocho.
    grouped = np.nanmean(np.array([np.median(expected[:8], axis=0), expected[8]]), axis=0)
    assert not np.allclose(reference.Progreso_Mediano, grouped)
    assert reference.attrs["source_2026"]["window_total_plm2"] == pytest.approx(3933.5)
    np.testing.assert_allclose(
        reference.set_index("Julian_days").loc[dates.dt.dayofyear, "Progreso_2026"],
        flows.cumsum() / flows.sum(),
    )
    assert reference.loc[reference.Julian_days.lt(32), "Progreso_2026"].isna().all()
    assert reference.loc[reference.Julian_days.lt(32), "N_Campanas_Dia"].eq(8).all()


def test_2026_total_is_unavailable_before_last_visit_even_if_future_counts_change(local_files):
    cutoff = "2026-08-29"
    earlier = load_local_seasonal_reference(local_files, cutoff)
    assert earlier.N_Campanas.eq(8).all()
    assert "Progreso_2026" not in earlier
    assert not earlier.attrs["source_2026"]["used"]
    counts = pd.read_csv(local_files / COUNTS)
    counts.loc[len(counts) - 1, "PLM2"] = 999999.
    counts.to_csv(local_files / COUNTS, index=False)
    altered = load_local_seasonal_reference(local_files, cutoff)
    np.testing.assert_array_equal(earlier.Progreso_Mediano, altered.Progreso_Mediano)
    assert historical_weekly_max(earlier, cutoff) == historical_weekly_max(altered, cutoff)
    available = load_local_seasonal_reference(local_files, "2026-08-30")
    assert available.N_Campanas.eq(9).all()
    assert available.attrs["source_2026"]["used"]


def test_density_scaling_does_not_change_campaign_weight(local_files):
    original = load_local_seasonal_reference(local_files, "2027-05-05")
    counts = pd.read_csv(local_files / COUNTS)
    counts["PLM2"] *= 100
    counts.to_csv(local_files / COUNTS, index=False)
    scaled = load_local_seasonal_reference(local_files, "2027-05-05")
    np.testing.assert_allclose(original.Progreso_Mediano, scaled.Progreso_Mediano)
    assert historical_weekly_max(original, "2027-05-05") == pytest.approx(
        historical_weekly_max(scaled, "2027-05-05")
    )


def test_excluded_campaigns_cannot_change_the_local_pool_or_intensity_reference(local_files):
    original = load_local_seasonal_reference(local_files, "2027-05-05")
    with (local_files / MODEL).open("rb") as handle:
        payload = pickle.load(handle)
    selected = {f"{year}.xlsx" for year in YEARS}
    for i, name in enumerate(payload["names"]):
        if name not in selected:
            payload["curves_interp"][i] = np.arange(len(payload["JD_common"])) * 999
    with (local_files / MODEL).open("wb") as handle:
        pickle.dump(payload, handle)
    altered = load_local_seasonal_reference(local_files, "2027-05-05")
    pd.testing.assert_frame_equal(original, altered)
    assert historical_weekly_max(original, "2027-05-05") == historical_weekly_max(altered, "2027-05-05")


def test_duplicate_or_missing_year_is_rejected(local_files):
    with (local_files / MODEL).open("rb") as handle:
        payload = pickle.load(handle)
    payload["names"][payload["names"].index("2008.xlsx")] = "2009.xlsx"
    with (local_files / MODEL).open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="ocho series"):
        load_seasonal_reference(local_files / MODEL)


@pytest.mark.parametrize("fault", ["duplicate", "negative", "nan", "no_initial_zero", "wrong_year"])
def test_invalid_counts_fail_instead_of_silently_changing_the_pool(local_files, fault):
    counts = pd.read_csv(local_files / COUNTS)
    if fault == "duplicate":
        counts.loc[2, "FECHA"] = counts.loc[1, "FECHA"]
    elif fault == "wrong_year":
        counts.loc[0, "FECHA"] = "2025-02-01"
    else:
        counts.loc[0 if fault == "no_initial_zero" else 2, "PLM2"] = {
            "negative": -1., "nan": np.nan, "no_initial_zero": 1.,
        }[fault]
    counts.to_csv(local_files / COUNTS, index=False)
    with pytest.raises(ValueError, match="Conteos 2026 inválidos"):
        load_local_seasonal_reference(local_files, "2027-05-05")
