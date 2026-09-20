from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem
from predweem_twin.seasonal import load_seasonal_reference


ROOT = Path(__file__).parents[1]


def test_shared_reference_excludes_balcarce_san_pedro_2010_and_2015():
    source = ROOT / "models/modelo_clusters_k3.pkl"
    reference = load_seasonal_reference(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    expected_names = {
        "2008.xlsx", "2009.xlsx", "2011.xlsx", "2012.xlsx", "2013.xlsx",
        "2014.xlsx", "2023.xlsx", "2024.xlsx", "test -emerel tresas 2025.xlsx",
    }
    selected = [i for i, name in enumerate(payload["names"]) if name in expected_names]
    curves = np.asarray(payload["curves_interp"])[selected]
    progress = np.cumsum(curves, axis=1) / curves.sum(axis=1, keepdims=True)
    assert len(selected) == 9
    assert reference.N_Campanas.eq(9).all()
    assert set(reference.Campanas.iloc[0].split(", ")) == expected_names
    for quantile, column in [(0.1, "Progreso_P10"), (.5, "Progreso_Mediano"), (.9, "Progreso_P90")]:
        np.testing.assert_allclose(reference[column], np.quantile(progress, quantile, axis=0))


def test_excluded_curves_cannot_change_the_reference(tmp_path):
    source = ROOT / "models/modelo_clusters_k3.pkl"
    expected = load_seasonal_reference(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    for i, name in enumerate(payload["names"]):
        if "balcarce" in name.lower() or "san pedro" in name.lower():
            payload["curves_interp"][i] = np.arange(len(payload["JD_common"])) * 1000
            payload["names"][i] = name.upper().replace("SAN PEDRO", "SAN   PEDRO")
    altered = tmp_path / "altered.pkl"
    with altered.open("wb") as handle:
        pickle.dump(payload, handle)
    result = load_seasonal_reference(altered)
    pd.testing.assert_frame_equal(result.drop(columns="Campanas_Excluidas"), expected.drop(columns="Campanas_Excluidas"))


def test_missing_curve_names_cannot_bypass_site_filter(tmp_path):
    path = tmp_path / "unnamed.pkl"
    with path.open("wb") as handle:
        pickle.dump({"JD_common": [1, 2], "curves_interp": [[0, 1]]}, handle)
    with pytest.raises(ValueError, match="nombre por curva"):
        load_seasonal_reference(path)


def test_local_selector_remains_exclusive_and_cannot_reinclude_excluded_sites():
    source = ROOT / "models/modelo_clusters_k3.pkl"
    local = load_seasonal_reference(source, include_patterns=("tresas",))
    assert local.N_Campanas.eq(1).all()
    assert local.Campanas.eq("test -emerel tresas 2025.xlsx").all()
    with pytest.raises(ValueError, match="no contiene campañas"):
        load_seasonal_reference(source, include_patterns=("balcarce", "san pedro"))
