"""Límites del semáforo térmico e integración con el estado de la fecha elegida."""

import numpy as np
import pandas as pd
import pytest

from predweem_twin.state import build_twin_snapshot, thermal_control_stage


@pytest.mark.parametrize("tt,stage", [
    (0., "AUN NO CONTROLAR"),
    (599.99, "AUN NO CONTROLAR"),
    (600., "CONTROL A TIEMPO"),
    (600.01, "CONTROL A TIEMPO"),
    (700., "CONTROL A TIEMPO"),
    (700.01, "ULTIMO PLAZO"),
    (800., "ULTIMO PLAZO"),
    (800.01, "FUERA DE CONTROL"),
    (1200., "FUERA DE CONTROL"),
    (np.nan, "SIN DATOS"),
    (np.inf, "SIN DATOS"),
])
def test_thermal_stages_keep_exact_boundaries_without_rounding(tt, stage):
    assert thermal_control_stage(tt) == stage


def test_snapshot_uses_thermal_time_at_cutoff_not_end_of_forecast():
    frame = pd.DataFrame({
        "Fecha": pd.date_range("2027-05-05", periods=8),
        "TT_DESDE_PICO": [700., 710., 730., 750., 770., 790., 810., 830.],
        "EMERAC_TWIN": .5, "EMERREL_TWIN": 0.,
        "W_superficial": 10., "Humedad_Relativa": .5, "Termoinhibida": False,
    })
    snapshot = build_twin_snapshot(frame, "test", "2027-05-05", "test")
    assert snapshot["thermal_time"] == 700.
    assert snapshot["thermal_control_stage"] == "CONTROL A TIEMPO"
