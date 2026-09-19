from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem
from predweem_twin.observations import prepare_observations, read_observation_file
from predweem_twin.storage import TwinStore


ROOT = Path(__file__).parents[1]


def real_trajectory():
    weather = pd.read_csv(ROOT / "meteo_daily.csv")
    model = PracticalANNModel.from_directory(ROOT / "models")
    return run_predweem(weather, model, ModelParameters())


def test_plm2_flow_is_converted_to_auditable_cumulative_fraction():
    raw = pd.DataFrame(
        {
            "FECHA": ["2026-03-02", "2026-03-16", "2026-03-30"],
            "PLM2": [10.0, 30.0, 20.0],
        }
    )
    prepared, metadata = prepare_observations(raw, real_trajectory(), mode="auto")
    assert metadata["modo"] == "flujo"
    assert metadata["total_observado_plm2"] == 60.0
    assert prepared["Observado"].is_monotonic_increasing
    assert prepared["Observado"].between(0, 1).all()
    assert prepared.iloc[-1]["Acumulado_PLM2"] == 60.0
    assert prepared["Unidad_original"].eq("plantas/m² por intervalo").all()


def test_cumulative_percent_is_detected_and_scaled():
    raw = pd.DataFrame(
        {
            "Fecha": ["2026-03-02", "2026-03-16", "2026-03-30"],
            "EMERGENCIA_ACUMULADA": [12.0, 45.0, 70.0],
        }
    )
    prepared, metadata = prepare_observations(raw, real_trajectory(), mode="auto")
    assert metadata["modo"] == "acumulado"
    assert np.allclose(prepared["Observado"], [0.12, 0.45, 0.70])


def test_excel_reader_finds_fecha_plm2_sheet():
    payload = BytesIO()
    source = pd.DataFrame({"FECHA": [pd.Timestamp("2026-03-02")], "PLM2": [15.0]})
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        source.to_excel(writer, sheet_name="Campo", index=False)
    payload.name = "campo.xlsx"
    frame, metadata = read_observation_file(payload)
    assert list(frame.columns) == ["FECHA", "PLM2"]
    assert metadata == {"archivo": "campo.xlsx", "hoja": "Campo"}


def test_three_repetitions_and_mean_per_m2_are_detected():
    payload = BytesIO()
    source = pd.DataFrame(
        {
            "Fecha": [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-16")],
            1: [10.0, 30.0],
            2: [12.0, 24.0],
            3: [8.0, 36.0],
            "media(SR).m2": [40.0, 120.0],
        }
    )
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        source.to_excel(writer, sheet_name="Hoja1", index=False)
    payload.name = "tres_repeticiones.xlsx"

    frame, _ = read_observation_file(payload)
    prepared, metadata = prepare_observations(frame, real_trajectory(), mode="auto")

    assert metadata["modo"] == "flujo"
    assert metadata["n_repeticiones"] == 3
    assert np.isclose(metadata["factor_conversion_repeticiones"], 4.0)
    assert np.isclose(metadata["area_cuadrante_inferida_m2"], 0.25)
    assert prepared["Valor_original"].tolist() == [40.0, 120.0]
    assert prepared["Incertidumbre"].between(0.05, 0.30).all()
    assert "error estándar del flujo" in metadata["metodo_incertidumbre"]


def test_bulk_storage_preserves_original_values(tmp_path):
    raw = pd.DataFrame(
        {"FECHA": ["2026-03-02", "2026-03-16"], "PLM2": [12.0, 30.0]}
    )
    prepared, _ = prepare_observations(raw, real_trajectory(), mode="flujo")
    store = TwinStore(tmp_path / "twin.db")
    store.upsert_observations("Lote-prueba", prepared)
    stored = store.observations("Lote-prueba")
    assert len(stored) == 2
    assert stored["Valor_original"].tolist() == [12.0, 30.0]
    assert stored["Unidad_original"].eq("plantas/m² por intervalo").all()
    assert stored["Fuente"].eq("Archivo cargado").all()


def test_bulk_storage_preserves_repetitions(tmp_path):
    raw = pd.DataFrame(
        {
            "FECHA": ["2026-03-02", "2026-03-16"],
            1: [10.0, 30.0],
            2: [12.0, 24.0],
            3: [8.0, 36.0],
            "media(SR).m2": [40.0, 120.0],
        }
    )
    prepared, _ = prepare_observations(raw, real_trajectory(), mode="flujo")
    store = TwinStore(tmp_path / "twin.db")
    store.upsert_observations("Lote-repeticiones", prepared)
    stored = store.observations("Lote-repeticiones")
    assert stored["Repeticiones_originales"].tolist() == [
        "[10.0, 12.0, 8.0]",
        "[30.0, 24.0, 36.0]",
    ]
    assert stored["Factor_conversion_repeticiones"].eq(4.0).all()
    assert stored["EE_repeticiones_PLM2"].notna().all()
    assert stored["Modo"].eq("flujo").all()
    assert stored["Flujo_observado_PLM2"].tolist() == [40.0, 120.0]
    assert stored["Acumulado_PLM2"].tolist() == [40.0, 160.0]


def test_delete_observations_only_removes_selected_dates_from_active_site(tmp_path):
    raw = pd.DataFrame(
        {"FECHA": ["2026-03-02", "2026-03-16"], "PLM2": [12.0, 30.0]}
    )
    prepared, _ = prepare_observations(raw, real_trajectory(), mode="flujo")
    store = TwinStore(tmp_path / "twin.db")
    store.upsert_observations("Lote-A", prepared)
    store.upsert_observations("Lote-B", prepared)

    deleted = store.delete_observations("Lote-A", ["2026-03-02"])

    assert deleted == 1
    assert store.observations("Lote-A")["Fecha"].dt.date.astype(str).tolist() == [
        "2026-03-16"
    ]
    assert len(store.observations("Lote-B")) == 2
    assert store.delete_observations("Lote-A", []) == 0
