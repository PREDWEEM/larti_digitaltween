"""Una app actualizada no debe combinar UI nueva con módulos antiguos en memoria."""

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from predweem_twin import assimilation, core, state, storage


ROOT = Path(__file__).parents[1]


def test_hot_update_reloads_snapshot_motor_assimilation_and_store(monkeypatch, tmp_path):
    shutil.copy2(ROOT / "app.py", tmp_path / "app.py")
    shutil.copy2(ROOT / "meteo_daily.csv", tmp_path / "meteo_daily.csv")
    for directory in ("models", "predweem_twin", "data"):
        shutil.copytree(ROOT / directory, tmp_path / directory,
                        ignore=shutil.ignore_patterns("*.db", "*.db-shm", "*.db-wal", "__pycache__"))

    # Una actualización despliega archivos nuevos sin vaciar sys.modules.
    # La versión anterior del constructor no producía normalization_available.
    current_builder = state.build_twin_snapshot

    def old_snapshot(*args, **kwargs):
        snapshot = current_builder(*args, **kwargs)
        snapshot.pop("normalization_available")
        return snapshot

    def stale_component(*args, **kwargs):
        raise AssertionError("Se ejecutó un componente de la versión anterior")

    st.cache_resource.clear()
    monkeypatch.setattr(state, "build_twin_snapshot", old_snapshot)
    monkeypatch.setattr(core, "run_predweem", stale_component)
    monkeypatch.setattr(assimilation, "assimilate_observations", stale_component)
    monkeypatch.setattr(storage, "TwinStore", stale_component)

    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=45).run()
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]
    assert state.build_twin_snapshot is not old_snapshot
    assert core.run_predweem is not stale_component
    assert assimilation.assimilate_observations is not stale_component
    assert storage.TwinStore is not stale_component
    current_motor = core.run_predweem

    for cutoff, available in [("2026-01-10", False), ("2026-09-01", True)]:
        app.date_input[0].set_value(pd.Timestamp(cutoff).date()).run()
        assert not app.exception, [error.message for error in app.exception]
        assert not app.error, [error.value for error in app.error]
        metrics = {item.label: item.value for item in app.metric}
        assert (metrics["Emergencia estimada"] != "Aún no estimable") == available
        assert (metrics["Emergencia remanente"] != "Aún no estimable") == available
        assert "nan" not in str(metrics).lower()
        if not available:
            assert "Aún no estimable" in metrics["Intensidad de emergencia · 7 días"]
        # Cambiar la fecha no vuelve a recargar clases ni recursos.
        assert core.run_predweem is current_motor

    # Una segunda modificación del paquete sí debe invalidar la carga anterior.
    with (tmp_path / "predweem_twin/state.py").open("a") as stream:
        stream.write("\n# Simulación de una nueva revisión durante la sesión.\n")
    monkeypatch.setattr(state, "build_twin_snapshot", old_snapshot)
    app.run()
    assert not app.exception, [error.message for error in app.exception]
    assert state.build_twin_snapshot is not old_snapshot

