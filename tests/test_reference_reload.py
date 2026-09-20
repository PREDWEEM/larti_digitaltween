"""Regresión de la referencia antigua conservada durante una actualización."""

import ast
from pathlib import Path

import streamlit as st

from predweem_twin.seasonal import load_seasonal_reference


ROOT = Path(__file__).parents[1]


def test_app_reloads_reference_after_loader_changes():
    current = load_seasonal_reference(ROOT / "models/modelo_clusters_k3.pkl")
    legacy = current.drop(columns=["Campanas", "Campanas_Excluidas"]).copy()
    legacy["N_Campanas"] = 11
    returned = {"frame": legacy, "calls": 0}

    def changing_loader(path, **kwargs):
        returned["calls"] += 1
        return returned["frame"].copy()

    # Ejecutar el cargador real de app.py sin abrir la interfaz completa.
    # Una caché sin argumentos ignora los cambios en la función importada.
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_progress_reference")
    namespace = {
        "__name__": "reference_reload_regression", "BASE": ROOT,
        "st": st, "load_seasonal_reference": changing_loader,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "app.py"), "exec"), namespace)
    loader = namespace["load_progress_reference"]
    try:
        assert "Campanas" not in loader()
        returned["frame"] = current
        updated = loader()
        assert returned["calls"] == 2
        assert {"Campanas", "Campanas_Excluidas"}.issubset(updated.columns)
        assert updated.N_Campanas.eq(9).all()
        assert not updated.Campanas.str.contains("balcarce|san pedro", case=False).any()
    finally:
        if hasattr(loader, "clear"):
            loader.clear()
