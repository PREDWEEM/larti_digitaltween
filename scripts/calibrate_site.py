"""Genera un perfil por sitio y diagnósticos reproducibles sin entrenar la ANN.

Desde la raíz: python scripts/calibrate_site.py
Los CSV de entrada son copias fijas; no se consulta meteorología en vivo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predweem_twin.calibration import (  # noqa: E402
    calibrated_progress, fit_site_calibration, model_fingerprint,
)
from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem  # noqa: E402
from predweem_twin.observations import prepare_observations, read_observation_file  # noqa: E402
from predweem_twin.seasonal import EXCLUDED_SITES, load_seasonal_reference  # noqa: E402


def build_calibration(observations_path, weather_path, output_path, site="Lartigau",
                      coverage=75.0, w_max=18.816, source_metadata_path=None):
    observations_path, weather_path, output_path = map(
        Path, (observations_path, weather_path, output_path)
    )
    source_metadata = {}
    if source_metadata_path:
        source_metadata = json.loads(Path(source_metadata_path).read_text(encoding="utf-8"))
    raw, metadata = read_observation_file(observations_path)
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    date_column = next((column for column in raw if str(column).strip().lower() == "fecha"), None)
    if date_column is None:
        raise ValueError("El archivo requiere la columna FECHA.")
    sample_dates = pd.to_datetime(raw[date_column], errors="raise")
    if sample_dates.isna().any() or sample_dates.duplicated().any():
        raise ValueError("Las fechas de muestreo deben ser válidas y únicas.")
    weather = pd.read_csv(weather_path)
    weather["Fecha"] = pd.to_datetime(weather["Fecha"], errors="raise")
    last_count = sample_dates.max()
    weather = weather.loc[weather["Fecha"] <= last_count].copy()
    if "TipoDato" in weather and weather["TipoDato"].eq("Pronostico").any():
        raise ValueError("La calibración histórica no admite filas de pronóstico.")
    model = PracticalANNModel.from_directory(ROOT / "models")
    reference = load_seasonal_reference(
        ROOT / "models/modelo_clusters_k3.pkl", excluded_years=("2010", "2015"),
    )
    parameters = ModelParameters(cobertura_pct=coverage, w_max=w_max)

    def simulate(cutoff, end=None):
        return run_predweem(
            weather.loc[weather["Fecha"] <= (end if end is not None else cutoff)],
            model, parameters, normalization_as_of=cutoff, seasonal_reference=reference,
        )

    trajectory = simulate(last_count)
    prepared, import_metadata = prepare_observations(
        raw, trajectory, source_name=observations_path.name
    )
    profile, comparison = fit_site_calibration(trajectory, prepared, site=site)

    # Mantener los cortes de una revisión previa cuando están registrados,
    # aunque se agregue una fecha inicial. Sólo conteos y meteorología hasta el
    # corte entran al ajuste. Para evaluar el intervalo siguiente se usa su
    # meteorología realizada: es un hindcast condicional, no pronóstico archivado.
    holdout_rows = []
    validation_cutoffs = source_metadata.get("validation_cutoffs")
    if validation_cutoffs:
        counts = []
        for value in validation_cutoffs:
            cutoff = pd.Timestamp(value)
            matches = prepared.index[prepared["Fecha"].eq(cutoff)].tolist()
            if not matches or matches[0] < 6 or matches[0] + 1 >= len(prepared):
                raise ValueError(f"Corte de evaluación no disponible: {value}")
            counts.append(matches[0] + 1)
    else:
        counts = range(7, len(prepared))
    for count in counts:
        training = prepared.iloc[:count].copy()
        cutoff = pd.Timestamp(training["Fecha"].iloc[-1])
        target = prepared.iloc[count]
        target_date = pd.Timestamp(target["Fecha"])
        fitted, _ = fit_site_calibration(simulate(cutoff), training, site=site)
        evaluation = simulate(cutoff, end=target_date).set_index("Fecha")
        endpoints = evaluation.loc[[cutoff, target_date], "EMERAC_NORMALIZADA"].to_numpy(float)
        base = float(np.diff(endpoints)[0] * fitted["fit"]["nuisance_scale_base"])
        calibrated = float(np.diff(calibrated_progress(
            endpoints, **fitted["parameters"]
        ))[0] * fitted["fit"]["nuisance_scale_calibrated"])
        holdout_rows.append({
            "Corte_entrenamiento": cutoff.date().isoformat(),
            "Fecha_evaluacion": target_date.date().isoformat(),
            "Dias_intervalo": int((target_date - cutoff).days),
            "N_muestreos_ajuste": count,
            "Observado_PLM2": float(target["Flujo_observado_PLM2"]),
            "Base_PLM2": base,
            "Calibrado_PLM2": calibrated,
            "Offset": fitted["parameters"]["offset"],
            "Slope": fitted["parameters"]["slope"],
        })
    holdout = pd.DataFrame(holdout_rows)
    validation = {
        "kind": "evaluacion_temporal_condicional_con_meteobahia_archivada",
        "independent_season": False,
        "n_intervals": len(holdout),
        "note": (
            "Ajuste con datos hasta cada corte y evaluación del siguiente intervalo. "
            "Se utiliza la serie histórica de pronósticos MeteoBahía archivados, "
            "sin garantizar la emisión disponible en cada corte. No demuestra "
            "transferencia a otra campaña ni precisión operativa a siete días."
        ),
    }
    if not holdout.empty:
        y = holdout["Observado_PLM2"].to_numpy()
        validation.update({
            "rmse_base_plm2": float(np.sqrt(np.mean((holdout["Base_PLM2"].to_numpy() - y)**2))),
            "rmse_calibrated_plm2": float(np.sqrt(np.mean((holdout["Calibrado_PLM2"].to_numpy() - y)**2))),
            "intervals_improved": int((
                (holdout["Calibrado_PLM2"] - y).abs()
                < (holdout["Base_PLM2"] - y).abs()
            ).sum()),
        })
    first_date = pd.Timestamp(prepared["Fecha"].iloc[0]).date().isoformat()
    initial_zero = bool(prepared["Flujo_observado_PLM2"].iloc[0] == 0)
    initial_note = (
        f"El registro inicial de cero del {first_date} delimita el primer intervalo; "
        "no se infiere ausencia de emergencia en fechas anteriores."
        if initial_zero else
        "Primer conteo conservado pero excluido del ajuste: inicio del intervalo desconocido."
    )
    profile.update({
        "observations_start": first_date,
        "initial_zero_reference": initial_zero,
        "model_fingerprint": model_fingerprint(ROOT),
        "model_parameters": asdict(parameters),
        "seasonal_reference": {
            "include_patterns": [],
            "excluded_years": ["2010", "2015"],
            "excluded_sites": list(EXCLUDED_SITES),
            "excluded_campaigns": reference["Campanas_Excluidas"].iloc[0],
            "scope": "Referencia compartida; sin campaña histórica identificada como Lartigau",
            "n_campaigns": int(reference["N_Campanas"].iloc[0]),
            "campaigns": reference["Campanas"].iloc[0],
        },
        "source": {
            **source_metadata,
            "observations_file": observations_path.name,
            "observations_sha256": sha256(observations_path.read_bytes()).hexdigest(),
            "weather_file": weather_path.name,
            "weather_sha256": sha256(weather_path.read_bytes()).hexdigest(),
            "weather_types": weather["TipoDato"].value_counts().to_dict() if "TipoDato" in weather else {},
            "observed_total_plm2": float(prepared["Flujo_observado_PLM2"].sum()),
            "replicate_count": import_metadata.get("n_repeticiones"),
            "replicate_conversion_factor": import_metadata.get("factor_conversion_repeticiones"),
        },
        "validation": validation,
        "limitations": [
            "Una sola campaña incompleta. No se estima ni transfiere un total estacional.",
            initial_note,
            f"Cobertura de {coverage:g} % y Wmax de {w_max:g} mm son supuestos de la configuración operativa; el archivo no informa manejo ni cobertura.",
            "El archivo FECHA + PLM2 no incluye repeticiones. Se utiliza un piso de ponderación común, no un error de muestreo medido.",
            "Se conserva el techo del 50 % y decaimiento desde el 15/04 del motor Lartigau. No se incorpora extinción post-pico de otra localidad.",
            "La referencia estacional es compartida, no una validación histórica local de Lartigau.",
            "La meteorología corresponde a pronósticos MeteoBahía archivados, no a observaciones de estación.",
            "La transformación no crea cohortes en fechas bloqueadas por el motor biofísico.",
            "Un parámetro en su límite indica que persisten diferencias estructurales.",
            "El ajuste no reduce automáticamente la incertidumbre de asimilación.",
            "Los gráficos de la campaña de ajuste son retrospectivos, no predicciones independientes.",
        ],
    })
    # Cambiar la identidad aun si una revisión conserva la última fecha.
    input_signature = sha256(json.dumps({
        "observations": profile["source"]["observations_sha256"],
        "weather": profile["source"]["weather_sha256"],
        "model": profile["model_fingerprint"],
        "parameters": profile["model_parameters"],
        "seasonal_reference": profile["seasonal_reference"],
        "method": profile["method"],
        "calibration_parameters": profile["parameters"],
    }, sort_keys=True).encode()).hexdigest()
    profile["profile_id"] += "-" + input_signature[:10]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    comparison.to_csv(output_path.with_name(output_path.stem + "_fit.csv"), index=False, date_format="%Y-%m-%d")
    holdout.to_csv(output_path.with_name(output_path.stem + "_holdout.csv"), index=False)
    print(json.dumps({"profile": profile["profile_id"], "fit": profile["fit"], "validation": validation}, indent=2))
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=ROOT / "data/calibration/lartigau_2026_counts.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "data/calibration/lartigau_2026_weather.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/calibration/lartigau_2026.json")
    parser.add_argument("--source-metadata", type=Path, default=ROOT / "data/calibration/lartigau_2026_source.json")
    parser.add_argument("--site", default="Lartigau")
    parser.add_argument("--coverage", type=float, default=75.0)
    parser.add_argument("--w-max", type=float, default=18.816)
    args = parser.parse_args()
    build_calibration(args.observations, args.weather, args.output, args.site,
                      args.coverage, args.w_max, args.source_metadata)


if __name__ == "__main__":
    main()
