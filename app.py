"""Interfaz PREDWEEM Digital Twin — Lartigau."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from predweem_twin.assimilation import assimilate_observations
from predweem_twin.calibration import (
    apply_site_calibration, load_site_profile, model_fingerprint,
)
from predweem_twin.coverage import (
    has_coverage_data,
    prepare_coverage_series,
    read_coverage_file,
)
from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem
from predweem_twin.observations import prepare_observations, read_observation_file
from predweem_twin.scenarios import apply_scenario
from predweem_twin.seasonal import load_seasonal_reference
from predweem_twin.state import (
    build_twin_snapshot,
    milestone_dates,
    thermal_window_dates,
)
from predweem_twin.storage import TwinStore
from predweem_twin.weather import (
    fetch_open_meteo,
    last_observed_weather_date,
    operational_weather_window,
    read_weather_file,
    weather_source_label,
)


BASE = Path(__file__).parent
CALIBRATION_DIR = BASE / "data" / "calibration"
st.set_page_config(page_title="PREDWEEM Digital Twin", page_icon="🌱", layout="wide")

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg,#f5f8f3 0%,#eef3ed 100%);}
      [data-testid="stSidebar"] {background:#11291f;}
      [data-testid="stSidebar"] * {color:#f5faf7;}
      div[data-testid="stMetric"] {background:white;border:1px solid #dfe8e1;
        border-radius:16px;padding:18px;box-shadow:0 8px 22px rgba(20,50,35,.06)}
      .hero {padding:22px 26px;border-radius:20px;color:white;margin-bottom:18px;
        background:linear-gradient(120deg,#173f2e,#287653 68%,#74a45b);}
      .eyebrow {letter-spacing:.14em;text-transform:uppercase;font-size:.76rem;opacity:.82}
      .hero h1 {margin:.15rem 0 .25rem;font-size:2.15rem}
      .hero p {margin:0;max-width:900px;opacity:.9}
      .status-pill {display:inline-block;padding:5px 11px;border-radius:999px;
        background:#d9f1dd;color:#174a2d;font-weight:700;font-size:.8rem}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return PracticalANNModel.from_directory(BASE / "models")


def load_progress_reference():
    """Recarga la referencia vigente; evita curvas o columnas obsoletas en caché."""
    return load_seasonal_reference(
        BASE / "models" / "modelo_clusters_k3.pkl",
        excluded_years=("2010", "2015"),
    )


def load_store():
    """Crea el acceso SQLite con el esquema vigente.

    No se almacena en cache_resource: durante una actualización en caliente,
    Streamlit podría conservar una instancia creada con una versión anterior
    de TwinStore y ocultar métodos o migraciones recién incorporados.
    """
    return TwinStore(BASE / "data" / "twin_state.db")


@st.cache_data(ttl=3600, show_spinner=False)
def load_open_meteo(latitude, longitude, start_date):
    return fetch_open_meteo(latitude, longitude, start_date)


def trajectory_chart(
    df,
    observations,
    as_of,
    audit=None,
    lower_thermal_time=600.0,
    upper_thermal_time=800.0,
):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=df["Fecha"],
            y=df["EMERREL_TWIN"] * 100,
            name="Flujo diario Twin",
            marker_color="#3b82f6",
            opacity=0.62,
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=df["Fecha"],
            y=df.get("EMERAC_BASE_SIN_CALIBRAR", df["EMERAC_NORMALIZADA"]) * 100,
            name="PREDWEEM base",
            line=dict(color="#83938b", width=2, dash="dot"),
        ),
        secondary_y=False,
    )
    if "Calibracion_Aplicada" in df and df["Calibracion_Aplicada"].any():
        figure.add_trace(
            go.Scatter(
                x=df["Fecha"], y=df["EMERAC_CALIBRADA"] * 100,
                name="Calibración Lartigau", line=dict(color="#9260bd", width=2),
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=df["Fecha"],
            y=df["EMERAC_TWIN"] * 100,
            name="Estado actualizado",
            line=dict(color="#155d3e", width=4),
            fill="tozeroy",
            fillcolor="rgba(66,137,87,.10)",
        ),
        secondary_y=False,
    )
    if audit is not None and not audit.empty and "Estado_campo_estimado" in audit:
        figure.add_trace(
            go.Scatter(
                x=audit["Fecha_asimilada"],
                y=audit["Estado_campo_estimado"] * 100,
                name="Estado estimado desde campo",
                mode="markers",
                marker=dict(color="#df5b3f", size=11, line=dict(color="white", width=2)),
            ),
            secondary_y=False,
        )
    elif observations is not None and not observations.empty:
        figure.add_trace(
            go.Scatter(
                x=observations["Fecha"],
                y=observations["Observado"] * 100,
                name="Conteo de campo",
                mode="markers",
                marker=dict(color="#df5b3f", size=11, line=dict(color="white", width=2)),
            ),
            secondary_y=False,
        )
    thermal_start, thermal_end = thermal_window_dates(
        df, lower_thermal_time, upper_thermal_time
    )
    if thermal_start is not None:
        displayed_thermal_end = thermal_end or pd.Timestamp(df["Fecha"].max())
        figure.add_vrect(
            x0=thermal_start,
            x1=displayed_thermal_end,
            fillcolor="rgba(255,193,7,.22)",
            line_width=0,
            annotation_text=(
                f"Ventana fenológica {lower_thermal_time:.0f}–"
                f"{upper_thermal_time:.0f} °Cd"
            ),
            annotation_position="top right",
            annotation_font_color="#6f5200",
        )
        figure.add_vline(
            x=thermal_start.timestamp() * 1000,
            line_color="#c48a00",
            line_dash="dot",
            line_width=1.5,
        )
        if thermal_end is not None:
            figure.add_vline(
                x=thermal_end.timestamp() * 1000,
                line_color="#c48a00",
                line_dash="dot",
                line_width=1.5,
            )
    figure.add_vline(x=pd.Timestamp(as_of).timestamp() * 1000, line_color="#162f25", line_dash="dash")
    forecast_start = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    if pd.Timestamp(df["Fecha"].max()) >= forecast_start:
        figure.add_vrect(
            x0=forecast_start,
            x1=pd.Timestamp(df["Fecha"].max()),
            fillcolor="rgba(223,127,52,.10)",
            line_width=0,
            annotation_text="Pronóstico 7 días",
            annotation_position="top left",
        )
    figure.update_yaxes(title_text="Emergencia acumulada (%)", range=[0, 105], secondary_y=False)
    figure.update_yaxes(title_text="Flujo diario (%)", rangemode="tozero", secondary_y=True)
    figure.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.15,
    )
    return figure


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">PREDWEEM by Guillermo R. Chantre</div>
      <h1>Gemelo Digital · Lolium Lartigau</h1>
      <p>Estado vivo del lote, actualizado con meteorología y observaciones de campo,
      con calibración local 2026, proyección a 7 días y simulación de escenarios.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Configuración del gemelo")
    site_id = st.text_input("Identificador del lote", "Lartigau-01")
    calibration_site = st.selectbox("Localidad del lote", ["Lartigau", "Otra localidad"])
    latitude = st.number_input("Latitud", value=-38.6166, format="%.6f")
    longitude = st.number_input("Longitud", value=-61.7000, format="%.6f")
    source_option = st.radio(
        "Meteorología",
        ["MeteoBahía · Coronel Falcón", "Open-Meteo", "Cargar archivo"],
    )
    uploaded_weather = None
    if source_option == "Cargar archivo":
        uploaded_weather = st.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])
    coverage_mode = st.radio(
        "Cobertura de rastrojo",
        ["Constante", "Serie observada"],
        help=(
            "La serie observada se carga por lote con las columnas "
            "FECHA + COBERTURA_PCT."
        ),
    )
    coverage = st.slider(
        "Cobertura constante o de respaldo (%)", 0, 100, 75, 5
    )
    w_max = st.number_input(
        "Agua superficial Wmax (mm)", min_value=5.0, max_value=60.0,
        value=18.816, step=0.1, format="%.3f",
    )
    model_uncertainty = st.slider("Incertidumbre del modelo", 0.03, 0.30, 0.12, 0.01)
    seasonal_potential_input = st.number_input(
        "Potencial estacional previo (plantas/m²)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help=(
            "Use 0 para estimación automática. Ingrese un valor histórico "
            "del lote si está disponible."
        ),
    )
    seasonal_potential_prior = (
        float(seasonal_potential_input) if seasonal_potential_input > 0 else None
    )
    st.markdown("**Perfil fisiológico Lartigau**")
    st.caption(
        "Latencia JD 25 · desde el 15/04, techo del 50 % del máximo previo "
        "con decaimiento τ=60 días, β=1 e intensidad 0,75."
    )
    st.caption("La asimilación modifica el estado estimado, no recalibra la ANN.")

try:
    if source_option == "Open-Meteo":
        weather = load_open_meteo(latitude, longitude, f"{date.today().year}-01-01")
    elif source_option == "Cargar archivo" and uploaded_weather is not None:
        weather = read_weather_file(uploaded_weather)
    else:
        weather = read_weather_file(BASE / "meteo_daily.csv")
except Exception as error:
    st.error(f"No fue posible cargar la meteorología: {error}")
    st.stop()

weather_date_column = "Fecha" if "Fecha" in weather else "FECHA"
weather_dates = pd.to_datetime(weather[weather_date_column], errors="coerce")
min_date = weather_dates.min().date()
max_date = weather_dates.max().date()
last_observed_date = pd.Timestamp(last_observed_weather_date(weather)).date()
max_state_date = min(last_observed_date, max_date)
default_date = min(max(date.today(), min_date), max_state_date)
as_of = st.sidebar.date_input(
    "Fecha del estado",
    value=default_date,
    min_value=min_date,
    max_value=max_state_date,
)
weather, forecast_metadata = operational_weather_window(
    weather, as_of=as_of, forecast_days=7
)
weather_dates = pd.to_datetime(weather[weather_date_column], errors="coerce")

parameters = ModelParameters(
    cobertura_pct=float(coverage),
    w_max=float(w_max),
    latitud=float(latitude),
    longitud=float(longitude),
)
model = load_model()
seasonal_reference = load_progress_reference()
reference_campaigns = int(seasonal_reference["N_Campanas"].iloc[0])
store = load_store()
coverage_observations = store.coverage_observations(site_id)
active_coverage = coverage_observations[
    pd.to_datetime(coverage_observations["Fecha"], errors="coerce")
    <= pd.Timestamp(as_of)
].copy()
coverage_series_for_model = (
    active_coverage
    if coverage_mode == "Serie observada" and not active_coverage.empty
    else None
)
if coverage_mode == "Serie observada" and active_coverage.empty:
    st.sidebar.warning(
        "No hay cobertura observada disponible hasta esta fecha. "
        "Se utiliza el valor de respaldo."
    )
base_trajectory = run_predweem(
    weather,
    model,
    parameters,
    coverage_series=coverage_series_for_model,
    normalization_as_of=as_of,
    seasonal_reference=seasonal_reference,
)
coverage_at_cutoff = float(
    base_trajectory.loc[
        base_trajectory["Fecha"] <= pd.Timestamp(as_of), "Cobertura_Rastrojo"
    ].iloc[-1]
)
observations = store.observations(site_id)
active_observations = observations[
    (pd.to_datetime(observations["Fecha"], errors="coerce") <= pd.Timestamp(as_of))
    & (pd.to_datetime(observations["Fecha"], errors="coerce").dt.year == pd.Timestamp(as_of).year)
].copy()
try:
    calibration_profile = load_site_profile(CALIBRATION_DIR / "lartigau_2026.json")
except (ValueError, KeyError, TypeError) as error:
    calibration_profile = None
    st.warning(f"No se pudo cargar el perfil de calibración: {error}")
current_model_fingerprint = model_fingerprint(BASE)
# El interruptor del gráfico actualiza su estado antes de cada recálculo.
calibration_enabled = st.session_state.get("local_calibration_enabled", True)
calibrated_trajectory, calibration_audit = apply_site_calibration(
    base_trajectory, calibration_profile,
    site=calibration_site, as_of=as_of, enabled=calibration_enabled,
    observations=active_observations, model_fingerprint=current_model_fingerprint,
)
twin_trajectory, assimilation_audit = assimilate_observations(
    calibrated_trajectory,
    active_observations,
    model_uncertainty=model_uncertainty,
    seasonal_potential_prior=seasonal_potential_prior,
)
source_label = weather_source_label(weather)
snapshot = build_twin_snapshot(
    twin_trajectory,
    site_id,
    as_of,
    source_label,
    len(assimilation_audit),
)
snapshot["calibration"] = calibration_audit
milestones = milestone_dates(twin_trajectory)

st.markdown(
    f'<span class="status-pill">● ACTUALIZADO AL {pd.Timestamp(snapshot["as_of"]).strftime("%d/%m/%Y")}</span>',
    unsafe_allow_html=True,
)
forecast_end_label = (
    pd.Timestamp(forecast_metadata["forecast_end"]).strftime("%d/%m/%Y")
    if forecast_metadata["forecast_end"] is not None
    else "sin pronóstico"
)
st.caption(
    "Campaña meteorológica cerrada al 01/10/2026."
    if forecast_metadata["campaign_closed"] else
    f'Meteorología histórica hasta **{pd.Timestamp(as_of).strftime("%d/%m/%Y")}** · '
    f'pronóstico disponible: **{forecast_metadata["forecast_days_available"]}/{forecast_metadata["forecast_days_expected"]} días** '
    f'(hasta {forecast_end_label}).'
)
if source_option == "MeteoBahía · Coronel Falcón":
    st.caption(
        "El histórico MeteoBahía contiene pronósticos archivados de Coronel Falcón; "
        "no son observaciones meteorológicas de estación."
    )
st.caption(
    f"Referencia estacional compartida: {reference_campaigns} campañas del clasificador "
    "original, excluyendo 2010, 2015, Balcarce y San Pedro. No hay una serie histórica identificada "
    "como Lartigau; la calibración local utiliza los conteos 2026."
)
st.caption(
    "La selección conserva ocho curvas identificadas por año y Tres Arroyos 2025. "
    "Los nombres utilizados y excluidos se muestran en Trazabilidad."
)
if not forecast_metadata["complete"]:
    st.warning(
        "El horizonte meteorológico está incompleto. Los indicadores futuros "
        "se calculan solamente con los días disponibles."
    )
if snapshot["last_observation_date"]:
    st.caption(
        "Estado actualizado con flujos de campo hasta "
        f'**{pd.Timestamp(snapshot["last_observation_date"]).strftime("%d/%m/%Y")}**. '
        "Desde esa fecha se proyecta con el flujo diario de PREDWEEM."
    )
if coverage_series_for_model is not None:
    last_coverage = active_coverage.iloc[-1]
    st.caption(
        "Cobertura variable activa con mediciones hasta "
        f'**{pd.Timestamp(last_coverage["Fecha"]).strftime("%d/%m/%Y")}**; '
        "los días intermedios se interpolan y luego se mantiene el último valor."
    )

metric_columns = st.columns(5)
metric_columns[0].metric("Emergencia estimada", f'{snapshot["emergence"]:.0%}')
metric_columns[1].metric("Emergencia remanente", f'{snapshot["remaining"]:.0%}')
metric_columns[2].metric("Riesgo próximos 7 días", snapshot["risk_7d"], f'+{snapshot["increment_7d"]:.1%}')
metric_columns[3].metric("Agua superficial", f'{snapshot["soil_water"]:.1f} mm', f'{snapshot["soil_water_fraction"]:.0%} Wmax')
metric_columns[4].metric("TT desde primer pico", f'{snapshot["thermal_time"]:.0f} °Cd', f'{parameters.tt_limite:.0f} °Cd límite')

tab_state, tab_observations, tab_calibration, tab_scenarios, tab_audit = st.tabs(
    ["Estado del lote", "Observaciones", "Calibración por sitio", "Escenarios", "Trazabilidad"]
)

with tab_state:
    if snapshot["seasonal_potential_plm2"] is not None:
        field_metrics = st.columns(3)
        field_metrics[0].metric(
            "Acumulado estimado",
            f'{snapshot["emergence_density_plm2"]:.1f} plantas/m²',
        )
        field_metrics[1].metric(
            "Potencial estacional estimado",
            f'{snapshot["seasonal_potential_plm2"]:.1f} plantas/m²',
        )
        field_metrics[2].metric(
            "Modo de actualización",
            snapshot["assimilation_mode"].capitalize(),
        )
    st.toggle(
        "Usar calibración local 2026",
        value=calibration_enabled,
        key="local_calibration_enabled",
        help=(
            "Activada por defecto. Desactívela para comparar con PREDWEEM base. "
            "El perfil de Lartigau es experimental; su aplicación depende de la "
            "localidad, la fecha y las observaciones asimiladas."
        ),
    )
    st.caption(calibration_audit["reason"])
    st.plotly_chart(
        trajectory_chart(
            twin_trajectory,
            active_observations,
            as_of,
            assimilation_audit,
            parameters.tt_control,
            parameters.tt_limite,
        ),
        width="stretch",
    )
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Lectura agronómica")
        if snapshot["next_cohort_start"]:
            cohort = (
                f'{pd.Timestamp(snapshot["next_cohort_start"]).strftime("%d/%m")}–'
                f'{pd.Timestamp(snapshot["next_cohort_end"]).strftime("%d/%m")}'
            )
        else:
            cohort = "No detectada en el horizonte"
        st.write(
            f'El gemelo estima **{snapshot["emergence"]:.0%}** de la emergencia potencial y '
            f'**{snapshot["remaining"]:.0%}** remanente. La próxima cohorte probable es **{cohort}**. '
            f'La termoinhibición está **{"activa" if snapshot["thermoinhibited"] else "inactiva"}**.'
        )
        st.info(
            "La salida es soporte para decisión. Debe interpretarse junto con el monitoreo "
            "del lote y el criterio del profesional responsable."
        )
    with right:
        st.subheader("Hitos proyectados")
        milestone_table = pd.DataFrame(
            {
                "Hito": ["25 %", "50 %", "75 %", "95 %"],
                "Fecha": [milestones["d25"], milestones["d50"], milestones["d75"], milestones["d95"]],
            }
        )
        st.dataframe(milestone_table, hide_index=True, width="stretch")

with tab_observations:
    st.subheader("Cerrar el circuito con el campo")
    st.markdown("#### Carga de observaciones")
    st.caption(
        "Admite FECHA + PLM2 (flujo por intervalo) o FECHA + "
        "EMERGENCIA_ACUMULADA/OBSERVADO (0–1 o 0–100 %). También reconoce "
        "tres repeticiones (1, 2, 3), una columna media por m² y una columna "
        "opcional COBERTURA_PCT o cobertura."
    )
    uploaded_observations = st.file_uploader(
        "Archivo de emergencia observada",
        type=["xlsx", "xls", "csv", "tsv"],
        key="observed_emergence_upload",
    )
    upload_columns = st.columns([1.5, 1])
    upload_mode_label = upload_columns[0].selectbox(
        "Formato de los valores",
        [
            "Detectar automáticamente",
            "Flujo por intervalo (PLM2)",
            "Acumulada (%)",
        ],
    )
    upload_uncertainty_pct = upload_columns[1].number_input(
        "Incertidumbre si no hay repeticiones (%)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=1.0,
    )
    mode_map = {
        "Detectar automáticamente": "auto",
        "Flujo por intervalo (PLM2)": "flujo",
        "Acumulada (%)": "acumulado",
    }
    if uploaded_observations is not None:
        try:
            raw_observations, file_metadata = read_observation_file(uploaded_observations)
            embedded_coverage = None
            embedded_coverage_metadata = None
            if has_coverage_data(raw_observations):
                embedded_coverage, embedded_coverage_metadata = (
                    prepare_coverage_series(
                        raw_observations,
                        minimum_date=weather_dates.min(),
                        maximum_date=weather_dates.max(),
                        source_name=(
                            f'{file_metadata["archivo"]} · '
                            f'hoja {file_metadata["hoja"]}'
                        ),
                    )
                )
            prepared_observations, import_metadata = prepare_observations(
                raw_observations,
                base_trajectory,
                mode=mode_map[upload_mode_label],
                uncertainty=upload_uncertainty_pct / 100.0,
                seasonal_potential_prior=seasonal_potential_prior,
                source_name=(
                    f'{file_metadata["archivo"]} · hoja {file_metadata["hoja"]}'
                ),
            )
            st.success(
                f'{import_metadata["filas"]} observaciones válidas. '
                f'Modo detectado: {import_metadata["modo"]}.'
            )
            if embedded_coverage is not None:
                st.success(
                    f'Cobertura incluida: {embedded_coverage_metadata["filas"]} '
                    f'mediciones, rango '
                    f'{embedded_coverage_metadata["cobertura_minima"]:.0f}–'
                    f'{embedded_coverage_metadata["cobertura_maxima"]:.0f} %.'
                )
            if import_metadata["modo"] == "flujo":
                has_repetitions = "n_repeticiones" in import_metadata
                summary_columns = st.columns(4 if has_repetitions else 3)
                summary_columns[0].metric(
                    "Total observado",
                    f'{import_metadata["total_observado_plm2"]:.1f} plantas/m²',
                )
                summary_columns[1].metric(
                    "Potencial estacional estimado",
                    f'{import_metadata["potencial_estacional_plm2"]:.1f} plantas/m²',
                )
                summary_columns[2].metric(
                    "Progreso simulado en última fecha",
                    f'{import_metadata["progreso_modelo_ultima_fecha"]:.0%}',
                )
                if has_repetitions:
                    summary_columns[3].metric(
                        "Repeticiones",
                        f'{import_metadata["n_repeticiones"]} por fecha',
                    )
                st.caption(import_metadata["metodo_normalizacion"].capitalize() + ".")
                st.caption(
                    "Cada fila se interpreta como el flujo ocurrido desde el "
                    "muestreo anterior y se compara con la suma del flujo diario simulado."
                )
                if has_repetitions:
                    factor = import_metadata["factor_conversion_repeticiones"]
                    area = import_metadata["area_cuadrante_inferida_m2"]
                    st.info(
                        f"Repeticiones detectadas automáticamente. Factor de conversión "
                        f"a m²: ×{factor:.2f} (área inferida: {area:.2f} m²). "
                        f"Incertidumbre: {import_metadata['metodo_incertidumbre']} "
                        f"({import_metadata['incertidumbre_minima']:.1%}–"
                        f"{import_metadata['incertidumbre_maxima']:.1%})."
                    )

            preview_columns = [
                "Fecha",
                "Valor_original",
                "Unidad_original",
                "Observado",
                "Incertidumbre",
            ]
            if "Acumulado_PLM2" in prepared_observations:
                preview_columns.insert(2, "Acumulado_PLM2")
            repetition_preview = [
                column
                for column in prepared_observations.columns
                if column.startswith("Repeticion_")
                and column.endswith("_original")
            ]
            if repetition_preview:
                preview_columns[2:2] = repetition_preview + ["EE_repeticiones_PLM2"]
            st.dataframe(
                prepared_observations[preview_columns],
                hide_index=True,
                width="stretch",
                column_config={
                    "Observado": st.column_config.NumberColumn(
                        "Progreso acumulado estimado", format="percent"
                    ),
                    "Incertidumbre": st.column_config.NumberColumn(
                        "Incertidumbre", format="percent"
                    ),
                },
            )
            if embedded_coverage is not None:
                st.markdown("**Cobertura detectada en el mismo archivo**")
                st.dataframe(
                    embedded_coverage[["Fecha", "Cobertura_PCT"]],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Cobertura_PCT": st.column_config.NumberColumn(
                            "Cobertura (%)",
                            min_value=0,
                            max_value=100,
                            format="%.1f %%",
                        )
                    },
                )
            if st.button(
                (
                    "Incorporar emergencia y cobertura al gemelo"
                    if embedded_coverage is not None
                    else "Incorporar observaciones al gemelo"
                ),
                type="primary",
                key="save_observation_upload",
            ):
                store.upsert_observations(site_id, prepared_observations)
                if embedded_coverage is not None:
                    store.upsert_coverage_observations(site_id, embedded_coverage)
                st.success(
                    f'Se incorporaron {len(prepared_observations)} fechas de '
                    f'emergencia al lote {site_id}'
                    + (
                        f' y {len(embedded_coverage)} mediciones de cobertura.'
                        if embedded_coverage is not None
                        else "."
                    )
                )
                st.rerun()
        except Exception as error:
            st.error(f"No fue posible procesar las observaciones: {error}")

    st.divider()
    st.markdown("#### Observaciones guardadas")
    if observations.empty:
        st.info("No hay observaciones guardadas para este lote.")
    else:
        st.dataframe(
            observations,
            hide_index=True,
            width="stretch",
            column_config={
                "Observado": st.column_config.NumberColumn(
                    "Emergencia acumulada", format="percent"
                ),
                "Incertidumbre": st.column_config.NumberColumn(
                    "Incertidumbre", format="percent"
                ),
            },
        )
        with st.expander("Borrar observaciones"):
            available_dates = observations["Fecha"].dt.date.tolist()
            select_all_dates = st.checkbox(
                "Seleccionar todas las fechas",
                value=False,
                key="select_all_observation_dates",
            )
            selected_dates = st.multiselect(
                "Fechas que desea borrar",
                options=available_dates,
                format_func=lambda value: value.strftime("%d/%m/%Y"),
                disabled=select_all_dates,
                key="observation_dates_to_delete",
            )
            dates_to_delete = available_dates if select_all_dates else selected_dates
            confirmation = st.checkbox(
                f"Confirmo que deseo borrar {len(dates_to_delete)} observación(es) "
                f"del lote {site_id}.",
                value=False,
                key="confirm_observation_deletion",
            )
            if st.button(
                "Borrar observaciones seleccionadas",
                disabled=not dates_to_delete or not confirmation,
                key="delete_selected_observations",
            ):
                deleted = store.delete_observations(site_id, dates_to_delete)
                st.success(f"Se borraron {deleted} observación(es) del lote {site_id}.")
                st.rerun()

    st.divider()
    st.markdown("#### Cobertura observada del rastrojo")
    st.caption(
        "Cargue un CSV o Excel con las columnas FECHA + COBERTURA_PCT. "
        "Los valores deben estar entre 0 y 100."
    )
    uploaded_coverage = st.file_uploader(
        "Archivo de cobertura",
        type=["xlsx", "xls", "csv", "tsv"],
        key="observed_coverage_upload",
    )
    if uploaded_coverage is not None:
        try:
            raw_coverage, coverage_file_metadata = read_coverage_file(
                uploaded_coverage
            )
            prepared_coverage, coverage_import_metadata = prepare_coverage_series(
                raw_coverage,
                minimum_date=weather_dates.min(),
                maximum_date=weather_dates.max(),
                source_name=(
                    f'{coverage_file_metadata["archivo"]} · '
                    f'hoja {coverage_file_metadata["hoja"]}'
                ),
            )
            st.success(
                f'{coverage_import_metadata["filas"]} mediciones válidas; '
                f'rango {coverage_import_metadata["cobertura_minima"]:.0f}–'
                f'{coverage_import_metadata["cobertura_maxima"]:.0f} %.'
            )
            st.dataframe(
                prepared_coverage,
                hide_index=True,
                width="stretch",
                column_config={
                    "Cobertura_PCT": st.column_config.NumberColumn(
                        "Cobertura (%)", min_value=0, max_value=100, format="%.1f %%"
                    )
                },
            )
            if st.button(
                "Incorporar serie de cobertura",
                type="primary",
                key="save_coverage_upload",
            ):
                store.upsert_coverage_observations(site_id, prepared_coverage)
                st.success(
                    f'Se incorporaron {len(prepared_coverage)} mediciones de cobertura '
                    f'al lote {site_id}.'
                )
                st.rerun()
        except Exception as error:
            st.error(f"No fue posible procesar la cobertura: {error}")

    if coverage_observations.empty:
        st.info("No hay mediciones de cobertura guardadas para este lote.")
    else:
        st.dataframe(
            coverage_observations,
            hide_index=True,
            width="stretch",
            column_config={
                "Cobertura_PCT": st.column_config.NumberColumn(
                    "Cobertura (%)", min_value=0, max_value=100, format="%.1f %%"
                )
            },
        )
        with st.expander("Borrar mediciones de cobertura"):
            coverage_dates = coverage_observations["Fecha"].dt.date.tolist()
            selected_coverage_dates = st.multiselect(
                "Fechas de cobertura que desea borrar",
                options=coverage_dates,
                format_func=lambda value: value.strftime("%d/%m/%Y"),
                key="coverage_dates_to_delete",
            )
            confirm_coverage_deletion = st.checkbox(
                f"Confirmo que deseo borrar {len(selected_coverage_dates)} "
                "medición(es) de cobertura.",
                value=False,
                key="confirm_coverage_deletion",
            )
            if st.button(
                "Borrar cobertura seleccionada",
                disabled=(
                    not selected_coverage_dates or not confirm_coverage_deletion
                ),
                key="delete_selected_coverage",
            ):
                deleted = store.delete_coverage_observations(
                    site_id, selected_coverage_dates
                )
                st.success(f"Se borraron {deleted} medición(es) de cobertura.")
                st.rerun()

with tab_calibration:
    st.subheader("Memoria local de Lartigau · campaña 2026")
    st.info(calibration_audit["reason"])
    if calibration_profile is not None:
        fit = calibration_profile["fit"]
        source = calibration_profile["source"]
        fit_table = pd.read_csv(CALIBRATION_DIR / "lartigau_2026_fit.csv")
        st.write(
            f'**{fit["n_observations"]} fechas**, del '
            f'**{pd.Timestamp(calibration_profile["observations_start"]).strftime("%d/%m/%Y")} '
            f'al {pd.Timestamp(calibration_profile["training_through"]).strftime("%d/%m/%Y")}**. '
            'El archivo aporta flujos en plantas/m², sin repeticiones. Se comparan '
            'los conteos con las sumas del flujo simulado en cada intervalo real.'
        )
        st.caption(
            f'Total registrado: {source["observed_total_plm2"]:.1f} plantas/m². '
            'Es un total observado parcial, no el potencial estacional del lote. '
            'Al no contar con repeticiones, el ajuste utiliza un piso de ponderación '
            'común; no se dispone de un error de muestreo medido.'
        )
        if calibration_profile.get("initial_zero_reference"):
            first_interval = fit_table.iloc[0]
            st.caption(
                'El registro inicial de cero delimita el primer intervalo: '
                f'{pd.Timestamp(first_interval["Inicio_exclusivo"]).strftime("%d/%m/%Y")} '
                f'a {pd.Timestamp(first_interval["Fecha"]).strftime("%d/%m/%Y")} '
                f'({int(first_interval["Dias_intervalo"])} días). Su conteo participa del ajuste.'
            )
        calibration_metrics = st.columns(3)
        calibration_metrics[0].metric("Intervalos de ajuste", fit["n_intervals"])
        calibration_metrics[1].metric("RMSE base · ajuste", f'{fit["rmse_base_plm2"]:.1f} plantas/m²')
        calibration_metrics[2].metric("RMSE calibrado · ajuste", f'{fit["rmse_calibrated_plm2"]:.1f} plantas/m²')
        st.caption(
            'Los RMSE anteriores se calculan sobre los datos usados para ajustar. '
            'Las curvas se escalan al total de la ventana muestreada; esa escala '
            'no se transfiere como densidad a otros lotes.'
        )
        fit_figure = go.Figure()
        for column, label, color in (
            ("Observado_PLM2", "Observado por intervalo", "#df5b3f"),
            ("Base_PLM2_ajuste", "PREDWEEM base", "#83938b"),
            ("Calibrado_PLM2_ajuste", "Calibración retrospectiva", "#9260bd"),
        ):
            fit_figure.add_trace(go.Scatter(
                x=fit_table["Fecha"], y=fit_table[column], name=label,
                mode="lines+markers", line=dict(color=color),
            ))
        fit_figure.update_layout(
            height=380, xaxis_title="Fin del intervalo", yaxis_title="Plantas/m² por intervalo",
            hovermode="x unified", plot_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fit_figure, width="stretch")
        st.markdown("#### Evaluación en intervalos posteriores")
        st.write(calibration_profile["validation"]["note"])
        validation = calibration_profile["validation"]
        validation_metrics = st.columns(2)
        validation_metrics[0].metric("RMSE base · evaluación temporal", f'{validation["rmse_base_plm2"]:.2f} plantas/m²')
        validation_metrics[1].metric("RMSE calibrado · evaluación temporal", f'{validation["rmse_calibrated_plm2"]:.2f} plantas/m²')
        if validation["rmse_calibrated_plm2"] > validation["rmse_base_plm2"]:
            st.warning(
                "La calibración empeora el RMSE de la evaluación temporal. "
                "La mejora del ajuste retrospectivo no demuestra una mejora predictiva."
            )
        st.caption(
            'La evaluación utiliza intervalos posteriores al pico principal. '
            'Se requieren nuevas campañas para evaluar la transferencia del ajuste '
            'y pronósticos con emisiones fechadas para medir precisión operativa.'
        )
        st.dataframe(
            pd.read_csv(CALIBRATION_DIR / "lartigau_2026_holdout.csv"),
            hide_index=True, width="stretch",
        )
        st.warning(
            'Perfil experimental de una sola campaña. Un parámetro alcanzó el límite '
            'permitido; persisten diferencias que esta capa no puede corregir. '
            'La transferencia a otras campañas requiere validación.'
            if fit["parameter_at_bound"] else
            'Perfil experimental de una sola campaña; falta validar su transferencia a otros años.'
        )
        with st.expander("Supuestos, parámetros y procedencia"):
            st.write(calibration_profile["limitations"])
            st.json(calibration_profile)
        st.download_button(
            "Descargar conteos originales 2026 (CSV)",
            (CALIBRATION_DIR / "lartigau_2026_counts.csv").read_bytes(),
            "lartigau_2026_counts.csv", "text/csv",
        )
        st.caption(
            'Los conteos se conservan como referencia de calibración. Para asimilarlos '
            'en un lote, cargue este CSV en Observaciones. La carga mantiene las reglas '
            'existentes de incorporación y borrado de datos.'
        )

with tab_scenarios:
    st.subheader("¿Qué pasa si…?")
    scenario_columns = st.columns(3)
    extra_rain = scenario_columns[0].slider("Lluvia adicional (mm)", 0, 80, 30, 5)
    rain_days = scenario_columns[1].slider("Distribuida en días", 1, 7, 3)
    temperature_delta = scenario_columns[2].slider("Cambio de temperatura (°C)", -5.0, 5.0, 0.0, 0.5)
    scenario_weather = apply_scenario(weather, as_of, extra_rain, rain_days, temperature_delta)
    scenario_base = run_predweem(
        scenario_weather,
        model,
        parameters,
        coverage_series=coverage_series_for_model,
        normalization_as_of=as_of,
        seasonal_reference=seasonal_reference,
    )
    scenario_calibrated, _ = apply_site_calibration(
        scenario_base, calibration_profile,
        site=calibration_site, as_of=as_of, enabled=calibration_enabled,
        observations=active_observations, model_fingerprint=current_model_fingerprint,
    )
    scenario_twin, _ = assimilate_observations(
        scenario_calibrated,
        active_observations,
        model_uncertainty=model_uncertainty,
        seasonal_potential_prior=seasonal_potential_prior,
    )
    scenario_snapshot = build_twin_snapshot(
        scenario_twin, site_id, as_of, "Escenario", len(assimilation_audit)
    )
    scenario_milestones = milestone_dates(scenario_twin)
    comparison = pd.DataFrame(
        {
            "Indicador": ["Incremento próximos 7 días", "Riesgo", "d50", "d75", "d95"],
            "Escenario base": [
                f'{snapshot["increment_7d"]:.1%}', snapshot["risk_7d"],
                milestones["d50"], milestones["d75"], milestones["d95"],
            ],
            "Escenario simulado": [
                f'{scenario_snapshot["increment_7d"]:.1%}', scenario_snapshot["risk_7d"],
                scenario_milestones["d50"], scenario_milestones["d75"], scenario_milestones["d95"],
            ],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    comparison_figure = go.Figure()
    comparison_figure.add_trace(go.Scatter(x=twin_trajectory["Fecha"], y=twin_trajectory["EMERAC_TWIN"] * 100, name="Base", line=dict(color="#155d3e", width=4)))
    comparison_figure.add_trace(go.Scatter(x=scenario_twin["Fecha"], y=scenario_twin["EMERAC_TWIN"] * 100, name="Escenario", line=dict(color="#df7f34", width=3, dash="dash")))
    comparison_figure.update_layout(height=390, yaxis_title="Emergencia acumulada (%)", hovermode="x unified", plot_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(comparison_figure, width="stretch")
    st.caption("Los escenarios son contrafactuales exploratorios; no modifican el estado guardado del lote.")

with tab_audit:
    st.subheader("Trazabilidad científica")
    st.write("Campañas utilizadas: " + seasonal_reference["Campanas"].iloc[0])
    st.caption("Campañas excluidas: " + seasonal_reference["Campanas_Excluidas"].iloc[0])
    st.write(calibration_audit["reason"])
    if calibration_audit["profile_id"]:
        st.caption(f'Perfil: {calibration_audit["profile_id"]}')
    audit_summary = pd.DataFrame(
        {
            "Variable": [
                "Lote", "Fuente meteorológica", "Corte meteorológico",
                "Horizonte pronosticado", "Cobertura", "Wmax",
                "Incertidumbre modelo", "Potencial previo", "Modo de asimilación",
                "Observaciones asimiladas", "Perfil fisiológico",
                "Decaimiento desde 15/04", "Referencia estacional compartida",
            ],
            "Valor": [
                site_id, source_label,
                pd.Timestamp(as_of).strftime("%d/%m/%Y"),
                f'{forecast_metadata["forecast_days_available"]}/{forecast_metadata["forecast_days_expected"]} días',
                (
                    f"Serie observada; {coverage_at_cutoff:.1f} % al corte"
                    if coverage_series_for_model is not None
                    else f"Constante {coverage} %"
                ),
                f"{w_max:.3f} mm",
                f"{model_uncertainty:.0%}",
                f"{seasonal_potential_prior:.1f} plantas/m²"
                if seasonal_potential_prior is not None else "Automático",
                snapshot["assimilation_mode"], str(len(assimilation_audit)),
                "Lartigau vK4.9.15 Adaptada",
                (
                    f"Techo inicial {parameters.decay_cap_fraction:.0%}; tau={parameters.decay_tau_days:.1f} d; "
                    f"beta={parameters.decay_beta:.5f}; "
                    f"intensidad={parameters.decay_intensity:.2f}"
                ),
                f"Clasificador original; n={reference_campaigns} campañas; excluye 2010, 2015, Balcarce y San Pedro",
            ],
        }
    )
    st.dataframe(audit_summary, hide_index=True, width="stretch")
    if assimilation_audit.empty:
        st.info(
            "Aún no hay observaciones de campo asimiladas. La curva Twin sigue "
            + ("la trayectoria calibrada." if calibration_audit["applied"] else "PREDWEEM base.")
        )
    else:
        st.dataframe(assimilation_audit, hide_index=True, width="stretch")
    export_columns = [
        "Fecha", "TMAX", "TMIN", "Prec", "FUENTE", "TIPODATO", "CALIDADDATO", "EMISION_UTC",
        "EMERREL", "EMERAC_NORMALIZADA",
        "EMERAC_BASE_SIN_CALIBRAR", "EMERAC_CALIBRADA", "EMERREL_CALIBRADA",
        "Calibracion_Perfil", "Calibracion_Aplicada", "Calibracion_Motivo",
        "EMERREL_TWIN", "EMERAC_TWIN", "W_superficial", "Humedad_Relativa",
        "EMERREL_TWIN_PLM2", "EMERAC_TWIN_PLM2", "POTENCIAL_ESTACIONAL_PLM2",
        "MODO_ASIMILACION", "ULTIMA_OBSERVACION", "Cobertura_Rastrojo",
        "Cobertura_Modo", "Cobertura_Observada", "Ke_Suelo",
        "Modulador_Termico_Cobertura",
        "Normalizacion_Modo", "Total_EMERREL_Referencia",
        "Progreso_Estacional_P10", "Progreso_Estacional_Referencia",
        "Progreso_Estacional_P90",
        "Termoinhibida", "TT_DESDE_PICO", "EMERREL_ANTES_DECAIMIENTO",
        "Dias_Desde_15Abr", "Factor_Decaimiento_15Abr", "Techo_EMERREL_15Abr",
        "Tau_Decaimiento_15Abr_d", "Beta_Decaimiento_15Abr",
        "Intensidad_Decaimiento_15Abr", "Fraccion_Maxima_15Abr",
    ]
    export_columns = [
        column for column in export_columns if column in twin_trajectory.columns
    ]
    st.download_button(
        "Descargar trayectoria auditable (CSV)",
        twin_trajectory[export_columns].to_csv(index=False).encode("utf-8"),
        f"{site_id}_predweem_twin.csv",
        "text/csv",
    )
