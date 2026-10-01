"""Normalización causal, persistencia de conteos y disponibilidad operativa.

La meteorología 2026 desplazada a 2027 se usa sólo como fixture de regresión;
no constituye un pronóstico ni una validación de la campaña 2027.
"""
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predweem_twin.assimilation import assimilate_observations
from predweem_twin.calibration import apply_site_calibration
from predweem_twin.charts import trajectory_charts
from predweem_twin.core import ModelParameters, PracticalANNModel, run_predweem
from predweem_twin.observations import prepare_observations
from predweem_twin.onset import onset_alert
from predweem_twin.state import build_twin_snapshot, milestone_dates
from predweem_twin.storage import SCHEMA, TwinStore

ROOT = Path(__file__).parents[1]
SITE_KEY = 'lartigau'
SITE_NAME = 'Lartigau'
RESERVOIR = SITE_KEY == 'san_pedro'


def reference_at(cutoff):
    if RESERVOIR:
        from predweem_twin.seasonal import load_seasonal_reference
        return load_seasonal_reference(ROOT / 'data/reference/san_pedro_2025_2026.json', as_of=cutoff)
    from predweem_twin.seasonal import load_local_seasonal_reference
    return load_local_seasonal_reference(ROOT, as_of=cutoff)


@pytest.fixture(scope='module')
def inputs():
    weather = pd.read_csv(ROOT / f'data/calibration/{SITE_KEY}_2026_weather.csv', parse_dates=['Fecha'])
    weather['Fecha'] += pd.DateOffset(years=1)
    return weather, PracticalANNModel.from_directory(ROOT / 'models')


def run(inputs, cutoff, days=7, operational=True):
    weather, model = inputs
    cutoff = pd.Timestamp(cutoff)
    return run_predweem(
        weather.loc[weather.Fecha.le(cutoff + pd.Timedelta(days=days))], model, ModelParameters(),
        normalization_as_of=cutoff if operational else None,
        seasonal_reference=reference_at(cutoff) if operational else None,
    )


def pending_trajectory(inputs):
    result = run(inputs, '2027-01-10')
    if RESERVOIR:
        # The mass-conserving motor has a known scale even before onset.
        # Inject missing normalization only to exercise downstream safeguards.
        result['Normalizacion_Disponible'] = False
        result['EMERAC_NORMALIZADA'] = np.nan
        result['Total_EMERREL_Referencia'] = np.nan
    return result


@pytest.mark.parametrize('cutoff', ['2027-01-10', '2027-02-09', '2027-03-10', '2027-03-30', '2027-04-30'])
def test_extending_horizon_does_not_change_past_percentage_or_raw_onset(inputs, cutoff):
    past = run(inputs, cutoff, days=0)
    future = run(inputs, cutoff)
    np.testing.assert_allclose(past.EMERAC_NORMALIZADA,
        future.loc[future.Fecha.le(cutoff), 'EMERAC_NORMALIZADA'], rtol=0, atol=1e-12, equal_nan=True)
    if RESERVOIR:
        assert future.Normalizacion_Disponible.all()
        assert future.Total_EMERREL_Referencia.eq(1).all()
        np.testing.assert_allclose(future.EMERAC_NORMALIZADA, future.EMERAC)
    else:
        assert pd.to_datetime(future.Fecha_Ancla_Normalizacion).dropna().le(cutoff).all()
        if cutoff == '2027-01-10':
            assert not future.Normalizacion_Disponible.any()
            assert future.EMERAC_NORMALIZADA.isna().all()
            assert future.Total_EMERREL_Referencia.isna().all()
        if cutoff == '2027-04-30':
            assert future.Normalizacion_Disponible.all()
    raw = run(inputs, cutoff, operational=False)
    for column in ['EMERREL', 'EMERAC', 'Primer_Pico_Habilitado', 'TT_DESDE_PICO']:
        pd.testing.assert_series_equal(future[column], raw[column])
    assert onset_alert(future, cutoff) == onset_alert(raw, cutoff)


def test_future_weather_cannot_supply_the_denominator(inputs):
    cutoff = pd.Timestamp('2027-03-10')
    weather, model = inputs
    altered = weather.copy()
    altered.loc[altered.Fecha.gt(cutoff), ['TMAX', 'TMIN', 'Prec']] = [28., 17., 80.]
    original = run(inputs, cutoff)
    changed = run((altered, model), cutoff)
    np.testing.assert_allclose(original.loc[original.Fecha.le(cutoff), 'EMERAC_NORMALIZADA'],
        changed.loc[changed.Fecha.le(cutoff), 'EMERAC_NORMALIZADA'], rtol=0, atol=1e-12, equal_nan=True)


def test_nonstandard_index_and_no_anchor_before_cutoff(inputs):
    if RESERVOIR:
        return  # No historical anchor is used by this motor.
    from predweem_twin.seasonal import partial_season_normalization
    frame = run(inputs, '2027-04-30')
    frame.index = np.arange(len(frame)) * 3 + 100
    reference = reference_at('2027-04-30')
    total, metadata = partial_season_normalization(frame, '2026-12-31', reference)
    assert total is None and 'anchor_date' not in metadata
    total, metadata = partial_season_normalization(frame, '2027-04-30', reference)
    assert total > 0 and metadata['anchor_date'] == pd.Timestamp('2027-04-30')


def test_no_reference_is_unknown_unless_the_motor_has_a_fixed_reservoir(inputs):
    weather, model = inputs
    result = run_predweem(weather.iloc[:40], model, ModelParameters(), normalization_as_of='2027-02-09')
    if RESERVOIR:
        assert result.Normalizacion_Disponible.all()
        np.testing.assert_allclose(result.EMERAC_NORMALIZADA, result.EMERAC)
    else:
        assert not result.Normalizacion_Disponible.any()
        assert result.EMERAC_NORMALIZADA.isna().all()


def test_pending_counts_are_retained_and_assimilated_when_scale_becomes_available(inputs, tmp_path):
    early = pending_trajectory(inputs)
    raw = pd.DataFrame({'FECHA': ['2027-01-03', '2027-01-08'], 'PLM2': [12., 21.33]})
    prepared, metadata = prepare_observations(raw, early)
    assert prepared.Observado.isna().all()
    assert metadata['potencial_estacional_plm2'] is None
    assert metadata['progreso_modelo_ultima_fecha'] is None
    assert prepared.Incertidumbre.notna().all()
    store = TwinStore(tmp_path / 'state.db')
    store.upsert_observations('lote', prepared)
    stored = store.observations('lote')
    assert stored.Observado.isna().all()
    np.testing.assert_allclose(stored.Flujo_observado_PLM2, [12., 21.33])
    calibrated, audit = apply_site_calibration(early, None, site=SITE_NAME, as_of='2027-01-10')
    assert not audit['applied']
    twin, audit = assimilate_observations(calibrated, stored)
    assert audit.empty and twin.EMERAC_TWIN.isna().all() and twin.EMERREL_TWIN.isna().all()
    assert twin.POTENCIAL_ESTACIONAL_PLM2.isna().all()
    assert onset_alert(twin, '2027-01-10', stored)['status'] == 'observed'
    snapshot = build_twin_snapshot(twin, 'lote', '2027-01-10', 'test', seasonal_reference=reference_at('2027-01-10'))
    assert snapshot['emergence'] is None and snapshot['remaining'] is None
    assert snapshot['intensity_7d'] == 'Aún no estimable' and snapshot['increment_7d'] is None
    json.dumps(snapshot, allow_nan=False)
    assert all(value is None for value in milestone_dates(twin).values())
    for frequency in ['Diario', 'Semanal']:
        flow, cumulative = trajectory_charts(twin, stored, '2027-01-10', audit,
            seasonal_reference=reference_at('2027-01-10'), flow_frequency=frequency)
        assert 'Intensidad nula del gemelo' not in [t.name for t in flow.data]
        assert 'Conteo de campo' not in [t.name for t in cumulative.data]
        gemelo = next(t for t in flow.data if t.name == f'Flujo {frequency.lower()} del gemelo')
        assert not np.isfinite(np.asarray(gemelo.y, dtype=float)).any()
        assert any(a.name == 'normalization_unavailable' for a in flow.layout.annotations)
    later = run(inputs, '2027-04-30')
    twin, audit = assimilate_observations(later, stored)
    assert len(audit) == len(stored)
    assert twin.EMERAC_TWIN.notna().all()
    assert audit.Flujo_observado_PLM2.sum() == pytest.approx(33.33)


def test_existing_database_migrates_without_losing_rows_or_indexes(tmp_path, inputs):
    path = tmp_path / "legacy.db"
    old_schema = SCHEMA.replace("cumulative REAL CHECK", "cumulative REAL NOT NULL CHECK")
    with sqlite3.connect(path) as connection:
        connection.executescript(old_schema)
        connection.execute("CREATE INDEX observations_note ON observations(note)")
        connection.execute("CREATE TABLE insert_audit (observed_at TEXT)")
        connection.execute("CREATE TRIGGER observe_insert AFTER INSERT ON observations BEGIN INSERT INTO insert_audit VALUES (NEW.observed_at); END")
        connection.execute("INSERT INTO observations(site_id, observed_at, cumulative, uncertainty, note) VALUES ('lote', '2026-01-20', .1, .05, 'original')")
        connection.execute("INSERT INTO snapshots(site_id, as_of, payload) VALUES ('lote', '2026-01-20', '{}')")
        connection.execute("UPDATE sqlite_sequence SET seq=10 WHERE name='observations'")
    store = TwinStore(path)
    with store.connect() as connection:
        assert connection.execute("SELECT id, cumulative, note FROM observations").fetchall() == [(1, .1, "original")]
        assert connection.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='observations_note'").fetchone()
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='observe_insert'").fetchone()
        assert connection.execute("SELECT count(*) FROM insert_audit").fetchone()[0] == 1
        assert next(r for r in connection.execute("PRAGMA table_info(observations)") if r[1] == "cumulative")[3] == 0
    prepared, _ = prepare_observations(pd.DataFrame({"Fecha": ["2027-01-08"], "PLM2": [12.]}), pending_trajectory(inputs))
    store.upsert_observations("lote", prepared)
    stored = TwinStore(path).observations("lote")
    assert len(stored) == 2 and stored.iloc[0].Observado == .1
    assert pd.isna(stored.iloc[1].Observado) and stored.iloc[1].Flujo_observado_PLM2 == 12.
    with store.connect() as connection:
        assert connection.execute("SELECT max(id) FROM observations").fetchone()[0] == 11
        assert connection.execute("SELECT count(*) FROM insert_audit").fetchone()[0] == 2
    store.upsert_observations("lote", prepared)
    assert len(store.observations("lote")) == 2



def test_app_keeps_metrics_and_counts_available_during_pending_normalization(tmp_path, inputs):
    from streamlit.testing.v1 import AppTest

    for path in ROOT.glob('*.py'):
        shutil.copy2(path, tmp_path / path.name)
    for path in ROOT.glob('*.json'):
        shutil.copy2(path, tmp_path / path.name)
    for path in ROOT.glob('*.csv'):
        shutil.copy2(path, tmp_path / path.name)
    for directory in ('models', 'predweem_twin', 'data'):
        shutil.copytree(ROOT / directory, tmp_path / directory,
            ignore=shutil.ignore_patterns('*.db', '*.db-shm', '*.db-wal', '__pycache__'))
    raw = pd.DataFrame({'FECHA': ['2027-01-08'], 'PLM2': [21.33]})
    prepared, _ = prepare_observations(raw, pending_trajectory(inputs))
    prepared['Fecha'] -= pd.DateOffset(years=1)
    store = TwinStore(tmp_path / 'data/twin_state.db')
    store.upsert_observations(SITE_NAME + '-01', prepared)
    at = AppTest.from_file(str(tmp_path / 'app.py'), default_timeout=45).run()
    assert not at.exception, [error.message for error in at.exception]
    cases = [('2026-01-10', RESERVOIR), ('2026-09-01', True)]
    if SITE_KEY == 'azul':
        cases.insert(1, ('2026-03-10', False))
    for cutoff, available in cases:
        at.date_input[0].set_value(pd.Timestamp(cutoff).date()).run()
        assert not at.exception, [error.message for error in at.exception]
        assert not at.error, [error.value for error in at.error]
        metrics = {metric.label: metric.value for metric in at.metric}
        assert (metrics['Emergencia estimada'] != 'Aún no estimable') == available
        assert (metrics['Emergencia remanente'] != 'Aún no estimable') == available
        assert 'nan' not in str(metrics).lower()
        assert len(at.get('plotly_chart')) >= 2
        if not available:
            assert 'Aún no estimable' in metrics['Intensidad de emergencia · 7 días']
            assert '21.3' in metrics['Conteos acumulados registrados']
        stored = store.observations(SITE_NAME + '-01')
        assert stored.Flujo_observado_PLM2.sum() == pytest.approx(21.33)
