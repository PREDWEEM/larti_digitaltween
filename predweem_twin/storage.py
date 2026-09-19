"""Persistencia SQLite por lote para observaciones, estados y auditoría."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    cumulative REAL NOT NULL CHECK(cumulative BETWEEN 0 AND 1),
    uncertainty REAL NOT NULL CHECK(uncertainty > 0),
    note TEXT NOT NULL DEFAULT '',
    raw_value REAL,
    raw_unit TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT 'Manual',
    replicates_json TEXT NOT NULL DEFAULT '',
    replicate_scale REAL,
    replicate_se_plm2 REAL,
    observation_mode TEXT NOT NULL DEFAULT 'acumulado',
    flow_plm2 REAL,
    cumulative_plm2 REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, observed_at)
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS coverage_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    coverage_pct REAL NOT NULL CHECK(coverage_pct BETWEEN 0 AND 100),
    source_name TEXT NOT NULL DEFAULT 'Archivo cargado',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, observed_at)
);
"""


class TwinStore:
    def __init__(self, path: str | Path = "data/twin_state.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            existing = {
                row[1] for row in connection.execute("PRAGMA table_info(observations)")
            }
            migrations = {
                "raw_value": "ALTER TABLE observations ADD COLUMN raw_value REAL",
                "raw_unit": "ALTER TABLE observations ADD COLUMN raw_unit TEXT NOT NULL DEFAULT ''",
                "source_name": "ALTER TABLE observations ADD COLUMN source_name TEXT NOT NULL DEFAULT 'Manual'",
                "replicates_json": "ALTER TABLE observations ADD COLUMN replicates_json TEXT NOT NULL DEFAULT ''",
                "replicate_scale": "ALTER TABLE observations ADD COLUMN replicate_scale REAL",
                "replicate_se_plm2": "ALTER TABLE observations ADD COLUMN replicate_se_plm2 REAL",
                "observation_mode": "ALTER TABLE observations ADD COLUMN observation_mode TEXT NOT NULL DEFAULT 'acumulado'",
                "flow_plm2": "ALTER TABLE observations ADD COLUMN flow_plm2 REAL",
                "cumulative_plm2": "ALTER TABLE observations ADD COLUMN cumulative_plm2 REAL",
            }
            for column, statement in migrations.items():
                if column not in existing:
                    connection.execute(statement)

    def connect(self):
        return sqlite3.connect(self.path)

    def upsert_observation(
        self,
        site_id: str,
        observed_at,
        cumulative: float,
        uncertainty: float,
        note="",
        raw_value=None,
        raw_unit="",
        source_name="Manual",
    ):
        date = pd.Timestamp(observed_at).date().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO observations(
                    site_id, observed_at, cumulative, uncertainty, note,
                    raw_value, raw_unit, source_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id, observed_at) DO UPDATE SET
                    cumulative=excluded.cumulative,
                    uncertainty=excluded.uncertainty,
                    note=excluded.note,
                    raw_value=excluded.raw_value,
                    raw_unit=excluded.raw_unit,
                    source_name=excluded.source_name,
                    replicates_json='',
                    replicate_scale=NULL,
                    replicate_se_plm2=NULL,
                    observation_mode='acumulado',
                    flow_plm2=NULL,
                    cumulative_plm2=NULL,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    site_id,
                    date,
                    float(cumulative),
                    float(uncertainty),
                    note,
                    None if raw_value is None else float(raw_value),
                    raw_unit,
                    source_name,
                ),
            )

    def upsert_observations(self, site_id: str, observations: pd.DataFrame):
        """Guarda una carga validada dentro de una única transacción."""
        rows = []
        repetition_columns = [
            column
            for column in observations.columns
            if str(column).startswith("Repeticion_")
            and str(column).endswith("_original")
        ]
        for _, row in observations.iterrows():
            repetitions = [float(row[column]) for column in repetition_columns]
            flow_value = row.get("Flujo_observado_PLM2")
            cumulative_value = row.get("Acumulado_PLM2")
            is_flow = pd.notna(flow_value)
            rows.append(
                (
                    site_id,
                    pd.Timestamp(row["Fecha"]).date().isoformat(),
                    float(row["Observado"]),
                    float(row["Incertidumbre"]),
                    str(row.get("Nota", "")),
                    float(row["Valor_original"]),
                    str(row.get("Unidad_original", "")),
                    str(row.get("Fuente", "Archivo cargado")),
                    json.dumps(repetitions, ensure_ascii=False) if repetitions else "",
                    float(row["Factor_conversion_repeticiones"])
                    if repetitions
                    else None,
                    float(row["EE_repeticiones_PLM2"])
                    if repetitions
                    else None,
                    "flujo" if is_flow else "acumulado",
                    float(flow_value) if is_flow else None,
                    float(cumulative_value) if pd.notna(cumulative_value) else None,
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO observations(
                    site_id, observed_at, cumulative, uncertainty, note,
                    raw_value, raw_unit, source_name, replicates_json,
                    replicate_scale, replicate_se_plm2, observation_mode,
                    flow_plm2, cumulative_plm2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id, observed_at) DO UPDATE SET
                    cumulative=excluded.cumulative,
                    uncertainty=excluded.uncertainty,
                    note=excluded.note,
                    raw_value=excluded.raw_value,
                    raw_unit=excluded.raw_unit,
                    source_name=excluded.source_name,
                    replicates_json=excluded.replicates_json,
                    replicate_scale=excluded.replicate_scale,
                    replicate_se_plm2=excluded.replicate_se_plm2,
                    observation_mode=excluded.observation_mode,
                    flow_plm2=excluded.flow_plm2,
                    cumulative_plm2=excluded.cumulative_plm2,
                    created_at=CURRENT_TIMESTAMP
                """,
                rows,
            )

    def observations(self, site_id: str) -> pd.DataFrame:
        with self.connect() as connection:
            frame = pd.read_sql_query(
                """
                SELECT observed_at AS Fecha, cumulative AS Observado,
                       uncertainty AS Incertidumbre, raw_value AS Valor_original,
                       raw_unit AS Unidad_original, source_name AS Fuente,
                       replicates_json AS Repeticiones_originales,
                       replicate_scale AS Factor_conversion_repeticiones,
                       replicate_se_plm2 AS EE_repeticiones_PLM2,
                       observation_mode AS Modo,
                       flow_plm2 AS Flujo_observado_PLM2,
                       cumulative_plm2 AS Acumulado_PLM2,
                       note AS Nota, created_at AS Registrado
                FROM observations WHERE site_id=? ORDER BY observed_at
                """,
                connection,
                params=(site_id,),
                parse_dates=["Fecha", "Registrado"],
            )
        if frame.empty:
            return frame
        legacy_flow = frame["Unidad_original"].astype(str).str.contains(
            "por intervalo", case=False, na=False
        )
        frame.loc[legacy_flow, "Modo"] = "flujo"
        frame.loc[legacy_flow & frame["Flujo_observado_PLM2"].isna(), "Flujo_observado_PLM2"] = (
            frame.loc[
                legacy_flow & frame["Flujo_observado_PLM2"].isna(),
                "Valor_original",
            ]
        )
        flow_mask = frame["Modo"].eq("flujo") & frame["Flujo_observado_PLM2"].notna()
        calculated_cumulative = frame["Flujo_observado_PLM2"].where(flow_mask, 0.0).cumsum()
        frame.loc[flow_mask & frame["Acumulado_PLM2"].isna(), "Acumulado_PLM2"] = (
            calculated_cumulative[flow_mask & frame["Acumulado_PLM2"].isna()]
        )
        return frame

    def delete_observations(self, site_id: str, observed_dates) -> int:
        """Borra fechas explícitas de un lote y devuelve la cantidad eliminada."""
        dates = sorted(
            {
                pd.Timestamp(observed_at).date().isoformat()
                for observed_at in observed_dates
            }
        )
        if not dates:
            return 0
        placeholders = ", ".join("?" for _ in dates)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM observations
                WHERE site_id=? AND observed_at IN ({placeholders})
                """,
                (site_id, *dates),
            )
            return int(cursor.rowcount)

    def upsert_coverage_observations(
        self, site_id: str, coverage: pd.DataFrame
    ) -> None:
        """Guarda mediciones fechadas de cobertura para un lote."""
        rows = [
            (
                site_id,
                pd.Timestamp(row["Fecha"]).date().isoformat(),
                float(row["Cobertura_PCT"]),
                str(row.get("Fuente", "Archivo cargado")),
            )
            for _, row in coverage.iterrows()
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO coverage_observations(
                    site_id, observed_at, coverage_pct, source_name
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(site_id, observed_at) DO UPDATE SET
                    coverage_pct=excluded.coverage_pct,
                    source_name=excluded.source_name,
                    created_at=CURRENT_TIMESTAMP
                """,
                rows,
            )

    def coverage_observations(self, site_id: str) -> pd.DataFrame:
        """Recupera la serie de cobertura observada de un lote."""
        with self.connect() as connection:
            return pd.read_sql_query(
                """
                SELECT observed_at AS Fecha, coverage_pct AS Cobertura_PCT,
                       source_name AS Fuente, created_at AS Registrado
                FROM coverage_observations
                WHERE site_id=? ORDER BY observed_at
                """,
                connection,
                params=(site_id,),
                parse_dates=["Fecha", "Registrado"],
            )

    def delete_coverage_observations(self, site_id: str, observed_dates) -> int:
        """Borra mediciones explícitas de cobertura para un lote."""
        dates = sorted(
            {
                pd.Timestamp(observed_at).date().isoformat()
                for observed_at in observed_dates
            }
        )
        if not dates:
            return 0
        placeholders = ", ".join("?" for _ in dates)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM coverage_observations
                WHERE site_id=? AND observed_at IN ({placeholders})
                """,
                (site_id, *dates),
            )
            return int(cursor.rowcount)

    def save_snapshot(self, snapshot: dict):
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO snapshots(site_id, as_of, payload) VALUES (?, ?, ?)",
                (snapshot["site_id"], snapshot["as_of"], json.dumps(snapshot, ensure_ascii=False)),
            )
