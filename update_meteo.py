# -*- coding: utf-8 -*-
"""Meteorología PREDWEEM Lartigau 2026.

Histórico operativo: pronósticos MeteoBahía archivados (no observaciones).
Tramo vigente/futuro: MeteoBahía XML de Coronel Falcón.
No utiliza ERA5, ERA5-Land, ERA5-Seamless ni ECMWF histórico.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ZONA_HORARIA = "America/Argentina/Buenos_Aires"
CAMPANIA_START = date(2026, 1, 1)
CAMPANIA_END = date(2026, 10, 1)  # Última fecha meteorológica, inclusive.
TBASE = 2.0
URL_XML = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
ARCHIVO_MAESTRO = Path("meteo_daily.csv")
ARCHIVO_HISTORICO = Path("data/meteo_falcon_pronosticos_archivados_2026.csv")
ARCHIVO_ESTADO = Path("data/estado_actualizacion_meteo.json")
DIR_PRONOSTICOS = Path("data/historico_pronosticos")
FUENTE_HISTORICA = "METEOBAHIA_XML_ARCHIVADO"
FUENTE_PRONOSTICO = "METEOBAHIA_XML_CORONEL_FALCON"
TIPO_HISTORICO = "Historico_pronostico"
CALIDAD_HISTORICA = "Pronostico_MeteoBahia_archivado_no_observado"
COLUMNAS = [
    "Fecha", "TMAX", "TMIN", "Prec", "TMEDIA", "GD_Tb2", "Fuente",
    "TipoDato", "CalidadDato", "Latitud_grilla", "Longitud_grilla",
    "Elevacion_grilla_m", "Emision_UTC",
]


def hoy_argentina() -> date:
    return datetime.now(ZoneInfo(ZONA_HORARIA)).date()


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get(url: str, *, headers=None, timeout=30) -> requests.Response:
    ultimo: Exception | None = None
    for intento in range(1, 5):
        try:
            respuesta = requests.get(url, headers=headers, timeout=timeout)
            respuesta.raise_for_status()
            return respuesta
        except requests.RequestException as error:
            ultimo = error
            print(f"⚠️ HTTP {intento}/4: {error}")
            if intento < 4:
                time.sleep(5 * intento)
    raise RuntimeError(f"No fue posible consultar {url}") from ultimo


def columnas(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    for nombre in COLUMNAS:
        if nombre not in salida:
            salida[nombre] = pd.NA
    return salida[COLUMNAS]


def escribir_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporal, index=False, float_format=float_format)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    temporal.replace(path)


def numero(valor: Any) -> float | None:
    try:
        texto = str(valor).strip().replace(",", ".")
        return float(texto) if texto else None
    except (TypeError, ValueError):
        return None


def normalizar(
    df: pd.DataFrame,
    *,
    fuente: str,
    tipo: str,
    calidad: str,
    emision_utc: str,
) -> pd.DataFrame:
    requeridas = {"Fecha", "TMAX", "TMIN", "Prec"}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas meteorológicas: {sorted(faltantes)}")

    salida = df[["Fecha", "TMAX", "TMIN", "Prec"]].copy()
    salida["Fecha"] = pd.to_datetime(salida["Fecha"], errors="coerce").dt.normalize()
    for columna in ["TMAX", "TMIN", "Prec"]:
        salida[columna] = pd.to_numeric(salida[columna], errors="coerce")

    salida = salida.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"])
    salida = salida.loc[
        salida["TMAX"].between(-25, 55)
        & salida["TMIN"].between(-35, 45)
        & (salida["TMAX"] >= salida["TMIN"])
        & salida["Prec"].between(0, 500)
    ].copy()
    salida["TMEDIA"] = (salida["TMAX"] + salida["TMIN"]) / 2
    salida["GD_Tb2"] = (salida["TMEDIA"] - TBASE).clip(lower=0)
    salida["Fuente"] = fuente
    salida["TipoDato"] = tipo
    salida["CalidadDato"] = calidad
    salida["Latitud_grilla"] = pd.NA
    salida["Longitud_grilla"] = pd.NA
    salida["Elevacion_grilla_m"] = pd.NA
    salida["Emision_UTC"] = emision_utc
    salida["Fecha"] = salida["Fecha"].dt.strftime("%Y-%m-%d")
    return columnas(
        salida.drop_duplicates("Fecha", keep="last")
        .sort_values("Fecha")
        .reset_index(drop=True)
    )


def leer_archivo_historico() -> pd.DataFrame:
    if not ARCHIVO_HISTORICO.exists():
        raise FileNotFoundError(
            f"No existe el histórico requerido: {ARCHIVO_HISTORICO}"
        )
    bruto = pd.read_csv(ARCHIVO_HISTORICO)
    historico = normalizar(
        bruto,
        fuente=FUENTE_HISTORICA,
        tipo=TIPO_HISTORICO,
        calidad=CALIDAD_HISTORICA,
        emision_utc="No_disponible_archivo",
    )
    if historico.empty:
        raise ValueError("El archivo histórico MeteoBahía no contiene filas válidas.")
    return historico


def meteobahia() -> pd.DataFrame:
    if hoy_argentina() > CAMPANIA_END:
        return pd.DataFrame(columns=COLUMNAS)
    print("📡 MeteoBahía XML / Coronel Falcón")
    respuesta = get(
        URL_XML,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://meteobahia.com.ar/",
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
        },
        timeout=30,
    )
    filas: list[dict[str, Any]] = []
    for dia in ET.fromstring(respuesta.content).findall(".//forecast/tabular/day"):
        def valor(tag: str):
            nodo = dia.find(f"./{tag}")
            return nodo.get("value") if nodo is not None else None

        filas.append({
            "Fecha": valor("fecha"),
            "TMAX": numero(valor("tmax")),
            "TMIN": numero(valor("tmin")),
            "Prec": numero(valor("precip")),
        })

    if not filas:
        raise ValueError("El XML de MeteoBahía no contiene días procesables.")

    emision = utc_iso()
    pronostico = normalizar(
        pd.DataFrame(filas),
        fuente=FUENTE_PRONOSTICO,
        tipo="Pronostico",
        calidad="Pronostico_deterministico_Coronel_Falcon",
        emision_utc=emision,
    )
    hoy = hoy_argentina()
    fechas = pd.to_datetime(pronostico["Fecha"], errors="coerce")
    pronostico = pronostico.loc[(fechas.dt.date >= hoy) & (fechas.dt.date <= CAMPANIA_END)].copy()
    if pronostico.empty or pd.to_datetime(pronostico["Fecha"]).min().date() != hoy:
        raise ValueError("MeteoBahía no incluye la fecha actual.")
    return pronostico.reset_index(drop=True)


def faltantes(df: pd.DataFrame, inicio: date, fin: date) -> list[date]:
    if inicio > fin:
        return []
    esperadas = pd.date_range(inicio, fin, freq="D")
    presentes = pd.DatetimeIndex(
        pd.to_datetime(df["Fecha"], errors="coerce").dropna()
    ).normalize()
    return [marca.date() for marca in esperadas.difference(presentes)]


def validar(total: pd.DataFrame, hoy: date, fin: date) -> None:
    fechas = pd.to_datetime(total["Fecha"], errors="coerce")
    if total.empty or fechas.isna().any() or fechas.duplicated().any():
        raise ValueError("La serie está vacía o contiene fechas inválidas/duplicadas.")

    if (fechas.dt.date > CAMPANIA_END).any():
        raise ValueError("Hay fechas posteriores al cierre de campaña.")

    criticas = total[["TMAX", "TMIN", "TMEDIA", "Prec"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if criticas.isna().any().any():
        raise ValueError("Hay valores meteorológicos nulos.")
    if (criticas["TMAX"] < criticas["TMIN"]).any() or (criticas["Prec"] < 0).any():
        raise ValueError("Hay valores meteorológicos físicamente inválidos.")

    huecos = faltantes(total, CAMPANIA_START, fin)
    if huecos:
        raise ValueError(
            "La serie no es continua: "
            + ", ".join(fecha.isoformat() for fecha in huecos[:20])
        )

    pasadas = fechas.dt.date < hoy
    futuras = fechas.dt.date >= hoy
    if not pasadas.any() or (hoy <= CAMPANIA_END and not futuras.any()):
        raise ValueError("Deben existir histórico y pronóstico desde hoy.")
    if not total.loc[pasadas, "Fuente"].astype(str).eq(FUENTE_HISTORICA).all():
        raise ValueError("El histórico no proviene exclusivamente del archivo MeteoBahía.")
    if not total.loc[pasadas, "TipoDato"].astype(str).eq(TIPO_HISTORICO).all():
        raise ValueError("El histórico no quedó identificado como pronóstico archivado.")
    if not total.loc[futuras, "Fuente"].astype(str).eq(FUENTE_PRONOSTICO).all():
        raise ValueError("El tramo futuro no proviene exclusivamente de MeteoBahía XML.")
    if total["Fuente"].astype(str).str.contains("ERA5|ECMWF", case=False, regex=True).any():
        raise ValueError("Persisten fuentes ERA5 o ECMWF en la serie operativa.")


def actualizar_archivo(historico: pd.DataFrame, pronostico: pd.DataFrame) -> pd.DataFrame:
    combinado = pd.concat([
        historico[["Fecha", "TMAX", "TMIN", "Prec"]],
        pronostico[["Fecha", "TMAX", "TMIN", "Prec"]],
    ], ignore_index=True)
    combinado["Fecha"] = pd.to_datetime(combinado["Fecha"], errors="coerce")
    combinado = (
        combinado.dropna(subset=["Fecha"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
    )
    combinado = combinado.loc[combinado["Fecha"].dt.date <= CAMPANIA_END].copy()
    combinado["Fecha"] = combinado["Fecha"].dt.strftime("%Y-%m-%d")
    return combinado.reset_index(drop=True)


def ejecutar() -> pd.DataFrame:
    hoy = hoy_argentina()
    ayer = min(hoy - timedelta(days=1), CAMPANIA_END)
    historico_base = leer_archivo_historico()
    pronostico = meteobahia()

    archivo_actualizado = actualizar_archivo(historico_base, pronostico)
    historico = normalizar(
        archivo_actualizado,
        fuente=FUENTE_HISTORICA,
        tipo=TIPO_HISTORICO,
        calidad=CALIDAD_HISTORICA,
        emision_utc="No_disponible_archivo",
    )
    fechas_historicas = pd.to_datetime(historico["Fecha"], errors="coerce")
    historico = historico.loc[
        (fechas_historicas.dt.date >= CAMPANIA_START)
        & (fechas_historicas.dt.date <= ayer)
    ].copy()

    total = columnas(pd.concat([historico, pronostico], ignore_index=True))
    total["Fecha_dt"] = pd.to_datetime(total["Fecha"], errors="coerce")
    total = (
        total.dropna(subset=["Fecha_dt"])
        .sort_values("Fecha_dt")
        .drop_duplicates("Fecha_dt", keep="last")
        .sort_values("Fecha_dt")
    )
    fin = CAMPANIA_END if hoy > CAMPANIA_END else total["Fecha_dt"].max().date()
    total["Fecha"] = total["Fecha_dt"].dt.strftime("%Y-%m-%d")
    total = columnas(total.drop(columns=["Fecha_dt"])).reset_index(drop=True)
    validar(total, hoy, fin)

    escribir_csv(total, ARCHIVO_MAESTRO)
    escribir_csv(
        archivo_actualizado[["Fecha", "TMAX", "TMIN", "Prec"]],
        ARCHIVO_HISTORICO,
        float_format="%.1f",
    )
    if not pronostico.empty:
        DIR_PRONOSTICOS.mkdir(parents=True, exist_ok=True)
        marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        escribir_csv(
            pronostico,
            DIR_PRONOSTICOS / f"meteobahia_coronel_falcon_{marca}.csv",
        )

    estado = {
        "ejecucion_utc": utc_iso(),
        "sitio": "Lartigau",
        "inicio_campania": CAMPANIA_START.isoformat(),
        "fin_campania": CAMPANIA_END.isoformat(),
        "fuente_historica": FUENTE_HISTORICA,
        "tipo_historico": TIPO_HISTORICO,
        "calidad_historica": CALIDAD_HISTORICA,
        "naturaleza_historico": "Pronosticos MeteoBahia archivados; no son observaciones de estacion",
        "archivo_historico": str(ARCHIVO_HISTORICO),
        "inicio_historico": str(historico["Fecha"].min()),
        "fin_historico": str(historico["Fecha"].max()),
        "filas_historicas": len(historico),
        "fuente_pronostico": FUENTE_PRONOSTICO if len(pronostico) else None,
        "inicio_pronostico": str(pronostico["Fecha"].min()) if len(pronostico) else None,
        "fin_pronostico": str(pronostico["Fecha"].max()) if len(pronostico) else None,
        "filas_pronostico": len(pronostico),
        "huecos_finales": [
            fecha.isoformat() for fecha in faltantes(total, CAMPANIA_START, fin)
        ],
        "advertencia": (
            "El histórico operativo corresponde a pronósticos MeteoBahía archivados, "
            "no a observaciones. ERA5-Seamless y ECMWF fueron retirados."
        ),
    }
    ARCHIVO_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    temporal = ARCHIVO_ESTADO.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    temporal.replace(ARCHIVO_ESTADO)

    print(
        f"✅ Histórico MeteoBahía={len(historico)}; "
        f"pronóstico vigente={len(pronostico)}; total={len(total)}"
    )
    return total


if __name__ == "__main__":
    try:
        ejecutar()
    except Exception as error:
        print(
            f"❌ Error: {error}. No se reemplazó meteo_daily.csv.",
            file=sys.stderr,
        )
        raise SystemExit(1)
