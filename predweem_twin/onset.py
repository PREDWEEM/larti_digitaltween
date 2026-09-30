"""Aviso preventivo de inicio; no modifica la trayectoria ni el reloj térmico."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weather import forecast_mask


def onset_alert(trajectory, as_of, observations=None, enabled=True) -> dict:
    """Consulta el primer pico del motor en la campaña hasta el corte + 7 días.

    Requiere la trayectoria desde el comienzo de la campaña. Los conteos son
    opcionales y solo se consultan hasta el corte. Una visita positiva prueba
    que ya había emergencia, sin convertirla en fecha exacta de inicio.
    """
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    horizon = pd.date_range(cutoff + pd.Timedelta(days=1), periods=7)
    result = dict(
        enabled=bool(enabled), horizon_days=7, status="disabled", level="info",
        title="Alerta preventiva de inicio desactivada", message="",
        model_onset_date=None, first_positive_date=None,
        monitoring_alert_date=None,
        forecast_days_available=0, mode="Sin datos",
    )
    if not enabled:
        return result
    frame = trajectory.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.tz_localize(None).dt.normalize()
    frame = frame.loc[frame["Fecha"].dt.year.eq(cutoff.year) & frame["Fecha"].le(horizon[-1])]
    frame = frame.loc[~frame["Fecha"].duplicated(keep=False)].sort_values("Fecha")
    if "Primer_Pico_Habilitado" in frame:
        reached = frame.loc[frame["Primer_Pico_Habilitado"].eq(True), "Fecha"]
        if not reached.empty:
            result["model_onset_date"] = reached.iloc[0].date().isoformat()
            # Referencia gráfica calculada, no registro de una emisión pasada.
            result["monitoring_alert_date"] = (
                reached.iloc[0] - pd.Timedelta(days=7)
            ).date().isoformat()

    if observations is not None and not observations.empty:
        dates = pd.to_datetime(observations["Fecha"], errors="coerce").dt.tz_localize(None).dt.normalize()
        positive = pd.Series(False, index=observations.index)
        for column in ("Flujo_observado_PLM2", "Observado"):
            if column in observations:
                values = pd.to_numeric(observations[column], errors="coerce")
                positive |= values.gt(0) & np.isfinite(values)
        dates = dates[positive & dates.le(cutoff) & dates.dt.year.eq(cutoff.year)]
        if not dates.empty:
            first = dates.min()
            result.update(
                status="observed", mode="Conteo de campo", level="warning",
                title="Emergencia ya registrada en el lote",
                first_positive_date=first.date().isoformat(),
                message=f"Había plantas a más tardar el {first:%d/%m/%Y}. "
                        "El primer conteo positivo no fija el día exacto de inicio. "
                        "Continúe el seguimiento de los nuevos nacimientos.",
            )
            return result

    onset = pd.Timestamp(result["model_onset_date"]) if result["model_onset_date"] else None
    if onset is not None and onset <= cutoff:
        result.update(
            status="started", mode="Inicio modelado",
            title=f"Inicio modelado: {onset:%d/%m/%Y}",
            message="El modelo ya activó el primer pico. La fecha es estimada; "
                    "contraste con una recorrida cuando sea posible.",
        )
        return result

    future = frame.loc[frame["Fecha"].isin(horizon)].copy()
    if "Primer_Pico_Habilitado" not in future or "EMERREL" not in future:
        result.update(status="unavailable", title="Inicio no evaluable", message="Faltan datos del inicio modelado.")
        return result
    values = pd.to_numeric(future["EMERREL"], errors="coerce")
    future = future.loc[np.isfinite(values) & values.ge(0) & future["Primer_Pico_Habilitado"].notna()]
    available = len(future)
    result["forecast_days_available"] = available
    result["mode"] = "Sin horizonte meteorológico" if not available else "Revisión retrospectiva / fuente no verificada"
    if available and forecast_mask(future).all():
        result["mode"] = "Pronóstico disponible; emisión no documentada"
        if "EMISION_UTC" in future:
            emissions = pd.to_datetime(future["EMISION_UTC"], errors="coerce", utc=True)
            day_end = (cutoff + pd.Timedelta(days=1)).tz_localize("America/Argentina/Buenos_Aires")
            if emissions.notna().all() and emissions.lt(day_end).all():
                result["mode"] = "Pronóstico disponible al corte"
            elif emissions.notna().any():
                result["mode"] = "Revisión retrospectiva / emisión posterior al corte"
    # Una señal positiva en un horizonte parcial también merece vigilancia.
    if onset is not None and onset in set(future["Fecha"]):
        days = (onset - cutoff).days
        result.update(
            status="watch", level="warning",
            title="Alerta preventiva: posible inicio en los próximos 7 días",
            message=f"Inicio modelado para el {onset:%d/%m/%Y} (dentro de {days} días). "
                    "Priorice una recorrida para detectar los primeros nacimientos. "
                    f"Horizonte disponible: {available}/7 días. La fecha puede cambiar con la meteorología.",
        )
    elif available < 7:
        result.update(
            status="unavailable", title="Alerta de inicio: horizonte incompleto",
            message=f"Disponibles {available}/7 días. No se puede descartar un inicio en la próxima semana.",
        )
    else:
        result.update(
            status="no_signal", title="Sin inicio previsto en los próximos 7 días",
            message="El modelo no activa el primer pico en este horizonte. "
                    "Esto no descarta nacimientos: mantenga el seguimiento del lote.",
        )
    return result
