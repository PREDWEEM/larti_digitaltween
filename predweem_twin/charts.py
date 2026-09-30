"""Gráfico operativo con referencia histórica anual orientativa."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .flows import annual_historical_reference, historical_weekly_max, weekly_flow_groups
from .state import intensity_category, thermal_window_dates


INTENSITY_COLORS = {
    "Alta": "#dc2626", "Media": "#f97316",
    "Baja": "#eab308", "Nula": "#16a34a",
    "Semana parcial": "#94a3b8", "Sin referencia": "#94a3b8",
}


def _weekly_flow_trace(trace, cutoff, year_start, display_end, historical=False, historical_peak=None):
    """Suma porcentajes diarios en semanas comunes de lunes a domingo.

    No renormaliza, completa datos ausentes ni extrapola semanas parciales.
    Las barras comparten semanas, sin extenderse más allá de las fechas con
    datos. El rayado identifica semanas con menos de siete días.
    """
    dates, totals, widths, offsets, details, patterns, colors = [], [], [], [], [], [], []
    for monday, group in weekly_flow_groups(trace.x, trace.y):
        sunday = monday + pd.Timedelta(days=6)
        start = max(monday, year_start)
        end = min(sunday, display_end)
        valid = group.loc[
            np.isfinite(group["Flujo"]) & group["Flujo"].ge(0)
            & ~group["Fecha"].duplicated(keep=False)
        ]
        count = valid["Fecha"].nunique()
        total = valid["Flujo"].sum(min_count=1)
        dates.append(start)
        totals.append(total)
        first_day = max(start, valid["Fecha"].min()) if count else start
        last_day = min(end, valid["Fecha"].max()) if count else end
        widths.append(max(1, (last_day - first_day).days + 1) * 86400000 * .9)
        offsets.append((first_day - start).days * 86400000)
        patterns.append("/" if count < 7 else "")
        coverage = f"{count}/7 días · " + ("semana completa" if count == 7 else "semana parcial")
        available = (
            f"Datos: {valid['Fecha'].min():%d/%m}–{valid['Fecha'].max():%d/%m}"
            if count else "Sin datos disponibles"
        )
        if historical:
            source = "Total de las ventanas históricas · orientativo"
        else:
            future_days = int((valid["Fecha"] > cutoff).sum())
            source = "Total estacional estimado"
            if future_days:
                source += f" · incluye {future_days} día(s) de proyección"
        ratio = None
        if count < 7:
            category = "Semana parcial"
        elif total == 0:
            category = "Nula"
            ratio = 0.0 if historical_peak is not None else None
        elif historical_peak is None:
            category = "Sin referencia"
        else:
            # Las barras usan porcentaje; el máximo compartido usa fracción.
            ratio = total / (historical_peak * 100)
            category = intensity_category(ratio)
        colors.append(INTENSITY_COLORS[category])
        ratio_label = (
            f"{ratio:.1%} del máximo semanal histórico" if ratio is not None
            else "Sin comparación con el máximo histórico"
        )
        category_label = (
            f"Intensidad: {category}" if category in {"Alta", "Media", "Baja", "Nula"}
            else f"Sin clasificación: {category.lower()}"
        )
        details.append([
            f"{monday:%d/%m}–{sunday:%d/%m}", coverage, available, source,
            category_label, ratio_label,
        ])
    return go.Bar(
        x=dates, y=totals, width=widths, offset=offsets,
        name=trace.name if historical else "Flujo semanal del gemelo",
        marker=dict(color=colors, pattern_shape=patterns),
        opacity=.22 if historical else .9,
        customdata=details,
        hovertemplate=(
            "Semana %{customdata[0]}<br>Flujo: %{y:.2f} % del total<br>"
            "%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<br>"
            "%{customdata[4]}<br>%{customdata[5]}<extra>%{fullData.name}</extra>"
        ),
    )


def trajectory_charts(
    df,
    observations,
    as_of,
    audit=None,
    lower_thermal_time=600.0,
    upper_thermal_time=800.0,
    seasonal_reference=None,
    flow_frequency="Diario",
    onset_notice=None,
):
    if flow_frequency not in ("Diario", "Semanal"):
        raise ValueError("La frecuencia del flujo debe ser Diario o Semanal.")
    cutoff = pd.Timestamp(as_of).normalize()
    year_start = pd.Timestamp(cutoff.year, 1, 1)
    display_end = pd.Timestamp(cutoff.year, 10, 1)
    # La referencia anual sólo se dibuja: no extiende la meteorología ni
    # modifica el estado, las métricas, los hitos o los datos exportados.
    df = df.loc[
        (pd.to_datetime(df["Fecha"]) >= year_start)
        & (pd.to_datetime(df["Fecha"]) <= min(cutoff + pd.Timedelta(days=7), display_end))
    ].copy()
    if observations is not None and not observations.empty:
        observation_dates = pd.to_datetime(observations["Fecha"])
        observations = observations.loc[
            observation_dates.between(year_start, cutoff)
        ].copy()
    if audit is not None and not audit.empty and "Fecha_asimilada" in audit:
        audit_dates = pd.to_datetime(audit["Fecha_asimilada"])
        audit = audit.loc[audit_dates.between(year_start, cutoff)].copy()
    daily_figure = go.Figure()
    cumulative_figure = go.Figure()
    historical = None
    if seasonal_reference is not None:
        historical = annual_historical_reference(seasonal_reference, cutoff)
        historical = historical.loc[historical["Fecha"] <= display_end]
        daily_figure.add_trace(
            go.Bar(
                x=historical["Fecha"], y=historical["Flujo_Diario"] * 100,
                name="Flujo histórico · orientativo",
                marker_color="rgba(144,158,167,.24)",
                hovertemplate=("%{x|%d/%m/%Y}<br>Flujo histórico orientativo: "
                               "%{y:.2f} % del total/día<br>"
                               "Total de las ventanas históricas registradas<br>"
                               "Derivado del acumulado histórico<extra></extra>"),
            ),
        )
        # Las campañas se conservan en el pool, sin curvas individuales.
        cumulative_figure.add_trace(
            go.Scatter(
                x=historical["Fecha"], y=historical["Progreso_Mediano"] * 100,
                name="Pool histórico · orientativo", mode="lines",
                line=dict(color="rgba(131,161,142,.65)", width=2.2, dash="dash"),
                connectgaps=False,
                hovertemplate=("%{x|%d/%m/%Y}<br>Acumulado histórico orientativo: "
                               "%{y:.1f}%<extra>No es pronóstico</extra>"),
            ),
        )
    daily_figure.add_trace(
        go.Bar(
            x=df["Fecha"],
            y=df["EMERREL_TWIN"] * 100,
            name="Flujo diario del gemelo",
            marker_color="#3b82f6",
            opacity=0.62,
            hovertemplate=("%{x|%d/%m/%Y}<br>Flujo diario del gemelo: "
                           "%{y:.2f} % del total/día<br>"
                           "Total estacional estimado<extra></extra>"),
        ),
    )
    if flow_frequency == "Semanal":
        peak = historical_weekly_max(seasonal_reference, cutoff)
        daily_figure = go.Figure([
            _weekly_flow_trace(
                trace, cutoff, year_start, display_end,
                historical=trace.name == "Flujo histórico · orientativo",
                historical_peak=peak,
            )
            for trace in daily_figure.data
        ])
        # Un flujo cero no tiene altura: mostrar una marca verde en y=0,
        # sin inventar una barra positiva ni marcar semanas incompletas.
        twin_weekly = daily_figure.data[-1]
        null_weeks = [
            i for i, detail in enumerate(twin_weekly.customdata)
            if detail[4] == "Intensidad: Nula"
        ]
        if null_weeks:
            daily_figure.add_trace(go.Scatter(
                x=[pd.Timestamp(twin_weekly.x[i]) + pd.Timedelta(
                    milliseconds=twin_weekly.offset[i] + twin_weekly.width[i] / 2
                ) for i in null_weeks],
                y=[0.] * len(null_weeks), mode="markers",
                name="Intensidad nula del gemelo", showlegend=False,
                marker=dict(color=INTENSITY_COLORS["Nula"], size=11, symbol="line-ew",
                            line=dict(color=INTENSITY_COLORS["Nula"], width=3)),
                cliponaxis=False,
                customdata=[twin_weekly.customdata[i] for i in null_weeks],
                hovertemplate=twin_weekly.hovertemplate,
            ))
    cumulative_figure.add_trace(
        go.Scatter(
            x=df["Fecha"],
            y=df.get("EMERAC_BASE_SIN_CALIBRAR", df["EMERAC_NORMALIZADA"]) * 100,
            name="Acumulado PREDWEEM base",
            line=dict(color="#83938b", width=2, dash="dot"),
        ),
    )
    if "Calibracion_Aplicada" in df and df["Calibracion_Aplicada"].any():
        cumulative_figure.add_trace(
            go.Scatter(
                x=df["Fecha"], y=df["EMERAC_CALIBRADA"] * 100,
                name="Calibración Lartigau", line=dict(color="#9260bd", width=2),
            ),
        )
    cumulative_figure.add_trace(
        go.Scatter(
            x=df.loc[df["Fecha"] <= cutoff, "Fecha"],
            y=df.loc[df["Fecha"] <= cutoff, "EMERAC_TWIN"] * 100,
            name="Estado actualizado",
            line=dict(color="#155d3e", width=4),
            fill="tozeroy",
            fillcolor="rgba(66,137,87,.10)",
        ),
    )
    if (df["Fecha"] > cutoff).any():
        projection = df.loc[df["Fecha"] >= cutoff]
        cumulative_figure.add_trace(
            go.Scatter(
                x=projection["Fecha"], y=projection["EMERAC_TWIN"] * 100,
                name="Proyección meteorológica · hasta 7 días",
                mode="lines", line=dict(color="#155d3e", width=3, dash="dash"),
                hovertemplate="%{x|%d/%m/%Y}<br>Proyección: %{y:.1f}%<extra></extra>",
            ),
        )
    if audit is not None and not audit.empty and "Estado_campo_estimado" in audit:
        cumulative_figure.add_trace(
            go.Scatter(
                x=audit["Fecha_asimilada"],
                y=audit["Estado_campo_estimado"] * 100,
                name="Estado estimado desde campo",
                mode="markers",
                marker=dict(color="#df5b3f", size=11, line=dict(color="white", width=2)),
            ),
        )
    elif observations is not None and not observations.empty:
        cumulative_figure.add_trace(
            go.Scatter(
                x=observations["Fecha"],
                y=observations["Observado"] * 100,
                name="Conteo de campo",
                mode="markers",
                marker=dict(color="#df5b3f", size=11, line=dict(color="white", width=2)),
            ),
        )
    thermal_start, thermal_end = thermal_window_dates(
        df, lower_thermal_time, upper_thermal_time
    )
    for figure in (daily_figure, cumulative_figure):
        if thermal_start is not None:
            displayed_thermal_end = thermal_end or pd.Timestamp(df["Fecha"].max())
            figure.add_vrect(
                x0=thermal_start,
                x1=displayed_thermal_end,
                fillcolor="rgba(255,193,7,.22)",
                line_width=0,
                annotation_text=(
                    f"{lower_thermal_time:.0f}–"
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
        figure.add_vline(x=cutoff.timestamp() * 1000, line_color="#162f25", line_dash="dash")
        figure.add_annotation(
            x=0, y=1.07, xref="paper", yref="paper", showarrow=False, xanchor="left",
            text=f"Fecha de consulta · {cutoff:%d/%m/%Y}",
            font=dict(color="#162f25", size=12),
        )
        forecast_start = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        if pd.Timestamp(df["Fecha"].max()) >= forecast_start:
            figure.add_vrect(
                x0=forecast_start,
                x1=pd.Timestamp(df["Fecha"].max()),
                fillcolor="rgba(223,127,52,.10)",
                line_width=0,
                annotation_text="Pronóstico 7 d",
                annotation_position="top right",
            )
        if historical is not None:
            supported = historical.loc[historical["Progreso_Mediano"].notna(), "Fecha"]
            if not supported.empty:
                support_end = supported.iloc[-1]
                orientative_start = max(cutoff, pd.Timestamp(df["Fecha"].max())) + pd.Timedelta(days=1)
                if orientative_start < support_end:
                    figure.add_vrect(
                        x0=orientative_start, x1=support_end,
                        fillcolor="rgba(131,161,142,.04)", line_width=0,
                        layer="below", annotation_text="Histórico orientativo",
                        annotation_position="bottom right",
                        annotation_font_color="#77877b",
                    )
                if support_end < display_end:
                    figure.add_vrect(
                        x0=support_end + pd.Timedelta(days=1), x1=display_end,
                        fillcolor="rgba(150,158,167,.08)", line_width=0,
                        layer="below", annotation_text="Sin referencia disponible",
                        annotation_position="top right",
                        annotation_font_color="#7c8580",
                    )
        months = pd.date_range(year_start, display_end, freq="MS")
        figure.update_xaxes(
            range=[year_start, display_end], title_text=f"Calendario {cutoff.year}",
            tickmode="array", tickvals=months,
            ticktext=["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "01 Oct"],
        )
        figure.update_layout(
            height=570,
            margin=dict(l=65, r=25, t=50, b=165, autoexpand=False),
            legend=dict(orientation="h", y=-.22, yanchor="top", x=0, font=dict(size=11)),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.15,
            barmode="overlay",
        )
    daily_figure.update_yaxes(
        title_text=f"Flujo {flow_frequency.lower()} (% del total)",
        ticksuffix=" %", rangemode="tozero",
    )
    if onset_notice and onset_notice.get("enabled") and onset_notice.get("monitoring_alert_date"):
        monitoring_date = pd.Timestamp(onset_notice["monitoring_alert_date"])
        if (
            onset_notice.get("status") in {"watch", "started", "observed"}
            and year_start <= monitoring_date <= min(cutoff, display_end)
        ):
            # X es el día exacto, no el lunes ni el centro de su barra semanal.
            # La punta queda sobre el calendario y la flecha es vertical.
            daily_figure.add_annotation(
                name="initial_monitoring_alert",
                x=monitoring_date.timestamp() * 1000, xref="x",
                y=0, yref="paper", ax=0, ay=-245,
                axref="pixel", ayref="pixel", showarrow=True,
                arrowhead=2, arrowsize=1.3, arrowwidth=2.5,
                arrowcolor="#6d28d9", standoff=0,
                text=("<b>Alerta inicial de monitoreo</b><br>"
                      f"{monitoring_date:%d/%m/%Y} · estimada"),
                font=dict(color="#5b21b6", size=11),
                bgcolor="rgba(255,255,255,.96)",
                bordercolor="#6d28d9", borderwidth=1, borderpad=5,
                xanchor=("left" if monitoring_date < year_start + pd.Timedelta(days=20)
                         else "right" if monitoring_date > display_end - pd.Timedelta(days=30)
                         else "center"),
                hovertext=("Inicio modelado menos siete días. La fecha se recalcula con "
                           "la información disponible y no acredita una alerta emitida ese día. "
                           "El reloj térmico conserva su origen en el primer pico modelado."),
                captureevents=True,
            )
    cumulative_figure.update_yaxes(title_text="Emergencia acumulada (%)", range=[0, 105])
    return daily_figure, cumulative_figure
