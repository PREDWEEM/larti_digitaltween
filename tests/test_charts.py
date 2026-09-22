"""Límites temporales del fondo histórico frente al pronóstico operativo."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predweem_twin.charts import annual_historical_reference, trajectory_charts
from predweem_twin.seasonal import load_local_seasonal_reference


ROOT = Path(__file__).parents[1]


@pytest.fixture
def reference():
    return load_local_seasonal_reference(ROOT, as_of="2027-05-05")


def trajectory():
    dates = pd.date_range("2027-04-25", "2027-05-20")
    return pd.DataFrame({
        "Fecha": dates,
        "EMERAC_NORMALIZADA": np.linspace(.3, .7, len(dates)),
        "EMERAC_TWIN": np.linspace(.4, .8, len(dates)),
        "EMERREL_TWIN": .016,
        "TT_DESDE_PICO": np.linspace(500, 900, len(dates)),
    })


def test_annual_reference_keeps_unknown_periods_and_observed_2026_window(reference):
    annual = annual_historical_reference(reference, "2027-05-05")
    assert annual.Fecha.tolist() == pd.date_range("2027-01-01", "2027-12-31").tolist()
    assert annual.loc[annual.Fecha.lt("2027-02-01"), "Progreso_2026"].isna().all()
    assert annual.loc[annual.Fecha.gt("2027-08-30"), "Progreso_2026"].isna().all()
    last_day = pd.Timestamp("2027-01-01") + pd.Timedelta(days=reference.Julian_days.max() - 1)
    outside = annual.Fecha.gt(last_day)
    assert annual.loc[outside, ["Progreso_Mediano", "Flujo_Diario"]].isna().all().all()
    assert annual.Flujo_Diario.sum() == pytest.approx(1.)


def test_reference_preserves_month_day_in_leap_year(reference):
    normal = annual_historical_reference(reference, "2027-05-05").set_index("Fecha")
    leap = annual_historical_reference(reference, "2028-05-05").set_index("Fecha")
    assert leap.loc["2028-03-01", "Progreso_Mediano"] == normal.loc["2027-03-01", "Progreso_Mediano"]
    assert leap.loc["2028-02-29", "Progreso_Mediano"] == pytest.approx(
        (normal.loc["2027-02-28", "Progreso_Mediano"] + normal.loc["2027-03-01", "Progreso_Mediano"]) / 2
    )


def test_chart_shows_annual_context_without_extending_weather_or_changing_state(reference):
    frame = trajectory()
    before = frame.copy(deep=True)
    observations = pd.DataFrame({
        "Fecha": pd.to_datetime(["2027-05-03", "2027-05-18"]), "Observado": [.5, .9]
    })
    daily, cumulative = trajectory_charts(frame, observations, "2027-05-05", seasonal_reference=reference)
    pd.testing.assert_frame_equal(frame, before)
    for fig in (daily, cumulative):
        assert pd.Timestamp(fig.layout.xaxis.range[1]) == pd.Timestamp("2027-10-01")
        assert "yaxis2" not in fig.layout
    assert daily.layout.xaxis.range == cumulative.layout.xaxis.range
    assert all(trace.type == "bar" for trace in daily.data)
    assert all(trace.type == "scatter" for trace in cumulative.data)
    traces = {trace.name: trace for trace in (*daily.data, *cumulative.data)}
    history = traces["Pool histórico · orientativo"]
    assert pd.to_datetime(history.x)[np.isfinite(history.y)].max() > pd.Timestamp("2027-05-12")
    assert pd.to_datetime(traces["Estado actualizado"].x).max() == pd.Timestamp("2027-05-05")
    assert pd.to_datetime(traces["Proyección meteorológica · hasta 7 días"].x).max() == pd.Timestamp("2027-05-12")
    assert pd.to_datetime(traces["Conteo de campo"].x).max() == pd.Timestamp("2027-05-03")
    for trace in (*daily.data, *cumulative.data):
        assert trace.yaxis in (None, "y")
        assert pd.to_datetime(trace.x).max() <= pd.Timestamp("2027-10-01")
        if "histórico" not in trace.name.casefold():
            assert pd.to_datetime(trace.x).max() <= pd.Timestamp("2027-05-12")
    for fig in (daily, cumulative):
        assert "Sin referencia disponible" not in [item.text for item in fig.layout.annotations]


def test_historical_backdrop_does_not_leak_2026_into_earlier_cutoffs():
    ref = load_local_seasonal_reference(ROOT, as_of="2026-05-05")
    annual = annual_historical_reference(ref, "2026-05-05")
    assert "Progreso_2026" not in annual
    assert annual.attrs["campaigns"] == "2008, 2009, 2011, 2012, 2013, 2014, 2023, 2024"


def test_no_forecast_trace_when_weather_ends_at_cutoff(reference):
    frame = trajectory().loc[lambda data: data.Fecha.le("2027-05-05")]
    figures = trajectory_charts(frame, None, "2027-05-05", seasonal_reference=reference)
    for fig in figures:
        assert not any("Proyección meteorológica" in trace.name for trace in fig.data)
        assert not any(item.text == "Pronóstico 7 d" for item in fig.layout.annotations)


def test_field_audit_is_only_in_cumulative_chart_and_stops_at_cutoff(reference):
    audit = pd.DataFrame({
        "Fecha_asimilada": pd.to_datetime(["2027-05-03", "2027-05-18"]),
        "Estado_campo_estimado": [.5, .9],
    })
    daily, cumulative = trajectory_charts(trajectory(), None, "2027-05-05", audit, seasonal_reference=reference)
    assert not any(trace.name == "Estado estimado desde campo" for trace in daily.data)
    field = next(trace for trace in cumulative.data if trace.name == "Estado estimado desde campo")
    assert field.yaxis in (None, "y")
    assert pd.to_datetime(field.x).max() == pd.Timestamp("2027-05-03")


def test_weekly_flows_conserve_totals_use_common_weeks_and_preserve_cumulative(reference):
    frame = trajectory()
    # Una señal grande fuera del horizonte no debe contaminar la última semana.
    frame.loc[frame.Fecha.gt("2027-05-12"), "EMERREL_TWIN"] = .5
    daily, daily_cumulative = trajectory_charts(
        frame, None, "2027-05-05", seasonal_reference=reference,
    )
    weekly, weekly_cumulative = trajectory_charts(
        frame, None, "2027-05-05", seasonal_reference=reference, flow_frequency="Semanal",
    )
    assert weekly_cumulative.to_json() == daily_cumulative.to_json()
    for daily_trace, weekly_trace in zip(daily.data, weekly.data):
        assert np.nansum(weekly_trace.y) == pytest.approx(np.nansum(daily_trace.y))
    history, twin = weekly.data
    assert set(twin.x).issubset(set(history.x))
    assert list(pd.to_datetime(twin.x)) == list(pd.to_datetime([
        "2027-04-19", "2027-04-26", "2027-05-03", "2027-05-10",
    ]))
    assert list(twin.y) == pytest.approx([1.6, 11.2, 11.2, 4.8])
    assert list(twin.marker.pattern.shape) == ["/", "", "", "/"]
    assert "3/7 días" in twin.customdata[-1][1]
    assert "10/05–12/05" in twin.customdata[-1][2]
    assert "3 día(s) de proyección" in twin.customdata[-1][3]
    assert weekly.layout.yaxis.ticksuffix == " %"


def test_weekly_chart_keeps_missing_flows_unknown_and_clips_october_boundary():
    dates = pd.date_range("2027-09-20", "2027-10-05")
    frame = pd.DataFrame({
        "Fecha": dates, "EMERREL_TWIN": .01,
        "EMERAC_TWIN": .5, "EMERAC_NORMALIZADA": .5, "TT_DESDE_PICO": 0.,
    })
    frame.loc[frame.Fecha.lt("2027-09-27"), "EMERREL_TWIN"] = np.nan
    # Fuera del eje: ni se dibuja ni se suma a la semana que contiene el 1/10.
    frame.loc[frame.Fecha.gt("2027-10-01"), "EMERREL_TWIN"] = .9
    weekly, _ = trajectory_charts(frame, None, "2027-09-29", flow_frequency="Semanal")
    twin = weekly.data[0]
    assert np.isnan(twin.y[0])
    assert twin.y[1] == pytest.approx(5.)
    assert "5/7 días" in twin.customdata[1][1]
    assert "27/09–01/10" in twin.customdata[1][2]
    assert pd.Timestamp(weekly.layout.xaxis.range[1]) == pd.Timestamp("2027-10-01")
