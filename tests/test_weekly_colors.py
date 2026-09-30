"""Los colores deben expresar los mismos umbrales y unidades del indicador."""

import numpy as np
import pandas as pd
import pytest

from predweem_twin.charts import trajectory_charts
from predweem_twin.state import weekly_flow_intensity


@pytest.fixture
def reference():
    # 1 % diario durante 100 días: máximo de una semana completa = 7 %.
    return pd.DataFrame({
        "Julian_days": np.arange(1, 366),
        "Progreso_Mediano": np.minimum(np.arange(1, 366) / 100, 1),
        "Campanas_Anos": "prueba",
    })


def frame(total):
    return pd.DataFrame({
        "Fecha": pd.date_range("2027-02-01", periods=7),
        "EMERREL_TWIN": total / 7, "EMERAC_TWIN": .5,
        "EMERAC_NORMALIZADA": .5, "TT_DESDE_PICO": 0.,
    })


@pytest.mark.parametrize("ratio,category,color", [
    (0, "Nula", "#16a34a"), (.0001, "Baja", "#eab308"),
    (.24999, "Baja", "#eab308"), (.25, "Media", "#f97316"),
    (.75, "Media", "#f97316"), (.75001, "Alta", "#dc2626"),
    (1.2, "Alta", "#dc2626"),
])
def test_colors_share_indicator_thresholds_and_fraction_to_percent_units(reference, ratio, category, color):
    data = frame(.07 * ratio)
    before = data.copy(deep=True)
    weekly, cumulative = trajectory_charts(data, None, "2027-02-07", seasonal_reference=reference, flow_frequency="Semanal")
    twin = weekly.data[1]
    assert twin.marker.color[0] == color
    assert twin.customdata[0][4] == f"Intensidad: {category}"
    assert twin.y[0] == pytest.approx(7 * ratio)
    indicator = weekly_flow_intensity(data, "2027-01-31", reference)
    assert indicator["intensity_7d"] == category
    assert indicator["intensity_7d_ratio"] == pytest.approx(ratio)
    assert weekly.data[0].opacity < twin.opacity
    _, daily_cumulative = trajectory_charts(data, None, "2027-02-07", seasonal_reference=reference)
    assert cumulative.to_json() == daily_cumulative.to_json()
    pd.testing.assert_frame_equal(data, before)
    if ratio == 0:
        marks = weekly.data[-1]
        assert marks.name == "Intensidad nula del gemelo"
        assert list(marks.y) == [0.]
        assert marks.marker.color == "#16a34a"


@pytest.mark.parametrize("defect", ["partial", "nan", "duplicate", "negative"])
def test_incomplete_week_is_not_classified_even_if_total_is_large(reference, defect):
    data = frame(.9)
    if defect == "partial":
        data = data.iloc[:6]
    elif defect == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    else:
        data.loc[0, "EMERREL_TWIN"] = np.nan if defect == "nan" else -.1
    weekly, _ = trajectory_charts(data, None, "2027-02-07", seasonal_reference=reference, flow_frequency="Semanal")
    twin = weekly.data[1]
    assert twin.marker.color[0] == "#94a3b8"
    assert twin.marker.pattern.shape[0] == "/"
    assert twin.customdata[0][4] == "Sin clasificación: semana parcial"


def test_positive_without_reference_is_gray_and_partial_zero_has_no_green_mark():
    weekly, _ = trajectory_charts(frame(.07), None, "2027-02-07", flow_frequency="Semanal")
    assert weekly.data[0].marker.color[0] == "#94a3b8"
    assert weekly.data[0].customdata[0][4] == "Sin clasificación: sin referencia"
    weekly, _ = trajectory_charts(frame(0).iloc[:3], None, "2027-02-07", flow_frequency="Semanal")
    assert all(t.type == "bar" for t in weekly.data)
    assert weekly.data[0].marker.color[0] == "#94a3b8"


def test_historical_colors_use_the_same_peak_and_daily_view_stays_blue(reference):
    weekly, _ = trajectory_charts(frame(.07), None, "2027-02-07", seasonal_reference=reference, flow_frequency="Semanal")
    history = weekly.data[0]
    complete_positive = [i for i, y in enumerate(history.y) if y == pytest.approx(7.)]
    assert complete_positive
    assert all(history.marker.color[i] == "#dc2626" for i in complete_positive)
    daily, _ = trajectory_charts(frame(.07), None, "2027-02-07", seasonal_reference=reference)
    assert daily.data[1].marker.color == "#3b82f6"
