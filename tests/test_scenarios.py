import pandas as pd

from predweem_twin.scenarios import apply_scenario


def test_scenario_changes_only_future_weather():
    weather = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-04-01", periods=5),
            "TMAX": [20.0] * 5,
            "TMIN": [10.0] * 5,
            "Prec": [0.0] * 5,
        }
    )
    result = apply_scenario(weather, "2026-04-02", rain_mm=30, rain_days=3, temperature_delta=2)
    assert result.loc[:1, "Prec"].sum() == 0
    assert result.loc[2:, "Prec"].sum() == 30
    assert (result.loc[:1, "TMAX"] == 20).all()
    assert (result.loc[2:, "TMAX"] == 22).all()
