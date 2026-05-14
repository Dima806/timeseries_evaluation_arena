import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualisation import plot_backtest_rmse, plot_forecast, plot_metric_comparison


@pytest.fixture()
def actual_series() -> pd.Series:
    index = pd.date_range("2020-01", periods=30, freq="MS")
    return pd.Series(range(30), index=index, dtype=float, name="actual")


@pytest.fixture()
def forecast_index() -> pd.DatetimeIndex:
    return pd.date_range("2022-07", periods=5, freq="MS")


@pytest.fixture()
def forecasts(forecast_index) -> dict[str, pd.Series]:
    return {
        "ModelA": pd.Series([30.0, 31.0, 32.0, 33.0, 34.0], index=forecast_index),
        "ModelB": pd.Series([28.0, 29.0, 30.0, 31.0, 32.0], index=forecast_index),
    }


@pytest.fixture()
def comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Naive", "Naive", "XGB", "XGB"],
            "dataset": ["DS1", "DS1", "DS1", "DS1"],
            "protocol": ["A", "B", "A", "B"],
            "rmse": [5.0, 6.0, 4.0, 4.5],
            "mae": [4.0, 5.0, 3.0, 3.5],
        }
    )


@pytest.fixture()
def backtest_dfs() -> dict[str, pd.DataFrame]:
    cutoffs = pd.date_range("2021-01", periods=4, freq="MS")
    return {
        "Naive": pd.DataFrame({"cutoff": cutoffs, "rmse": [5.0, 6.0, 4.5, 5.5]}),
        "XGB": pd.DataFrame({"cutoff": cutoffs, "rmse": [4.0, 5.0, 3.5, 4.5]}),
    }


# ── plot_forecast ─────────────────────────────────────────────────────────────


def test_plot_forecast_returns_figure(actual_series, forecasts):
    fig = plot_forecast(actual_series, forecasts)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_forecast_custom_title(actual_series, forecasts):
    fig = plot_forecast(actual_series, forecasts, title="My Title")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_forecast_empty_forecasts(actual_series):
    fig = plot_forecast(actual_series, {})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_forecast_n_history_clips_series(actual_series, forecasts):
    fig = plot_forecast(actual_series, forecasts, n_history=5)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_forecast_single_model(actual_series, forecast_index):
    fig = plot_forecast(
        actual_series,
        {"OnlyModel": pd.Series([1.0, 2.0], index=forecast_index[:2])},
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# ── plot_metric_comparison ────────────────────────────────────────────────────


def test_plot_metric_comparison_returns_figure(comparison_df):
    fig = plot_metric_comparison(comparison_df)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_metric_comparison_custom_metric(comparison_df):
    fig = plot_metric_comparison(comparison_df, metric="mae")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_metric_comparison_custom_title(comparison_df):
    fig = plot_metric_comparison(comparison_df, title="Custom")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_metric_comparison_default_title(comparison_df):
    fig = plot_metric_comparison(comparison_df, metric="rmse", title=None)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ── plot_backtest_rmse ────────────────────────────────────────────────────────


def test_plot_backtest_rmse_returns_figure(backtest_dfs):
    fig = plot_backtest_rmse(backtest_dfs)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_backtest_rmse_custom_title(backtest_dfs):
    fig = plot_backtest_rmse(backtest_dfs, title="Custom RMSE")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_backtest_rmse_single_model(backtest_dfs):
    fig = plot_backtest_rmse({"Naive": backtest_dfs["Naive"]})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_backtest_rmse_empty_dict():
    fig = plot_backtest_rmse({})
    assert isinstance(fig, Figure)
    plt.close(fig)
