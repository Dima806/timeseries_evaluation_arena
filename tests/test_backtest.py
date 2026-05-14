import pandas as pd
import pytest

from src.evaluation.backtest import expanding_window_backtest
from src.models.naive import SeasonalNaive


@pytest.fixture()
def monthly_series() -> pd.Series:
    index = pd.date_range("2010-01", periods=60, freq="MS")
    values = [float(i % 12 + 1) for i in range(60)]
    return pd.Series(values, index=index, name="value")


def test_backtest_returns_dataframe(monthly_series: pd.Series):
    result = expanding_window_backtest(
        monthly_series,
        model_factory=lambda: SeasonalNaive(period=12),
        n_cutoffs=5,
        horizon=1,
        min_train_size=24,
    )
    assert isinstance(result, pd.DataFrame)


def test_backtest_correct_number_of_rows(monthly_series: pd.Series):
    result = expanding_window_backtest(
        monthly_series,
        model_factory=lambda: SeasonalNaive(period=12),
        n_cutoffs=5,
        horizon=1,
        min_train_size=24,
    )
    assert len(result) == 5


def test_backtest_columns(monthly_series: pd.Series):
    result = expanding_window_backtest(
        monthly_series,
        model_factory=lambda: SeasonalNaive(period=12),
        n_cutoffs=3,
        horizon=1,
        min_train_size=24,
    )
    assert set(result.columns) >= {"cutoff", "rmse", "mae", "decision_cost"}


def test_backtest_rmse_non_negative(monthly_series: pd.Series):
    result = expanding_window_backtest(
        monthly_series,
        model_factory=lambda: SeasonalNaive(period=12),
        n_cutoffs=4,
        horizon=1,
        min_train_size=24,
    )
    assert (result["rmse"] >= 0).all()


def test_backtest_series_too_short_raises():
    short = pd.Series(range(10), dtype=float)
    with pytest.raises(ValueError, match="too short"):
        # min_train_size=9, horizon=1 → max_cutoff=8 < effective_min=9 → raises
        expanding_window_backtest(
            short,
            model_factory=lambda: SeasonalNaive(period=12),
            n_cutoffs=5,
            min_train_size=9,
        )
