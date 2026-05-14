import pandas as pd
import pytest

from src.features.lag_features import make_lag_features


@pytest.fixture()
def monthly_series() -> pd.Series:
    index = pd.date_range("2020-01", periods=36, freq="MS")
    return pd.Series(range(36), index=index, dtype=float, name="value")


def test_lag_columns_created(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[1, 2], rolling_windows=[])
    assert "lag_1" in df.columns
    assert "lag_2" in df.columns


def test_rolling_columns_created(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[], rolling_windows=[3])
    assert "rolling_mean_3" in df.columns


def test_calendar_columns_created(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[1], rolling_windows=[])
    assert "month" in df.columns
    assert "quarter" in df.columns


def test_no_nan_rows(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[1, 2, 3], rolling_windows=[3])
    assert not df.isnull().any().any()


def test_target_column_present(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[1], rolling_windows=[])
    assert "y" in df.columns


def test_lag_values_correct(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[1], rolling_windows=[])
    # For row at index t, lag_1 should equal y at t-1
    row = df.iloc[0]
    assert row["y"] == row["lag_1"] + 1


def test_empty_lags_and_windows(monthly_series: pd.Series):
    df = make_lag_features(monthly_series, lags=[], rolling_windows=[])
    assert len(df) == len(monthly_series)
    assert list(df.columns) == ["y", "month", "quarter", "week_of_year"]
