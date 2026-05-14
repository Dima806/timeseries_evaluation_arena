"""Model interface tests. Prophet is mocked; LSTM uses minimal config for speed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.naive import SeasonalNaive
from src.models.xgboost_model import XGBoostModel


@pytest.fixture()
def monthly_series() -> pd.Series:
    index = pd.date_range("2018-01", periods=36, freq="MS")
    rng = np.random.default_rng(0)
    values = 100.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, 36)) + rng.normal(0, 2, 36)
    return pd.Series(values, index=index, name="value")


# ── SeasonalNaive ──────────────────────────────────────────────────────────────


def test_naive_predict_length(monthly_series: pd.Series):
    model = SeasonalNaive(period=12)
    model.fit(monthly_series)
    assert len(model.predict(6)) == 6


def test_naive_predict_repeats_season(monthly_series: pd.Series):
    model = SeasonalNaive(period=12)
    model.fit(monthly_series)
    pred = model.predict(12).to_numpy()
    expected = monthly_series.to_numpy()[-12:]
    np.testing.assert_array_almost_equal(pred, expected)


def test_naive_predict_before_fit_raises():
    model = SeasonalNaive()
    with pytest.raises(AssertionError):
        model.predict(1)


# ── XGBoostModel ──────────────────────────────────────────────────────────────


def test_xgboost_fit_predict(monthly_series: pd.Series):
    model = XGBoostModel(n_estimators=10, max_depth=3, lags=[1, 2, 3], rolling_windows=[2])
    model.fit(monthly_series)
    pred = model.predict(3)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 3


def test_xgboost_predict_before_fit_raises():
    model = XGBoostModel()
    with pytest.raises(AssertionError):
        model.predict(1)


# ── ProphetModel (mocked) ─────────────────────────────────────────────────────


def test_prophet_fit_predict_interface(monthly_series: pd.Series):
    with patch("src.models.prophet_model.Prophet") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.predict.return_value = pd.DataFrame({"yhat": [1.0, 2.0, 3.0]})

        from src.models.prophet_model import ProphetModel

        model = ProphetModel()
        model.fit(monthly_series)
        pred = model.predict(3)
        assert len(pred) == 3
        assert isinstance(pred, pd.Series)


def test_prophet_predict_before_fit_raises():
    from src.models.prophet_model import ProphetModel

    model = ProphetModel()
    with pytest.raises(AssertionError):
        model.predict(1)


# ── ARIMAModel ────────────────────────────────────────────────────────────────


def test_arima_fit_predict(monthly_series: pd.Series):
    from src.models.arima import ARIMAModel

    model = ARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    model.fit(monthly_series)
    pred = model.predict(3)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 3


def test_arima_predict_before_fit_raises():
    from src.models.arima import ARIMAModel

    model = ARIMAModel()
    with pytest.raises(AssertionError):
        model.predict(1)


# ── LSTMModel ─────────────────────────────────────────────────────────────────


def test_lstm_fit_predict(monthly_series: pd.Series):
    from src.models.lstm_model import LSTMModel

    model = LSTMModel(hidden_size=4, num_layers=1, epochs=2, seq_len=6)
    model.fit(monthly_series)
    pred = model.predict(3)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 3


def test_lstm_fit_series_shorter_than_seq_len_does_not_crash():
    from src.models.lstm_model import LSTMModel

    # seq_len=20, series only 5 pts → early-exit branch (line 51) is hit
    model = LSTMModel(hidden_size=4, num_layers=1, epochs=1, seq_len=20)
    short = pd.Series(range(5), dtype=float)
    model.fit(short)  # must not raise
    assert model._net is None  # net was never built


def test_lstm_predict_before_fit_raises():
    from src.models.lstm_model import LSTMModel

    model = LSTMModel()
    with pytest.raises(AssertionError):
        model.predict(1)
