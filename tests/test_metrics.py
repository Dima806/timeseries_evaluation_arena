import numpy as np
import pytest

from src.evaluation.metrics import decision_cost, mae, mape, rmse


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # errors: 3, 4 → sqrt((9+16)/2) = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_mape_known():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # errors: 10%, 10% → 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_skips_zeros():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([50.0, 110.0])
    # Only the second pair matters: 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_all_zeros_returns_nan():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert np.isnan(mape(y_true, y_pred))


def test_decision_cost_symmetric_equals_mae():
    y_true = np.array([1.0, 3.0])
    y_pred = np.array([2.0, 2.0])
    # errors: -1, +1 → symmetric cost == mae == 1.0
    assert decision_cost(y_true, y_pred, cost_under=1.0, cost_over=1.0) == pytest.approx(1.0)


def test_decision_cost_under_more_expensive():
    y_true = np.array([10.0])
    y_pred = np.array([7.0])
    # under-forecast by 3 units, cost_under=3 → cost = 9.0
    assert decision_cost(y_true, y_pred, cost_under=3.0, cost_over=1.0) == pytest.approx(9.0)


def test_decision_cost_over_forecast():
    y_true = np.array([7.0])
    y_pred = np.array([10.0])
    # over-forecast by 3 units, cost_over=1 → cost = 3.0
    assert decision_cost(y_true, y_pred, cost_under=3.0, cost_over=1.0) == pytest.approx(3.0)
