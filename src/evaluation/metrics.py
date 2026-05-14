import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (skips zero actuals). Returns percent."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def decision_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_under: float = 3.0,
    cost_over: float = 1.0,
) -> float:
    """Asymmetric decision cost. Under-forecasting costs cost_under per unit."""
    errors = y_true - y_pred
    costs = np.where(errors > 0, errors * cost_under, -errors * cost_over)
    return float(costs.mean())
