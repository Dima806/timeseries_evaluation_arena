from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import decision_cost, mae, rmse


def expanding_window_backtest(
    series: pd.Series,
    model_factory: Callable[[], Any],
    n_cutoffs: int = 12,
    horizon: int = 1,
    min_train_size: int | None = None,
    cost_under: float = 3.0,
    cost_over: float = 1.0,
) -> pd.DataFrame:
    """Expanding-window backtest (Protocol B).

    At each cutoff, trains on all data up to that point and forecasts
    the next `horizon` steps. Returns a DataFrame with one row per cutoff.
    """
    n = len(series)
    effective_min = min_train_size if min_train_size is not None else max(n // 3, 24)
    max_cutoff = n - horizon - 1

    if effective_min > max_cutoff:
        raise ValueError("Series too short for the requested backtest configuration.")

    cutoff_indices = np.linspace(effective_min, max_cutoff, n_cutoffs, dtype=int)

    records = []
    for cutoff_idx in cutoff_indices:
        train = series.iloc[: cutoff_idx + 1]
        actual = series.iloc[cutoff_idx + 1 : cutoff_idx + 1 + horizon].to_numpy()

        model = model_factory()
        model.fit(train)
        raw_forecast = model.predict(horizon)
        forecast = raw_forecast.to_numpy()[: len(actual)]

        records.append(
            {
                "cutoff": series.index[cutoff_idx],
                "rmse": rmse(actual, forecast),
                "mae": mae(actual, forecast),
                "decision_cost": decision_cost(actual, forecast, cost_under, cost_over),
            }
        )

    return pd.DataFrame(records)
