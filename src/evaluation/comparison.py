from collections.abc import Callable
from typing import Any

import pandas as pd

from src.evaluation.backtest import expanding_window_backtest
from src.evaluation.metrics import decision_cost, mae, rmse


def single_cutoff_eval(
    series: pd.Series,
    model_factory: Callable[[], Any],
    cutoff_ratio: float = 0.8,
    cost_under: float = 3.0,
    cost_over: float = 1.0,
) -> dict[str, float]:
    """Protocol A: single train/test split at cutoff_ratio."""
    cutoff_idx = int(len(series) * cutoff_ratio)
    train = series.iloc[:cutoff_idx]
    test_values = series.iloc[cutoff_idx:].to_numpy()
    horizon = len(test_values)

    model = model_factory()
    model.fit(train)
    forecast = model.predict(horizon).to_numpy()[: len(test_values)]

    return {
        "rmse": rmse(test_values, forecast),
        "mae": mae(test_values, forecast),
        "decision_cost": decision_cost(test_values, forecast, cost_under, cost_over),
    }


def run_arena(
    datasets: dict[str, pd.Series],
    model_factories: dict[str, Callable[[], Any]],
    n_cutoffs: int = 12,
    cost_under: float = 3.0,
    cost_over: float = 1.0,
) -> pd.DataFrame:
    """Run all models × all datasets × both evaluation protocols."""
    records = []
    for dataset_name, series in datasets.items():
        for model_name, factory in model_factories.items():
            proto_a = single_cutoff_eval(
                series, factory, cost_under=cost_under, cost_over=cost_over
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "protocol": "A",
                    **proto_a,
                }
            )

            proto_b_df = expanding_window_backtest(
                series,
                factory,
                n_cutoffs=n_cutoffs,
                cost_under=cost_under,
                cost_over=cost_over,
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "protocol": "B",
                    "rmse": proto_b_df["rmse"].mean(),
                    "mae": proto_b_df["mae"].mean(),
                    "decision_cost": proto_b_df["decision_cost"].mean(),
                }
            )

    return pd.DataFrame(records)
