import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_forecast(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    title: str = "Forecast Comparison",
    n_history: int = 60,
) -> Figure:
    """Plot actual series (last n_history points) with overlaid model forecasts."""
    fig, ax = plt.subplots(figsize=(12, 5))
    tail = actual.iloc[-n_history:]
    ax.plot(tail.index, tail.values, label="Actual", color="black", linewidth=2)
    for model_name, forecast in forecasts.items():
        ax.plot(forecast.index, forecast.values, label=model_name, linestyle="--")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metric_comparison(
    results: pd.DataFrame,
    metric: str = "rmse",
    title: str | None = None,
) -> Figure:
    """Bar chart comparing models across datasets and protocols."""
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = results.pivot_table(index="model", columns=["dataset", "protocol"], values=metric)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title or f"Model Comparison — {metric.upper()}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Dataset / Protocol", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_backtest_rmse(
    backtest_results: dict[str, pd.DataFrame],
    title: str = "Expanding-Window RMSE by Cutoff",
) -> Figure:
    """Line plot of per-cutoff RMSE for each model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for model_name, df in backtest_results.items():
        ax.plot(df["cutoff"], df["rmse"], marker="o", label=model_name)
    ax.set_title(title)
    ax.set_xlabel("Cutoff date")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
