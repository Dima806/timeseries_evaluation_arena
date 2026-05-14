import numpy as np
import pandas as pd


class SeasonalNaive:
    """Seasonal naive forecaster: repeats the last complete seasonal cycle."""

    def __init__(self, period: int = 12) -> None:
        self.period = period
        self._last_season: np.ndarray | None = None

    def fit(self, series: pd.Series) -> None:
        values = series.to_numpy(dtype=float)
        self._last_season = values[-self.period :]

    def predict(self, horizon: int) -> pd.Series:
        assert self._last_season is not None, "Call fit() before predict()."
        reps = -(-horizon // self.period)  # ceiling division
        forecasts = np.tile(self._last_season, reps)[:horizon]
        return pd.Series(forecasts, name="forecast")
