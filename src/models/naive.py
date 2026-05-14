import numpy as np
import pandas as pd


class SeasonalNaive:
    """Seasonal naive forecaster: repeats the last complete seasonal cycle."""

    def __init__(self, period: int = 12) -> None:
        self.period = period
        self._last_season: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> None:
        values = series.to_numpy(dtype=float)
        self._last_season = values[-self.period :]
        if isinstance(series.index, pd.DatetimeIndex):
            self._last_date = series.index[-1]
            self._freq = pd.infer_freq(series.index) or "MS"

    def predict(self, horizon: int) -> pd.Series:
        assert self._last_season is not None, "Call fit() before predict()."
        reps = -(-horizon // self.period)  # ceiling division
        forecasts = np.tile(self._last_season, reps)[:horizon]
        if self._last_date is not None and self._freq is not None:
            idx = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        else:
            idx = None
        return pd.Series(forecasts, index=idx, name="forecast")
