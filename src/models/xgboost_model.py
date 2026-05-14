import pandas as pd
from xgboost import XGBRegressor

from src.features.lag_features import make_lag_features


class XGBoostModel:
    """XGBoost regressor with recursive multi-step forecasting via lag features."""

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        n_estimators: int = 200,
        max_depth: int = 3,
        n_jobs: int = 2,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        self.lags: list[int] = lags if lags is not None else [1, 2, 3, 6, 12, 24]
        self.rolling_windows: list[int] = (
            rolling_windows if rolling_windows is not None else [3, 6, 12]
        )
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
        self._series: pd.Series | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> None:
        self._series = series.copy()
        if isinstance(series.index, pd.DatetimeIndex):
            self._freq = pd.infer_freq(series.index)
        feat_df = make_lag_features(series, self.lags, self.rolling_windows)
        x_train = feat_df.drop(columns=["y"])
        y_train = feat_df["y"]
        self._model.fit(x_train, y_train)

    def predict(self, horizon: int) -> pd.Series:
        assert self._series is not None, "Call fit() before predict()."
        series = self._series.copy()
        assert isinstance(series.index, pd.DatetimeIndex)
        forecasts: list[float] = []
        future_dates: list[pd.Timestamp] = []
        freq_offset = pd.tseries.frequencies.to_offset(self._freq) if self._freq else None
        for _ in range(horizon):
            feat_df = make_lag_features(series, self.lags, self.rolling_windows)
            if feat_df.empty:
                break
            x_row = feat_df.drop(columns=["y"]).iloc[[-1]]
            pred = float(self._model.predict(x_row)[0])
            if freq_offset is not None:
                next_dt = series.index[-1] + freq_offset
            else:
                next_dt = series.index[-1] + (series.index[-1] - series.index[-2])
            forecasts.append(pred)
            future_dates.append(next_dt)
            series = pd.concat([series, pd.Series([pred], index=[next_dt], name=series.name)])
        return pd.Series(forecasts, index=future_dates, name="forecast")
