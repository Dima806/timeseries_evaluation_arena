import pandas as pd
from prophet import Prophet


class ProphetModel:
    """Prophet wrapper with a consistent fit/predict interface."""

    def __init__(self) -> None:
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> None:
        df = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(df)
        self._model = model
        assert isinstance(series.index, pd.DatetimeIndex)
        self._last_date = series.index[-1]
        inferred = pd.infer_freq(series.index)
        self._freq = inferred if inferred is not None else "MS"

    def predict(self, horizon: int) -> pd.Series:
        assert self._model is not None, "Call fit() before predict()."
        assert self._last_date is not None
        assert self._freq is not None
        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        future_df = pd.DataFrame({"ds": future_dates})
        forecast = self._model.predict(future_df)
        return pd.Series(forecast["yhat"].to_numpy(), index=future_dates, name="forecast")
