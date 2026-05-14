from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """SARIMAX wrapper with a consistent fit/predict interface."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._result: Any = None

    def fit(self, series: pd.Series) -> None:
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)

    def predict(self, horizon: int) -> pd.Series:
        assert self._result is not None, "Call fit() before predict()."
        forecast = self._result.forecast(steps=horizon)
        return forecast.rename("forecast")
