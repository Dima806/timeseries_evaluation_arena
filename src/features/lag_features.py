import pandas as pd


def make_lag_features(
    series: pd.Series,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build lag, rolling-mean, and calendar features from a time series.

    Returns a DataFrame with target column 'y' plus one column per feature.
    Rows with NaN (from shifting) are dropped.
    """
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = series.shift(1).rolling(window).mean()
    if isinstance(series.index, pd.DatetimeIndex):
        dti: pd.DatetimeIndex = series.index
        df["month"] = dti.month  # type: ignore
        df["quarter"] = dti.quarter  # type: ignore
        df["week_of_year"] = dti.isocalendar().week.astype(int)
    return df.dropna()
