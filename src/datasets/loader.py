import numpy as np
import pandas as pd


def load_airline() -> pd.Series:
    """Monthly airline passengers 1949-1960 (144 pts). Box-Jenkins classic."""
    values = np.array(
        [
            112,
            118,
            132,
            129,
            121,
            135,
            148,
            148,
            136,
            119,
            104,
            118,
            115,
            126,
            141,
            135,
            125,
            149,
            170,
            170,
            158,
            133,
            114,
            140,
            145,
            150,
            178,
            163,
            172,
            178,
            199,
            199,
            184,
            162,
            146,
            166,
            171,
            180,
            193,
            181,
            183,
            218,
            230,
            242,
            209,
            191,
            172,
            194,
            196,
            196,
            236,
            235,
            229,
            243,
            264,
            272,
            237,
            211,
            180,
            201,
            204,
            188,
            235,
            227,
            234,
            264,
            302,
            293,
            259,
            229,
            203,
            229,
            242,
            233,
            267,
            269,
            270,
            315,
            364,
            347,
            312,
            274,
            237,
            278,
            284,
            277,
            317,
            313,
            318,
            374,
            413,
            405,
            355,
            306,
            271,
            306,
            315,
            301,
            356,
            348,
            355,
            422,
            465,
            467,
            404,
            347,
            305,
            336,
            340,
            318,
            362,
            348,
            363,
            435,
            491,
            505,
            404,
            359,
            310,
            337,
            360,
            342,
            406,
            396,
            420,
            472,
            548,
            559,
            463,
            407,
            362,
            405,
            417,
            391,
            419,
            461,
            472,
            535,
            622,
            606,
            508,
            461,
            390,
            432,
        ],
        dtype=float,
    )
    index = pd.date_range("1949-01", periods=144, freq="MS")
    return pd.Series(values, index=index, name="airline_passengers")


def load_electricity(n: int = 397, seed: int = 42) -> pd.Series:
    """Synthetic monthly electricity production with trend + dual seasonality (397 pts)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 100.0 + 0.15 * t
    seasonal = 20.0 * np.sin(2 * np.pi * t / 12) + 8.0 * np.sin(2 * np.pi * t / 6)
    trend_shift = np.where(t > 200, 0.05 * (t - 200), 0.0)
    noise = rng.normal(0, 3, n)
    values = trend + seasonal + trend_shift + noise
    index = pd.date_range("1990-01", periods=n, freq="MS")
    return pd.Series(values, index=index, name="electricity_production")


def load_retail_base(n: int = 208, seed: int = 0) -> pd.Series:
    """Synthetic weekly retail sales base series (208 pts = 4 years), no anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 500.0 + 0.5 * t
    seasonal = 50.0 * np.sin(2 * np.pi * t / 52) + 20.0 * np.sin(2 * np.pi * t / 13)
    noise = rng.normal(0, 15, n)
    values = np.clip(trend + seasonal + noise, 0.0, None)
    index = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    return pd.Series(values, index=index, name="retail_sales")
