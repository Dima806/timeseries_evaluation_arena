import pandas as pd


def inject_anomalies(series: pd.Series) -> pd.Series:
    """Inject 3 known anomalies into the retail sales base series.

    Anomalies:
      - Week 52: +40% promotional spike for 2 weeks.
      - Week 130: -15% competitor decline over 20 weeks.
      - Week 180: supply disruption (3 weeks zero) then demand surge.
    """
    s = series.copy()
    # Promotional spike
    s.iloc[52:54] = s.iloc[52:54] * 1.4
    # Competitor decline: linear ramp down over 20 weeks
    for i in range(20):
        s.iloc[130 + i] = s.iloc[130 + i] * (1.0 - 0.15 * (i + 1) / 20)
    # Supply disruption: 3 weeks zero, then 3-week demand surge
    s.iloc[180:183] = 0.0
    s.iloc[183:186] = s.iloc[183:186] * 1.5
    return s


ANOMALY_EVENTS: list[dict[str, object]] = [
    {"label": "Promo spike", "start": 52, "end": 54},
    {"label": "Competitor decline", "start": 130, "end": 150},
    {"label": "Supply disruption", "start": 180, "end": 186},
]
