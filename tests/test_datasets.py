import pandas as pd
import pytest

from src.datasets.injector import inject_anomalies
from src.datasets.loader import load_airline, load_electricity, load_retail_base


def test_airline_shape_and_index():
    s = load_airline()
    assert isinstance(s, pd.Series)
    assert len(s) == 144
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.freq is not None or str(s.index[1] - s.index[0]) != "0 days"
    assert s.name == "airline_passengers"


def test_airline_first_value():
    s = load_airline()
    assert s.iloc[0] == 112.0


def test_electricity_shape():
    s = load_electricity()
    assert isinstance(s, pd.Series)
    assert len(s) == 397
    assert isinstance(s.index, pd.DatetimeIndex)


def test_electricity_custom_length():
    s = load_electricity(n=100)
    assert len(s) == 100


def test_retail_base_shape():
    s = load_retail_base()
    assert isinstance(s, pd.Series)
    assert len(s) == 208
    assert (s >= 0).all()


def test_inject_anomalies_changes_values():
    base = load_retail_base()
    injected = inject_anomalies(base)
    assert not base.equals(injected)


def test_inject_anomalies_promo_spike():
    base = load_retail_base()
    injected = inject_anomalies(base)
    # Promo spike at week 52 should be higher than base
    assert injected.iloc[52] > base.iloc[52]


def test_inject_anomalies_supply_disruption():
    base = load_retail_base()
    injected = inject_anomalies(base)
    # Supply disruption: 3 weeks of zero
    assert injected.iloc[180] == pytest.approx(0.0)
    assert injected.iloc[181] == pytest.approx(0.0)
    assert injected.iloc[182] == pytest.approx(0.0)


def test_inject_anomalies_preserves_length():
    base = load_retail_base()
    injected = inject_anomalies(base)
    assert len(injected) == len(base)
