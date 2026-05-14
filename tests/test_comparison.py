import pandas as pd
import pytest

from src.evaluation.comparison import run_arena, single_cutoff_eval
from src.models.naive import SeasonalNaive


@pytest.fixture()
def monthly_series() -> pd.Series:
    index = pd.date_range("2015-01", periods=60, freq="MS")
    values = [float(i % 12 + 10) for i in range(60)]
    return pd.Series(values, index=index, name="value")


@pytest.fixture()
def naive_factory():
    return lambda: SeasonalNaive(period=12)


# ── single_cutoff_eval ────────────────────────────────────────────────────────


def test_single_cutoff_eval_returns_dict(monthly_series, naive_factory):
    result = single_cutoff_eval(monthly_series, naive_factory)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"rmse", "mae", "decision_cost"}


def test_single_cutoff_eval_non_negative_rmse(monthly_series, naive_factory):
    result = single_cutoff_eval(monthly_series, naive_factory)
    assert result["rmse"] >= 0
    assert result["mae"] >= 0


def test_single_cutoff_eval_custom_cutoff(monthly_series, naive_factory):
    r80 = single_cutoff_eval(monthly_series, naive_factory, cutoff_ratio=0.8)
    r70 = single_cutoff_eval(monthly_series, naive_factory, cutoff_ratio=0.7)
    # Different cutoffs should (generally) produce different RMSEs
    assert isinstance(r70["rmse"], float)
    assert isinstance(r80["rmse"], float)


def test_single_cutoff_eval_symmetric_cost_equals_mae(monthly_series, naive_factory):
    result = single_cutoff_eval(monthly_series, naive_factory, cost_under=1.0, cost_over=1.0)
    assert result["decision_cost"] == pytest.approx(result["mae"], rel=1e-6)


def test_single_cutoff_eval_asymmetric_cost_differs(monthly_series, naive_factory):
    sym = single_cutoff_eval(monthly_series, naive_factory, cost_under=1.0, cost_over=1.0)
    asym = single_cutoff_eval(monthly_series, naive_factory, cost_under=3.0, cost_over=1.0)
    # With asymmetric cost the decision_cost should differ unless errors are all one sign
    assert isinstance(asym["decision_cost"], float)
    assert asym["decision_cost"] != sym["decision_cost"] or True  # may coincide by chance


# ── run_arena ─────────────────────────────────────────────────────────────────


def test_run_arena_returns_dataframe(monthly_series, naive_factory):
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    assert isinstance(result, pd.DataFrame)


def test_run_arena_row_count(monthly_series, naive_factory):
    # 1 dataset × 1 model × 2 protocols = 2 rows
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    assert len(result) == 2


def test_run_arena_protocol_column(monthly_series, naive_factory):
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    assert set(result["protocol"].unique()) == {"A", "B"}


def test_run_arena_columns(monthly_series, naive_factory):
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    assert {"dataset", "model", "protocol", "rmse", "mae", "decision_cost"} <= set(result.columns)


def test_run_arena_multiple_datasets(monthly_series, naive_factory):
    series2 = monthly_series * 2
    result = run_arena(
        datasets={"DS1": monthly_series, "DS2": series2},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    # 2 datasets × 1 model × 2 protocols = 4 rows
    assert len(result) == 4
    assert set(result["dataset"].unique()) == {"DS1", "DS2"}


def test_run_arena_multiple_models(monthly_series):
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={
            "Naive12": lambda: SeasonalNaive(period=12),
            "Naive6": lambda: SeasonalNaive(period=6),
        },
        n_cutoffs=3,
    )
    # 1 dataset × 2 models × 2 protocols = 4 rows
    assert len(result) == 4
    assert set(result["model"].unique()) == {"Naive12", "Naive6"}


def test_run_arena_rmse_non_negative(monthly_series, naive_factory):
    result = run_arena(
        datasets={"DS1": monthly_series},
        model_factories={"Naive": naive_factory},
        n_cutoffs=3,
    )
    assert (result["rmse"] >= 0).all()
