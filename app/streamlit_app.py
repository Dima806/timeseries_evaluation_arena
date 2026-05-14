"""Timeseries Evaluation Arena — interactive Streamlit app."""

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.datasets.injector import inject_anomalies
from src.datasets.loader import load_airline, load_electricity, load_retail_base
from src.evaluation.backtest import expanding_window_backtest
from src.evaluation.comparison import single_cutoff_eval
from src.models.arima import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.naive import SeasonalNaive
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.visualisation import plot_backtest_rmse, plot_forecast

st.set_page_config(page_title="Timeseries Evaluation Arena", layout="wide")
st.title("Timeseries Evaluation Arena")
st.caption("Your model is memorising the calendar, not learning the signal.")

cfg = get_settings()

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    dataset_name = st.selectbox(
        "Dataset",
        ["Airline Passengers", "Electricity Production", "Retail Sales"],
    )

    inject = False
    if dataset_name == "Retail Sales":
        inject = st.checkbox("Inject anomalies", value=True)

    selected_models = st.multiselect(
        "Models",
        ["Seasonal Naive", "ARIMA", "Prophet", "XGBoost", "LSTM"],
        default=["Seasonal Naive", "XGBoost"],
    )

    protocol = st.radio("Evaluation protocol", ["A — Single cutoff", "B — Expanding window"])

    cost_under = st.slider("Under-forecast cost", 1.0, 5.0, float(cfg.cost.under), 0.5)
    cost_over = st.slider("Over-forecast cost", 1.0, 5.0, float(cfg.cost.over), 0.5)

# ── Load data ─────────────────────────────────────────────────────────────────


@st.cache_data
def get_series(name: str, do_inject: bool) -> pd.Series:
    if name == "Airline Passengers":
        return load_airline()
    if name == "Electricity Production":
        return load_electricity()
    base = load_retail_base()
    return inject_anomalies(base) if do_inject else base


series = get_series(dataset_name, inject)

# ── Model factories ───────────────────────────────────────────────────────────

PERIOD = 52 if dataset_name == "Retail Sales" else 12

ALL_FACTORIES = {
    "Seasonal Naive": lambda: SeasonalNaive(period=PERIOD),
    "ARIMA": lambda: ARIMAModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
    "Prophet": ProphetModel,
    "XGBoost": lambda: XGBoostModel(
        n_estimators=cfg.models.xgboost.n_estimators,
        max_depth=cfg.models.xgboost.max_depth,
        n_jobs=cfg.models.xgboost.n_jobs,
    ),
    "LSTM": lambda: LSTMModel(
        hidden_size=cfg.models.lstm.hidden_size,
        num_layers=cfg.models.lstm.num_layers,
        epochs=cfg.models.lstm.epochs,
        lr=cfg.models.lstm.lr,
    ),
}

factories = {m: ALL_FACTORIES[m] for m in selected_models if m in ALL_FACTORIES}

if not factories:
    st.warning("Select at least one model.")
    st.stop()

# ── Run evaluation ────────────────────────────────────────────────────────────

use_protocol_b = protocol.startswith("B")
horizon = max(1, len(series) // 5)

results_table: list[dict] = []
backtest_dfs: dict[str, pd.DataFrame] = {}
forecast_series: dict[str, pd.Series] = {}

with st.spinner("Fitting models…"):
    for model_name, factory in factories.items():
        if use_protocol_b:
            df = expanding_window_backtest(
                series,
                factory,
                n_cutoffs=cfg.backtest.n_cutoffs,
                horizon=1,
                min_train_size=cfg.backtest.min_train_size,
                cost_under=cost_under,
                cost_over=cost_over,
            )
            backtest_dfs[model_name] = df
            results_table.append(
                {
                    "Model": model_name,
                    "RMSE (mean)": round(df["rmse"].mean(), 3),
                    "MAE (mean)": round(df["mae"].mean(), 3),
                    "Decision Cost (mean)": round(df["decision_cost"].mean(), 3),
                }
            )
        else:
            metrics = single_cutoff_eval(
                series, factory, cost_under=cost_under, cost_over=cost_over
            )
            results_table.append(
                {
                    "Model": model_name,
                    "RMSE": round(metrics["rmse"], 3),
                    "MAE": round(metrics["mae"], 3),
                    "Decision Cost": round(metrics["decision_cost"], 3),
                }
            )
            # Forecast for plot
            cutoff_idx = int(len(series) * 0.8)
            train = series.iloc[:cutoff_idx]
            m = factory()
            m.fit(train)
            forecast_series[model_name] = m.predict(len(series) - cutoff_idx)

# ── Display results ───────────────────────────────────────────────────────────

st.subheader("Leaderboard")
st.dataframe(pd.DataFrame(results_table).set_index("Model"), use_container_width=True)

if use_protocol_b and backtest_dfs:
    st.subheader("RMSE by cutoff (Protocol B)")
    fig = plot_backtest_rmse(backtest_dfs)
    st.pyplot(fig)
elif forecast_series:
    st.subheader("Forecast vs Actual (Protocol A)")
    fig = plot_forecast(series, forecast_series, title=dataset_name)
    st.pyplot(fig)

st.subheader("Raw series")
st.line_chart(series)
