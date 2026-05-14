# Project: timeseries_evaluation_arena

## Identity

Time series forecasting comparison. Thesis: the model with the lowest RMSE makes the worst business decisions when evaluated with an asymmetric decision-cost metric, and expanding-window backtests are the only honest evaluation.

5 models × 3 datasets × 2 evaluation protocols. 5 notebooks + 1 Streamlit app + `src/` library.

## Stack

- Python 3.11+
- `statsmodels`, `prophet`, `xgboost`, `torch` (CPU-only), `scikit-learn`
- `matplotlib`, `seaborn`, `streamlit`
- `pydantic` / `pydantic-settings` for config from `config/settings.yaml`
- Deps managed with `uv` (pyproject.toml + uv.lock). Never use pip.
- Constraint: 2-CPU / 8 GB GitHub Codespace. `n_jobs=2` max. LSTM: 2 layers, 32 hidden units.

## Repository Structure

```
notebooks/        01_data_exploration, 02_single_cutoff_trap, 03_expanding_window_backtest,
                  04_decision_cost_metric, 05_when_calendar_models_break
src/config.py     Pydantic Settings loaded from config/settings.yaml
src/datasets/     loader.py (airline/electricity/retail), injector.py (anomaly injection)
src/models/       naive.py, arima.py, prophet_model.py, xgboost_model.py, lstm_model.py
src/features/     lag_features.py (lag, rolling mean, calendar features for XGBoost)
src/evaluation/   metrics.py, backtest.py, comparison.py
src/visualisation.py
app/streamlit_app.py
config/settings.yaml   cost ratios, backtest params, model hyperparams
tests/            test_datasets.py, test_models.py, test_metrics.py, test_backtest.py, test_features.py
outputs/figures/
```

## Datasets

| # | Dataset | Frequency | Length | Notes |
|---|---|---|---|---|
| 1 | Airline Passengers (`statsmodels`) | Monthly | 144 pts (1949–1960) | Strong trend + seasonality |
| 2 | Electricity Production (`statsmodels`) | Monthly | 397 pts | Multiple seasonal cycles |
| 3 | Retail Sales (synthetic) | Weekly | 208 pts (4 yrs) | 3 injected anomalies (see below) |

Dataset 3 injected anomalies (known ground truth):
- Week 52: +40% promotional spike for 2 weeks
- Week 130: −15% competitor-driven decline over 20 weeks
- Week 180: 3-week supply disruption + demand surge

## Models

| # | Model | Key strength | Key weakness |
|---|---|---|---|
| 1 | Seasonal Naive | Zero-cost baseline | No trend/shock adaptation |
| 2 | SARIMAX | Interpretable | Slow to adapt to shocks |
| 3 | Prophet | Easy seasonality | Over-trusts calendar patterns |
| 4 | XGBoost + lags | Adapts to recent data | Needs manual feature engineering |
| 5 | LSTM (2L-32H) | No manual features | Slow on CPU, needs more data |

## Evaluation Framework

**Protocol A (wrong):** Single cutoff at 80%. Train before, test after. RMSE/MAE. This is how most tutorials work — demonstrated to be fragile.

**Protocol B (right):** Expanding-window backtest with 12 monthly cutoffs. Retrain at each cutoff, forecast next period, record error. Report RMSE ± std and decision-cost.

**Decision-cost metric (asymmetric):**
```python
def decision_cost(y_true, y_pred, cost_under=3.0, cost_over=1.0):
    errors = y_true - y_pred
    costs = np.where(errors > 0, errors * cost_under, -errors * cost_over)
    return costs.mean()
```
Default ratio 3:1 (under-forecast costs 3× over-forecast). Configurable in `config/settings.yaml`. Explored: 1:1 (≡ MAE), 3:1 (demand), 1:3 (perishable inventory).

**The provocative result:** Prophet wins RMSE on Protocol A for all 3 datasets. Under Protocol B with decision-cost 3:1, it drops to 3rd–4th on datasets with injected shocks because it under-forecasts during anomalies (trusting the calendar over recent observations).

## Makefile Commands

```
make setup      # First-time: install uv + all deps + ipykernel
make sync       # Sync deps from lockfile
make lint       # format + check + typecheck (ruff + ty)
make test       # pytest
make test-cov   # pytest with coverage
make notebooks  # Execute all 5 notebooks (< 5 min each)
make run        # Launch Streamlit app (port 8501)
make lab        # JupyterLab (port 8888)
make clean      # Remove caches
make ci         # sync + lint + test (CI pipeline)
```

## Code Conventions

- Docstrings: Google style, 1-line summary
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`)
- Line length: 99. Linter: ruff. Type checker: ty.
- All code must pass `make lint` before commit.
- Notebooks must run end-to-end in < 5 minutes each on a 2-CPU Codespace.
- No pip. No CUDA. No global state in `src/`.

## Caveman Mode

Active by default. Terse responses. Code-first. Explain only if asked.
`/caveman lite|full|ultra` to change intensity. `stop caveman` to disable.
