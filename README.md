# timeseries_evaluation_arena

> Your time series model is memorising the calendar, not learning the signal.

Head-to-head comparison of five forecasting methods on three datasets, demonstrating that:
1. The model with the lowest RMSE on a single cutoff makes the worst business decisions under an asymmetric cost metric.
2. Expanding-window backtests reveal variance that single-cutoff evaluation hides entirely.
3. RMSE rank and decision-cost rank can be inverted — sometimes dramatically.

## Quick start

```bash
git clone https://github.com/Dima806/timeseries_evaluation_arena.git
make setup      # install uv + deps
make notebooks  # run all 5 notebooks
make run        # Streamlit app (port 8501)
make lab        # JupyterLab (port 8888)
```

## Commands

| Command | Description |
|---|---|
| `make setup` | First-time setup |
| `make test` | Run test suite (99% coverage) |
| `make lint` | Ruff + ty |
| `make notebooks` | Execute all 5 notebooks end-to-end |
| `make run` | Streamlit app (port 8501) |
| `make ci` | Full CI pipeline |

## Datasets

| Dataset | Frequency | Length | Notes |
|---|---|---|---|
| Airline Passengers | Monthly | 144 pts (1949–1960) | Strong trend + 12-month seasonality |
| Electricity Production | Monthly | 397 pts | Multiple seasonal cycles + trend break at ~month 200 |
| Retail Sales (synthetic) | Weekly | 208 pts (4 yrs) | 3 injected anomalies at known timestamps |

## Models

| Model | Hyperparameters |
|---|---|
| Seasonal Naive | period=12 (monthly), period=52 (weekly) |
| SARIMAX | order=(1,1,1), seasonal_order=(1,1,1,12) for monthly |
| Prophet | yearly_seasonality=True |
| XGBoost | max_depth=3, lr=0.05, subsample=0.8, lags=[1,2,3,6,12,24] (monthly) |
| LSTM | 2 layers, 32 hidden units, 50 epochs, seq_len=12 |

## Results

### Protocol A — single 80% cutoff RMSE

| Model | Airline | Electricity | Retail |
|---|---|---|---|
| Naive | 75.23 | 9.71 | 177.75 |
| ARIMA | **31.79** | **3.20** | 185.90 |
| Prophet | 41.33 | 3.35 | **176.08** |
| XGBoost | 75.75 | 15.38 | 179.91 |

### Protocol B — expanding-window mean RMSE ± std (6 cutoffs)

| Model | Airline | Electricity | Retail |
|---|---|---|---|
| Naive | 30.0 ± 15.2 | 6.2 ± 4.0 | 34.6 ± 20.2 |
| ARIMA | **8.4 ± 5.7** | 5.1 ± 1.7 | **16.5 ± 6.7** |
| Prophet | 24.3 ± 17.9 | **4.3 ± 2.2** | 21.0 ± 15.6 |
| XGBoost | 15.4 ± 15.6 | 19.8 ± 2.5 | **16.5 ± 3.6** |

### Protocol B — mean decision cost at 3:1 under/over ratio

| Model | Airline | Electricity | Retail |
|---|---|---|---|
| Naive | 90.0 | 18.6 | 82.2 |
| ARIMA | **12.9** | 14.1 | **27.9** |
| Prophet | 41.8 | **12.5** | 38.5 |
| XGBoost | 32.5 | 59.4 | 35.4 |

### Key findings

- **Protocol A vs B:** Rankings change substantially between single-cutoff and expanding-window evaluation. ARIMA, which looks mediocre on Protocol A (Airline: 31.79), becomes the best Protocol B model there (8.4 ± 5.7).
- **RMSE vs decision cost:** XGBoost achieves competitive RMSE on Airline and Retail under Protocol B, but its decision cost on Electricity (59.4) is 5× worse than Prophet (12.5). It systematically under-forecasts around the electricity trend break — lag features cannot model a structural shift they have never seen in training.
- **XGBoost vs Naive on Electricity:** XGBoost is consistently worse than Naive on Protocol B (19.8 vs 6.2 mean RMSE). This is structural, not a hyperparameter problem: the synthetic electricity dataset has a trend break at month ~200, and lag-based models cannot extrapolate beyond the training distribution.
- **Stability matters:** ARIMA on Electricity has the lowest variance (std=1.7, CV=33%) among non-naive models — reliable across all cutoffs. XGBoost on Airline has CV=101%, meaning its error is as large as its mean — high-risk in production.

## Notebooks

| # | Title | Key question |
|---|---|---|
| 01 | Data Exploration | What do the datasets look like? Where are the anomalies? |
| 02 | The Single-Cutoff Trap | Shift the cutoff by 5% — how much do rankings change? |
| 03 | Expanding-Window Backtest | Which models are stable across cutoffs vs lucky on one? |
| 04 | Decision Cost Metric | Which model minimises business cost, not just RMSE? |
| 05 | When Calendar Models Break | Why does Prophet fail during injected demand shocks? |

## Stack

Python 3.11+ · `statsmodels` · `prophet` · `xgboost` · `torch` (CPU) · `streamlit` · `pydantic` · `uv`
