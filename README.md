# timeseries_evaluation_arena

> Your time series model is memorising the calendar, not learning the signal.

Head-to-head comparison of five forecasting methods on three datasets, showing that the model with the lowest RMSE makes the worst business decisions under an asymmetric decision-cost metric.

## Quick start

```bash
git clone <repo>
make setup   # install uv + deps
make lab     # JupyterLab
make run     # Streamlit app
```

## Commands

| Command | Description |
|---|---|
| `make setup` | First-time setup |
| `make test` | Run test suite |
| `make lint` | Ruff + ty |
| `make notebooks` | Execute all notebooks |
| `make run` | Streamlit app (port 8501) |
| `make ci` | Full CI pipeline |
