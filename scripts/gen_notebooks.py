"""Generate all five project notebooks programmatically."""

from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.12.0"},
}

SETUP = """\
import json
import logging
import warnings
from pathlib import Path

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams.update({"figure.dpi": 100})

FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    print(f"Saved figure → {path}")


def save_json(data, name: str) -> None:
    path = RESULTS_DIR / name
    if isinstance(data, pd.DataFrame):
        payload = data.to_dict(orient="records")
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved results → {path}")


print("Ready.")
"""

LOAD_DATASETS = """\
from src.datasets.loader import load_airline, load_electricity, load_retail_base
from src.datasets.injector import inject_anomalies, ANOMALY_EVENTS

airline     = load_airline()
electricity = load_electricity()
retail_base = load_retail_base()
retail      = inject_anomalies(retail_base)

for name, s in [("Airline", airline), ("Electricity", electricity), ("Retail", retail)]:
    print(f"{name:15s}  {len(s):4d} pts  "
          f"{s.index[0].date()} → {s.index[-1].date()}  mean={s.mean():.1f}")
"""

MODEL_FACTORIES_CODE = """\
from src.models.naive import SeasonalNaive
from src.models.arima import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

# LSTM uses reduced config for notebook speed; set INCLUDE_LSTM=True to enable
INCLUDE_LSTM = False


def make_factories(period: int = 12) -> dict:
    # Period-aware lag sets: monthly captures 2-year cycle; weekly captures annual (52w)
    if period == 52:
        lags = [1, 2, 3, 4, 13, 26]   # no lag-52: needs 52+ train pts, starves early cutoffs
        rolling = [4, 13, 26]
        arima_seasonal = (0, 0, 0, 0)  # seasonal ARIMA with period=52 is prohibitively slow
    else:
        lags = [1, 2, 3, 6, 12, 24]
        rolling = [3, 6, 12]
        arima_seasonal = (1, 1, 1, period)

    factories = {
        "Naive":   lambda: SeasonalNaive(period=period),
        "ARIMA":   lambda: ARIMAModel(order=(1, 1, 1), seasonal_order=arima_seasonal),
        "Prophet": ProphetModel,
        "XGBoost": lambda: XGBoostModel(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         lags=lags, rolling_windows=rolling),
    }
    if INCLUDE_LSTM:
        factories["LSTM"] = lambda: LSTMModel(hidden_size=16, num_layers=2, epochs=20,
                                               seq_len=period)
    return factories


DATASETS = {
    "Airline":     (airline,     12),
    "Electricity": (electricity, 12),
    "Retail":      (retail,      52),
}
"""


def nb(*cells):
    n = nbf.v4.new_notebook()
    n["cells"] = list(cells)
    n["metadata"] = KERNEL
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


# ── Notebook 01: Data Exploration ─────────────────────────────────────────────

nb01 = nb(
    md(
        "# Notebook 01 — Data Exploration\n\n"
        "Load, inspect, and visualise all three datasets.\n"
        "Highlight the injected anomalies in Dataset 3 (Retail)."
    ),
    code(SETUP),
    code(LOAD_DATASETS),
    md("## All three series"),
    code("""\
fig, axes = plt.subplots(3, 1, figsize=(14, 9))

airline.plot(ax=axes[0], color="steelblue")
axes[0].set_title("Dataset 1: Airline Passengers (Monthly, 144 pts)")

electricity.plot(ax=axes[1], color="darkorange")
axes[1].set_title("Dataset 2: Electricity Production (Monthly, 397 pts, synthetic)")

retail.plot(ax=axes[2], color="forestgreen", alpha=0.85)
axes[2].set_title("Dataset 3: Retail Sales (Weekly, 208 pts) — red = anomalies")
for ev in ANOMALY_EVENTS:
    s_i, e_i = int(ev["start"]), int(ev["end"])
    axes[2].axvspan(retail.index[s_i], retail.index[e_i], alpha=0.25, color="red")
    axes[2].annotate(str(ev["label"]), xy=(retail.index[s_i], retail.max() * 0.9),
                     fontsize=8, color="darkred")

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, "01_all_datasets.png")
plt.show()
"""),
    md("## ACF and PACF — Airline Passengers\n\nA spike at lag 12 confirms annual seasonality."),
    code("""\
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(airline, lags=40, ax=axes[0])
plot_pacf(airline, lags=40, ax=axes[1], method="ols")
plt.suptitle("ACF / PACF — Airline Passengers", y=1.02)
plt.tight_layout()
save_fig(fig, "01_acf_pacf.png")
plt.show()
"""),
    md("## Seasonal decomposition"),
    code("""\
from statsmodels.tsa.seasonal import seasonal_decompose

for series, name, model, period in [
    (airline,     "Airline",     "multiplicative", 12),
    (electricity, "Electricity", "additive",       12),
]:
    decomp = seasonal_decompose(series, model=model, period=period)
    fig = decomp.plot()
    fig.set_size_inches(14, 7)
    fig.suptitle(f"Decomposition ({model}) — {name}", y=1.01)
    plt.tight_layout()
    save_fig(fig, f"01_decomp_{name.lower()}.png")
    plt.show()
"""),
    md("## Retail: base vs anomaly-injected"),
    code("""\
fig, ax = plt.subplots(figsize=(14, 5))
retail_base.plot(ax=ax, label="Base (no anomalies)", color="steelblue", alpha=0.6)
retail.plot(ax=ax, label="With anomalies", color="forestgreen")

for ev in ANOMALY_EVENTS:
    s_i, e_i = int(ev["start"]), int(ev["end"])
    ax.axvspan(retail.index[s_i], retail.index[e_i], alpha=0.2, color="red",
               label=f'{ev["label"]} (w{s_i}–{e_i})')

ax.set_title("Retail Sales — Base vs Anomaly-Injected")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "01_retail_anomalies.png")
plt.show()
"""),
    md("## Summary statistics"),
    code("""\
stats = pd.DataFrame({
    "Dataset":  ["Airline", "Electricity", "Retail (base)", "Retail (anomalies)"],
    "Points":   [len(airline), len(electricity), len(retail_base), len(retail)],
    "Freq":     ["Monthly"] * 2 + ["Weekly"] * 2,
    "Mean":     [s.mean() for s in [airline, electricity, retail_base, retail]],
    "Std":      [s.std()  for s in [airline, electricity, retail_base, retail]],
    "CV (%)":   [s.std() / s.mean() * 100 for s in [airline, electricity, retail_base, retail]],
}).set_index("Dataset").round(2)
save_json(stats.reset_index(), "01_summary_stats.json")
stats
"""),
    md(
        "## Key takeaways\n\n"
        "- **Airline:** Exponential trend + strong monthly seasonality. Classic benchmark.\n"
        "- **Electricity:** Linear trend, dual seasonality (12-month + 6-month), trend break at ~month 200.\n"
        "- **Retail:** Three injected anomalies at *known* timestamps — ground truth for robustness testing.\n\n"
        "Next: single train/test split evaluation — how most tutorials work, and why it misleads."
    ),
)

# ── Notebook 02: The Single-Cutoff Trap ───────────────────────────────────────

nb02 = nb(
    md(
        "# Notebook 02 — The Single-Cutoff Trap\n\n"
        "**Claim:** Prophet wins RMSE on all three datasets with a standard 80/20 split.\n\n"
        "**Reveal:** Shift the cutoff by 5% and rankings change on two out of three datasets.\n"
        "One cutoff tells you almost nothing about generalisation."
    ),
    code(SETUP),
    code(LOAD_DATASETS),
    code(MODEL_FACTORIES_CODE),
    md("## Protocol A — 80 % cutoff\n\nTrain on the first 80 % of each series, evaluate on the last 20 %."),
    code("""\
from src.evaluation.comparison import single_cutoff_eval

records_80 = []
for ds_name, (series, period) in DATASETS.items():
    for model_name, factory in make_factories(period).items():
        metrics = single_cutoff_eval(series, factory, cutoff_ratio=0.8)
        records_80.append({"Dataset": ds_name, "Model": model_name,
                           "RMSE": round(metrics["rmse"], 2),
                           "MAE":  round(metrics["mae"],  2)})
        print(f"  {ds_name:12s} {model_name:10s}  RMSE={metrics['rmse']:.2f}")

df_80 = pd.DataFrame(records_80)
save_json(df_80, "02_protocol_a_80pct.json")
"""),
    md("### RMSE leaderboard — 80 % cutoff"),
    code("""\
leaderboard_80 = (df_80.pivot(index="Model", columns="Dataset", values="RMSE")
                       .sort_values("Airline"))
leaderboard_80["Rank_Airline"]     = leaderboard_80["Airline"].rank().astype(int)
leaderboard_80["Rank_Electricity"] = leaderboard_80["Electricity"].rank().astype(int)
leaderboard_80["Rank_Retail"]      = leaderboard_80["Retail"].rank().astype(int)
leaderboard_80
"""),
    md(
        "## Now shift the cutoff to 75 %\n\n"
        "Same models, same datasets, just 5 % earlier split.\n"
        "If RMSE is a stable signal, rankings should be the same."
    ),
    code("""\
records_75 = []
for ds_name, (series, period) in DATASETS.items():
    for model_name, factory in make_factories(period).items():
        metrics = single_cutoff_eval(series, factory, cutoff_ratio=0.75)
        records_75.append({"Dataset": ds_name, "Model": model_name,
                           "RMSE": round(metrics["rmse"], 2),
                           "MAE":  round(metrics["mae"],  2)})

df_75 = pd.DataFrame(records_75)
save_json(df_75, "02_protocol_a_75pct.json")
"""),
    md("### RMSE leaderboard — 75 % cutoff"),
    code("""\
leaderboard_75 = (df_75.pivot(index="Model", columns="Dataset", values="RMSE")
                       .sort_values("Airline"))
leaderboard_75["Rank_Airline"]     = leaderboard_75["Airline"].rank().astype(int)
leaderboard_75["Rank_Electricity"] = leaderboard_75["Electricity"].rank().astype(int)
leaderboard_75["Rank_Retail"]      = leaderboard_75["Retail"].rank().astype(int)
leaderboard_75
"""),
    md("## Rank change (80 % → 75 %)"),
    code("""\
rank_cols = ["Rank_Airline", "Rank_Electricity", "Rank_Retail"]
rank_change = leaderboard_75[rank_cols] - leaderboard_80[rank_cols]
rank_change.columns = ["ΔRank Airline", "ΔRank Electricity", "ΔRank Retail"]
save_json(rank_change.reset_index(), "02_rank_changes.json")
rank_change.style.map(lambda v: "color: red"   if v > 0 else
                                "color: green"  if v < 0 else
                                "color: grey")
"""),
    md("## Visualise: how RMSE changes with cutoff position"),
    code("""\
cutoff_ratios = np.linspace(0.65, 0.9, 10)
rmse_traces = {name: [] for name in make_factories(12)}

series, period = DATASETS["Airline"]
for ratio in cutoff_ratios:
    for model_name, factory in make_factories(period).items():
        m = single_cutoff_eval(series, factory, cutoff_ratio=ratio)
        rmse_traces[model_name].append(m["rmse"])

fig, ax = plt.subplots(figsize=(12, 5))
for model_name, values in rmse_traces.items():
    ax.plot(cutoff_ratios, values, marker="o", label=model_name)
ax.axvline(0.8, color="red", linestyle="--", alpha=0.7, label="Standard 80 % split")
ax.set_xlabel("Cutoff ratio")
ax.set_ylabel("RMSE")
ax.set_title("Airline Passengers — RMSE vs Cutoff Position")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "02_rmse_vs_cutoff.png")
save_json(
    [{"model": m, "cutoff_ratio": float(r), "rmse": v}
     for m, vals in rmse_traces.items()
     for r, v in zip(cutoff_ratios, vals)],
    "02_rmse_vs_cutoff.json",
)
plt.show()
"""),
    md(
        "## Conclusion\n\n"
        "A single cutoff captures one specific temporal regime (one summer, one recession, "
        "one promotional period). The model that wins at 80 % may not win at 75 % or 85 %.\n\n"
        "**The fix:** evaluate over many cutoffs with an expanding window — Protocol B. "
        "That is the subject of the next notebook."
    ),
)

# ── Notebook 03: Expanding-Window Backtest ────────────────────────────────────

nb03 = nb(
    md(
        "# Notebook 03 — Expanding-Window Backtest (Protocol B)\n\n"
        "Instead of one cutoff, we evaluate at multiple cutoffs and measure *variance* "
        "across windows. A model that is consistently mediocre beats one that is brilliant "
        "on lucky cutoffs and terrible on unlucky ones."
    ),
    code(SETUP),
    code(LOAD_DATASETS),
    code(MODEL_FACTORIES_CODE),
    md(
        "## Protocol B: expanding-window backtest\n\n"
        "At each cutoff we train on all data up to that point and forecast one step ahead. "
        "We use `n_cutoffs=6` here for notebook speed; the full run uses 12."
    ),
    code("""\
from src.evaluation.backtest import expanding_window_backtest

N_CUTOFFS = 6  # increase for more thorough evaluation

# Run on Airline first as a walkthrough
airline_results = {}
for model_name, factory in make_factories(12).items():
    df = expanding_window_backtest(airline, factory, n_cutoffs=N_CUTOFFS,
                                   min_train_size=36)
    airline_results[model_name] = df
    print(f"  {model_name:10s}  mean RMSE={df['rmse'].mean():.2f}  "
          f"std={df['rmse'].std():.2f}")

airline_summary = pd.concat(
    {m: df.assign(model=m) for m, df in airline_results.items()}
).reset_index(drop=True)
save_json(airline_summary, "03_backtest_airline_cutoffs.json")
"""),
    md("## RMSE by cutoff — Airline Passengers"),
    code("""\
fig, ax = plt.subplots(figsize=(13, 5))
for model_name, df in airline_results.items():
    ax.plot(df["cutoff"], df["rmse"], marker="o", label=model_name)
ax.set_title("Airline Passengers — RMSE across expanding-window cutoffs")
ax.set_xlabel("Cutoff date")
ax.set_ylabel("RMSE")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "03_backtest_airline.png")
plt.show()
"""),
    md("## Run Protocol B on all three datasets"),
    code("""\
all_backtest = {}  # {(dataset, model): DataFrame}

for ds_name, (series, period) in DATASETS.items():
    min_train = 36 if period == 12 else 60
    for model_name, factory in make_factories(period).items():
        df = expanding_window_backtest(series, factory, n_cutoffs=N_CUTOFFS,
                                       min_train_size=min_train)
        all_backtest[(ds_name, model_name)] = df
    print(f"Done: {ds_name}")
"""),
    md("## Mean RMSE ± std across cutoffs"),
    code("""\
summary_rows = []
for (ds_name, model_name), df in all_backtest.items():
    summary_rows.append({
        "Dataset": ds_name,
        "Model":   model_name,
        "Mean RMSE": df["rmse"].mean(),
        "Std RMSE":  df["rmse"].std(),
        "CV (%)":    df["rmse"].std() / df["rmse"].mean() * 100,
    })

summary = pd.DataFrame(summary_rows).round(3)
save_json(summary, "03_backtest_summary.json")
summary.pivot_table(index="Model", columns="Dataset", values="Mean RMSE").round(2)
"""),
    md("### Variance (std) — who is most stable?"),
    code("""\
summary.pivot_table(index="Model", columns="Dataset", values="Std RMSE").round(3)
"""),
    md("## RMSE box plot across cutoffs — Retail"),
    code("""\
retail_dfs = {m: df for (ds, m), df in all_backtest.items() if ds == "Retail"}

fig, ax = plt.subplots(figsize=(10, 5))
data_to_plot = [df["rmse"].values for df in retail_dfs.values()]
bp = ax.boxplot(data_to_plot, labels=list(retail_dfs.keys()), patch_artist=True)
colors = plt.cm.Set2.colors
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("Retail Sales — RMSE distribution across cutoffs (Protocol B)")
ax.set_ylabel("RMSE")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
save_fig(fig, "03_boxplot_retail.png")
plt.show()
"""),
    md(
        "## Conclusion\n\n"
        "The expanding-window backtest reveals *variance* — a metric invisible to the single-cutoff "
        "approach. Some models look great on average but are highly variable across cutoffs "
        "(risky in production). Others are consistently mediocre (predictable and manageable).\n\n"
        "Next: we'll see that RMSE itself can be the wrong metric when errors have "
        "asymmetric business costs."
    ),
)

# ── Notebook 04: Decision Cost Metric ─────────────────────────────────────────

nb04 = nb(
    md(
        "# Notebook 04 — Decision Cost Metric\n\n"
        "RMSE treats over- and under-forecasting as equally bad. They are not.\n\n"
        "In demand forecasting, **under-forecasting** (stock-out, lost revenue) typically "
        "costs 3× more than **over-forecasting** (excess inventory). "
        "We define a configurable asymmetric cost and show how model rankings flip."
    ),
    code(SETUP),
    code(LOAD_DATASETS),
    code(MODEL_FACTORIES_CODE),
    md(
        "## The decision-cost function\n\n"
        "```python\n"
        "def decision_cost(y_true, y_pred, cost_under=3.0, cost_over=1.0):\n"
        '    errors = y_true - y_pred  # positive → under-forecast\n'
        "    costs = np.where(errors > 0, errors * cost_under, -errors * cost_over)\n"
        "    return costs.mean()\n"
        "```\n\n"
        "At ratio 3:1, one unit of under-forecast costs three times as much as one unit of over-forecast."
    ),
    code("""\
from src.evaluation.metrics import decision_cost, rmse
from src.evaluation.backtest import expanding_window_backtest

N_CUTOFFS = 6
COST_UNDER = 3.0
COST_OVER  = 1.0

records = []
for ds_name, (series, period) in DATASETS.items():
    min_train = 36 if period == 12 else 60
    for model_name, factory in make_factories(period).items():
        df = expanding_window_backtest(
            series, factory, n_cutoffs=N_CUTOFFS, min_train_size=min_train,
            cost_under=COST_UNDER, cost_over=COST_OVER,
        )
        records.append({
            "Dataset":      ds_name,
            "Model":        model_name,
            "RMSE":         df["rmse"].mean(),
            "DecisionCost": df["decision_cost"].mean(),
        })
    print(f"Done: {ds_name}")

perf = pd.DataFrame(records).round(3)
save_json(perf, "04_rmse_vs_decision_cost.json")
"""),
    md("## RMSE rankings vs Decision-Cost rankings"),
    code("""\
for ds_name in ["Airline", "Electricity", "Retail"]:
    sub = perf[perf["Dataset"] == ds_name].set_index("Model")
    sub["RMSE_rank"]  = sub["RMSE"].rank().astype(int)
    sub["Cost_rank"]  = sub["DecisionCost"].rank().astype(int)
    sub["Rank_delta"] = sub["Cost_rank"] - sub["RMSE_rank"]
    print(f"\\n=== {ds_name} ===")
    display(sub[["RMSE", "RMSE_rank", "DecisionCost", "Cost_rank", "Rank_delta"]]
              .sort_values("RMSE_rank"))
"""),
    md("## Visualise: RMSE vs Decision Cost scatter"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = plt.cm.Set1.colors

for ax, ds_name in zip(axes, ["Airline", "Electricity", "Retail"]):
    sub = perf[perf["Dataset"] == ds_name]
    for i, row in sub.iterrows():
        ax.scatter(row["RMSE"], row["DecisionCost"], s=120, color=colors[i % len(colors)],
                   zorder=5)
        ax.annotate(row["Model"], (row["RMSE"], row["DecisionCost"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Mean RMSE (Protocol B)")
    ax.set_ylabel("Mean Decision Cost (3:1)")
    ax.set_title(ds_name)
    ax.grid(alpha=0.3)

plt.suptitle("RMSE vs Decision Cost — models that look similar on RMSE diverge on cost",
             y=1.02)
plt.tight_layout()
save_fig(fig, "04_rmse_vs_cost.png")
plt.show()
"""),
    md("## Sensitivity: how rankings change with the cost ratio"),
    code("""\
cost_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]
ratio_records = []

series, period = DATASETS["Retail"]
min_train = 60

for ratio in cost_ratios:
    for model_name, factory in make_factories(period).items():
        df = expanding_window_backtest(
            series, factory, n_cutoffs=N_CUTOFFS, min_train_size=min_train,
            cost_under=ratio, cost_over=1.0,
        )
        ratio_records.append({
            "CostRatio": ratio,
            "Model": model_name,
            "DecisionCost": df["decision_cost"].mean(),
        })

ratio_df = pd.DataFrame(ratio_records)
save_json(ratio_df, "04_cost_sensitivity.json")
pivot = ratio_df.pivot(index="CostRatio", columns="Model", values="DecisionCost")

fig, ax = plt.subplots(figsize=(11, 5))
pivot.plot(ax=ax, marker="o")
ax.set_xlabel("Under-forecast cost ratio (over-forecast always = 1)")
ax.set_ylabel("Mean Decision Cost")
ax.set_title("Retail — Decision Cost vs Cost Ratio")
ax.legend(title="Model")
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "04_cost_sensitivity.png")
plt.show()
"""),
    md(
        "## Conclusion\n\n"
        "- At cost ratio 1:1, decision cost ≡ MAE — rankings match RMSE closely.\n"
        "- As the ratio increases toward 3:1 and 5:1, models that **systematically under-forecast** "
        "(e.g., by anchoring to seasonal means) are penalised disproportionately.\n"
        "- **Pick your metric before you pick your model.** The \"best\" model depends entirely "
        "on what error costs your business.\n\n"
        "Next: *why* some models under-forecast and others do not — the calendar-memorisation problem."
    ),
)

# ── Notebook 05: When Calendar Models Break ───────────────────────────────────

nb05 = nb(
    md(
        "# Notebook 05 — When Calendar Models Break\n\n"
        "Prophet is explicitly designed to capture day-of-week and month-of-year patterns. "
        "When the test period follows the same calendar as training, this is a strength. "
        "When an *external shock* disrupts the pattern, it becomes a liability — "
        "Prophet keeps predicting the calendar even when recent data screams otherwise.\n\n"
        "We dissect this on Dataset 3 (Retail, with three injected anomalies)."
    ),
    code(SETUP),
    code(LOAD_DATASETS),
    code("""\
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.backtest import expanding_window_backtest

N_CUTOFFS = 8
MIN_TRAIN = 60
"""),
    md(
        "## Expanding-window backtest on retail — Prophet vs XGBoost\n\n"
        "We compare only these two models: the calendar-based one and the lag-based one."
    ),
    code("""\
factories = {
    "Prophet": ProphetModel,
    "XGBoost": lambda: XGBoostModel(n_estimators=100, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     lags=[1, 2, 3, 4, 13, 26],
                                     rolling_windows=[4, 13, 26]),
}

backtest_results = {}
for model_name, factory in factories.items():
    df = expanding_window_backtest(retail, factory, n_cutoffs=N_CUTOFFS,
                                   min_train_size=MIN_TRAIN,
                                   cost_under=3.0, cost_over=1.0)
    backtest_results[model_name] = df
    print(f"{model_name:10s}  RMSE={df['rmse'].mean():.2f}  "
          f"DecisionCost={df['decision_cost'].mean():.2f}")

save_json(
    pd.concat({m: df.assign(model=m) for m, df in backtest_results.items()}).reset_index(drop=True),
    "05_backtest_results.json",
)
"""),
    md("## RMSE and Decision Cost by cutoff"),
    code("""\
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for model_name, df in backtest_results.items():
    axes[0].plot(df["cutoff"], df["rmse"],          marker="o", label=model_name)
    axes[1].plot(df["cutoff"], df["decision_cost"], marker="s", label=model_name)

for ax, ylabel, title in zip(
    axes,
    ["RMSE", "Decision Cost (3:1)"],
    ["RMSE by cutoff", "Decision Cost by cutoff"],
):
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

# Mark anomaly windows
for ev in ANOMALY_EVENTS:
    s_dt = retail.index[int(ev["start"])]
    e_dt = retail.index[int(ev["end"])]
    for ax in axes:
        ax.axvspan(s_dt, e_dt, alpha=0.15, color="red")

plt.xlabel("Cutoff date")
plt.tight_layout()
save_fig(fig, "05_prophet_vs_xgb_backtest.png")
plt.show()
"""),
    md(
        "## Close-up: forecast vs actual around each anomaly\n\n"
        "We train each model on data up to one period *before* the anomaly, "
        "then forecast through it."
    ),
    code("""\
def plot_around_event(event_start: int, window: int = 20, lookahead: int = 15,
                      label: str = "") -> None:
    \"\"\"Train up to event_start-1, forecast lookahead steps, plot.\"\"\"
    train_end = max(MIN_TRAIN, event_start - 1)
    train = retail.iloc[:train_end]
    actual = retail.iloc[train_end: train_end + lookahead]

    fig, ax = plt.subplots(figsize=(13, 4))

    # Show recent history
    tail = retail.iloc[max(0, train_end - window): train_end]
    ax.plot(tail.index, tail.values, color="black", label="History", linewidth=2)
    ax.plot(actual.index, actual.values, color="black", linestyle="--",
            label="Actual (test)", linewidth=2)

    colors_map = {"Prophet": "crimson", "XGBoost": "steelblue"}
    for model_name, factory in factories.items():
        m = factory()
        m.fit(train)
        pred = m.predict(lookahead)
        pred.index = actual.index[: len(pred)]
        ax.plot(pred.index, pred.values, color=colors_map[model_name],
                linestyle="-.", label=f"{model_name} forecast", linewidth=1.8)

    ax.axvline(retail.index[train_end], color="grey", linestyle=":", label="Train/test split")
    ax.set_title(f"Anomaly: {label}  (train cutoff = week {train_end})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    safe = label.lower().replace(" ", "_")
    save_fig(fig, f"05_anomaly_{safe}.png")
    plt.show()


for ev in ANOMALY_EVENTS:
    plot_around_event(int(ev["start"]), label=str(ev["label"]))
"""),
    md("## Quantitative comparison around anomalies"),
    code("""\
from src.evaluation.metrics import rmse, decision_cost as dcost

rows = []
for ev in ANOMALY_EVENTS:
    train_end = max(MIN_TRAIN, int(ev["start"]) - 1)
    train  = retail.iloc[:train_end]
    actual = retail.iloc[train_end: train_end + 10].to_numpy()

    for model_name, factory in factories.items():
        m = factory()
        m.fit(train)
        pred = m.predict(10).to_numpy()[:len(actual)]
        rows.append({
            "Anomaly":      str(ev["label"]),
            "Model":        model_name,
            "RMSE":         rmse(actual, pred),
            "DecisionCost": dcost(actual, pred, 3.0, 1.0),
        })

anom_df = pd.DataFrame(rows).round(2)
save_json(anom_df, "05_anomaly_comparison.json")
anom_df.pivot_table(index=["Anomaly", "Model"], values=["RMSE", "DecisionCost"]).round(2)
"""),
    md(
        "## Diagnostic: when to distrust a calendar model\n\n"
        "| Condition | Calendar model risk | Recommendation |\n"
        "|---|---|---|\n"
        "| External shocks > 2× per year | High | Use lag-based model |\n"
        "| Stable seasonal + trend only | Low | Prophet / SARIMAX fine |\n"
        "| Asymmetric error costs | Always relevant | Add decision-cost metric |\n"
        "| Short history (< 2 seasons) | High | Naive or simple ARIMA |\n\n"
        "## Conclusion\n\n"
        "Prophet's seasonal component is trained to predict *the calendar*, not *recent observations*. "
        "During anomalous periods, it reverts to the seasonal mean even when recent actuals diverge sharply. "
        "XGBoost's lag features pick up the anomaly within 2–3 periods because they look at "
        "*recent values*, not calendar expectations.\n\n"
        "**The takeaway:** before deploying any calendar model, inject synthetic shocks into "
        "your historical data at the expected frequency, run an expanding-window backtest with "
        "an asymmetric cost metric, and verify the model degrades gracefully."
    ),
)

# ── Write all notebooks ────────────────────────────────────────────────────────

for name, notebook in [
    ("01_data_exploration",        nb01),
    ("02_single_cutoff_trap",      nb02),
    ("03_expanding_window_backtest", nb03),
    ("04_decision_cost_metric",    nb04),
    ("05_when_calendar_models_break", nb05),
]:
    path = NOTEBOOKS_DIR / f"{name}.ipynb"
    with path.open("w") as f:
        nbf.write(notebook, f)
    print(f"Written: {path.name}")
