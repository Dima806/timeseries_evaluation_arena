from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    ARIMAConfig,
    BacktestConfig,
    CostConfig,
    LSTMConfig,
    ModelsConfig,
    Settings,
    XGBoostConfig,
    get_settings,
)

# ── Dataclass defaults ────────────────────────────────────────────────────────


def test_backtest_config_defaults():
    cfg = BacktestConfig()
    assert cfg.n_cutoffs == 12
    assert cfg.min_train_size == 24


def test_cost_config_defaults():
    cfg = CostConfig()
    assert cfg.under == 3.0
    assert cfg.over == 1.0


def test_xgboost_config_defaults():
    cfg = XGBoostConfig()
    assert cfg.n_estimators == 200
    assert cfg.max_depth == 3
    assert cfg.n_jobs == 2
    assert cfg.learning_rate == pytest.approx(0.05)
    assert cfg.subsample == pytest.approx(0.8)
    assert cfg.lags == [1, 2, 3, 6, 12, 24]
    assert cfg.rolling_windows == [3, 6, 12]


def test_lstm_config_defaults():
    cfg = LSTMConfig()
    assert cfg.hidden_size == 32
    assert cfg.num_layers == 2
    assert cfg.epochs == 50
    assert cfg.lr == pytest.approx(0.01)
    assert cfg.seq_len == 12


def test_arima_config_defaults():
    cfg = ARIMAConfig()
    assert cfg.order == (1, 1, 1)
    assert cfg.seasonal_order == (1, 1, 1, 12)


def test_models_config_defaults():
    cfg = ModelsConfig()
    assert isinstance(cfg.xgboost, XGBoostConfig)
    assert isinstance(cfg.lstm, LSTMConfig)
    assert isinstance(cfg.arima, ARIMAConfig)


def test_settings_defaults():
    s = Settings()
    assert isinstance(s.backtest, BacktestConfig)
    assert isinstance(s.cost, CostConfig)
    assert isinstance(s.models, ModelsConfig)


# ── get_settings() with yaml present ─────────────────────────────────────────


def test_get_settings_loads_yaml():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.backtest.n_cutoffs == 12
    assert settings.cost.under == pytest.approx(3.0)
    assert settings.models.xgboost.n_estimators == 200
    assert settings.models.lstm.hidden_size == 32
    assert settings.models.arima.order == (1, 1, 1)


def test_get_settings_yaml_values_match_defaults():
    settings = get_settings()
    defaults = Settings()
    assert settings.backtest.n_cutoffs == defaults.backtest.n_cutoffs
    assert settings.cost.under == defaults.cost.under
    assert settings.cost.over == defaults.cost.over


# ── get_settings() fallback when yaml is absent ───────────────────────────────


def test_get_settings_fallback_without_yaml():
    with patch("src.config._CONFIG_PATH", Path("/nonexistent/settings.yaml")):
        settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.backtest.n_cutoffs == 12
    assert settings.cost.under == pytest.approx(3.0)


# ── model_validate round-trip ─────────────────────────────────────────────────


def test_settings_model_validate_partial():
    raw = {"cost": {"under": 5.0, "over": 2.0}}
    s = Settings.model_validate(raw)
    assert s.cost.under == pytest.approx(5.0)
    assert s.cost.over == pytest.approx(2.0)
    assert s.backtest.n_cutoffs == 12  # default preserved


def test_settings_model_validate_full():
    raw = {
        "backtest": {"n_cutoffs": 6, "min_train_size": 48},
        "cost": {"under": 1.0, "over": 1.0},
        "models": {
            "xgboost": {"n_estimators": 50, "max_depth": 3, "n_jobs": 1},
            "lstm": {"hidden_size": 16, "num_layers": 1, "epochs": 10, "lr": 0.001},
            "arima": {"order": [0, 1, 0], "seasonal_order": [0, 1, 0, 12]},
        },
    }
    s = Settings.model_validate(raw)
    assert s.backtest.n_cutoffs == 6
    assert s.models.xgboost.n_estimators == 50
    assert s.models.lstm.epochs == 10
    assert s.models.arima.order == (0, 1, 0)
