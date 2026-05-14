from pathlib import Path

import yaml
from pydantic import BaseModel


class BacktestConfig(BaseModel):
    n_cutoffs: int = 12
    min_train_size: int = 24


class CostConfig(BaseModel):
    under: float = 3.0
    over: float = 1.0


class XGBoostConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 3
    n_jobs: int = 2
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    lags: list[int] = [1, 2, 3, 6, 12, 24]
    rolling_windows: list[int] = [3, 6, 12]


class LSTMConfig(BaseModel):
    hidden_size: int = 32
    num_layers: int = 2
    epochs: int = 50
    lr: float = 0.01
    seq_len: int = 12


class ARIMAConfig(BaseModel):
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)


class ModelsConfig(BaseModel):
    xgboost: XGBoostConfig = XGBoostConfig()
    lstm: LSTMConfig = LSTMConfig()
    arima: ARIMAConfig = ARIMAConfig()


class Settings(BaseModel):
    backtest: BacktestConfig = BacktestConfig()
    cost: CostConfig = CostConfig()
    models: ModelsConfig = ModelsConfig()


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def get_settings() -> Settings:
    if _CONFIG_PATH.exists():
        raw = yaml.safe_load(_CONFIG_PATH.read_text())
        return Settings.model_validate(raw)
    return Settings()
