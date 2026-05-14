import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMModel:
    """Two-layer LSTM for time series forecasting (CPU-only)."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 2,
        epochs: int = 50,
        lr: float = 0.01,
        seq_len: int = 12,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.seq_len = seq_len
        self._net: _LSTMNet | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._last_seq: torch.Tensor | None = None

    def fit(self, series: pd.Series) -> None:
        values = series.to_numpy(dtype=float)
        self._mean = float(values.mean())
        self._std = float(values.std()) or 1.0
        scaled = (values - self._mean) / self._std

        xs, ys = [], []
        for i in range(len(scaled) - self.seq_len):
            xs.append(scaled[i : i + self.seq_len])
            ys.append(scaled[i + self.seq_len])

        if not xs:
            return

        x_tensor = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

        net = _LSTMNet(self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(net(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        self._net = net
        self._last_seq = (
            torch.tensor(scaled[-self.seq_len :], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )

    def predict(self, horizon: int) -> pd.Series:
        assert self._net is not None, "Call fit() before predict()."
        assert self._last_seq is not None
        self._net.eval()
        seq = self._last_seq.clone()
        forecasts: list[float] = []
        with torch.no_grad():
            for _ in range(horizon):
                pred_scaled = self._net(seq).item()
                forecasts.append(pred_scaled * self._std + self._mean)
                new_pt = torch.tensor([[[pred_scaled]]], dtype=torch.float32)
                seq = torch.cat([seq[:, 1:, :], new_pt], dim=1)
        return pd.Series(forecasts, name="forecast")
