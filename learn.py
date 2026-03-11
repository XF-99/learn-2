import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

try:
    from scipy.signal import stft
except Exception:  # pragma: no cover - optional dependency
    stft = None


WELLS = [
    ("岩溶水_每周数据.csv", "岩溶水"),
    ("孔隙水_每周数据.csv", "孔隙水"),
    ("裂隙水_每周数据.csv", "裂隙水"),
]


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dates_calib: np.ndarray
    dates_test: np.ndarray
    scaler: StandardScaler


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        # 多层 LSTM 编码序列，并使用最后一个时间步的表示做回归。
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Transformer 使用固定的正弦位置编码。
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int, heads: int, layers: int, dropout: float):
        super().__init__()
        # 先将输入特征投影到模型维度，再用自注意力编码。
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        # 空洞卷积用于扩大时间感受野。
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x[:, :, : -self.conv.padding[0]]
        return self.drop(self.relu(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: List[int], kernel: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, ch, kernel=kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.head(x).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str) -> pd.DataFrame:
    # 读取数据并按日期排序，保证时序一致。
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_sequences(
    data_scaled: np.ndarray,
    lookback: int,
    horizon: int,
    target_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_out, target_idx = [], [], []
    # 构建滑动窗口序列用于监督学习。
    for i in range(lookback, len(data_scaled) - horizon + 1):
        X.append(data_scaled[i - lookback : i])
        y_out.append(data_scaled[i + horizon - 1, target_index])
        target_idx.append(i + horizon - 1)
    return np.array(X), np.array(y_out), np.array(target_idx)


def prepare_splits(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    calib_ratio: float,
) -> SplitData:
    # 按时间顺序切分：train/val/calib/test。
    data = df[features].values
    if train_ratio + val_ratio + calib_ratio >= 1:
        raise ValueError("train_ratio + val_ratio + calib_ratio must be less than 1.0")

    n_sequences = len(data) - lookback - horizon + 1
    if n_sequences <= 0:
        raise ValueError("Not enough samples for the given lookback and horizon.")

    train_size = int(train_ratio * n_sequences)
    val_size = int(val_ratio * n_sequences)
    calib_size = int(calib_ratio * n_sequences)
    test_size = n_sequences - train_size - val_size - calib_size
    if min(train_size, val_size, calib_size, test_size) <= 0:
        raise ValueError("train/val/calib/test split has empty part. Adjust ratios or data size.")

    # 仅在训练期拟合标准化器，避免信息泄漏。
    train_end_idx = lookback + train_size + horizon - 2
    scaler = StandardScaler()
    scaler.fit(data[: train_end_idx + 1])
    data_scaled = scaler.transform(data)

    X, y, idx = build_sequences(data_scaled, lookback, horizon, target_index=0)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]

    calib_start = train_size + val_size
    calib_end = calib_start + calib_size
    X_calib, y_calib = X[calib_start:calib_end], y[calib_start:calib_end]
    X_test, y_test = X[calib_end:], y[calib_end:]
    dates_calib = df.loc[idx[calib_start:calib_end], "Date"].values
    dates_test = df.loc[idx[calib_end:], "Date"].values

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test,
        y_test=y_test,
        dates_calib=dates_calib,
        dates_test=dates_test,
        scaler=scaler,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
) -> nn.Module:
    # 带早停的训练循环，用于抑制过拟合。
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    # 批量推理以提升预测效率。
    model.eval()
    preds = []
    loader = DataLoader(SequenceDataset(X, np.zeros(len(X))), batch_size=256, shuffle=False)
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # 在该任务下 NSE 与 R2 等价，保留两者便于水文报告。
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse}


def inverse_target(scaler: StandardScaler, y_scaled: np.ndarray, n_features: int) -> np.ndarray:
    # 仅对目标列（第 0 列）做反标准化。
    zeros = np.zeros((len(y_scaled), n_features))
    zeros[:, 0] = y_scaled
    return scaler.inverse_transform(zeros)[:, 0]


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    quantile = float(np.clip(quantile, 0.0, 1.0))
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = np.maximum(weights[sorter], 0.0)
    weight_sum = weights_sorted.sum()
    if weight_sum <= 0:
        return float(np.quantile(values_sorted, quantile))
    cdf = np.cumsum(weights_sorted) / weight_sum
    idx = np.searchsorted(cdf, quantile, side="left")
    idx = min(idx, len(values_sorted) - 1)
    return float(values_sorted[idx])


def init_tau_from_calib_z(z_calib_std: np.ndarray) -> float:
    if len(z_calib_std) <= 1:
        return 1.0
    diffs = np.diff(z_calib_std, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    tau = float(np.median(dists)) if len(dists) > 0 else 1.0
    if tau <= 1e-8:
        tau = 1.0
    return tau


def compute_weighted_conformal_intervals(
    yhat: np.ndarray,
    z_target_std: np.ndarray,
    target_dates: np.ndarray,
    calib_scores: np.ndarray,
    z_calib_std: np.ndarray,
    calib_dates: np.ndarray,
    gamma: float,
    lambda_t: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    tau2 = max(float(tau) ** 2, 1e-12)
    calib_dates = pd.to_datetime(calib_dates)
    target_dates = pd.to_datetime(target_dates)

    pi90_l = np.zeros(len(yhat))
    pi90_u = np.zeros(len(yhat))
    pi95_l = np.zeros(len(yhat))
    pi95_u = np.zeros(len(yhat))

    for j in range(len(yhat)):
        delta_days = np.abs((calib_dates - target_dates[j]).days.values.astype(float))
        delta_weeks = delta_days / 7.0
        w_time = np.exp(-lambda_t * delta_weeks)

        dz = z_calib_std - z_target_std[j]
        dist2 = np.sum(dz * dz, axis=1)
        w_sim = np.exp(-dist2 / tau2)

        w_time_sum = w_time.sum()
        w_sim_sum = w_sim.sum()
        if w_time_sum <= 0:
            w_time_norm = np.full_like(w_time, 1.0 / len(w_time), dtype=float)
        else:
            w_time_norm = w_time / w_time_sum

        if w_sim_sum <= 0:
            w_sim_norm = np.full_like(w_sim, 1.0 / len(w_sim), dtype=float)
        else:
            w_sim_norm = w_sim / w_sim_sum

        w_mix = gamma * w_time_norm + (1.0 - gamma) * w_sim_norm
        w_mix_sum = w_mix.sum()
        if w_mix_sum <= 0:
            w_mix = np.full_like(w_mix, 1.0 / len(w_mix), dtype=float)
        else:
            w_mix = w_mix / w_mix_sum

        q90 = weighted_quantile(calib_scores, 0.90, w_mix)
        q95 = weighted_quantile(calib_scores, 0.95, w_mix)

        pi90_l[j] = yhat[j] - q90
        pi90_u[j] = yhat[j] + q90
        pi95_l[j] = yhat[j] - q95
        pi95_u[j] = yhat[j] + q95

    return pi90_l, pi90_u, pi95_l, pi95_u


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    return picp, mpiw


def make_time_frequency_plot(series: pd.Series, out_path: str) -> None:
    if stft is None:
        return
    # 使用 STFT 可视化时频特征。
    f, t, Z = stft(series.values, fs=1.0, nperseg=52)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, np.abs(Z), shading="auto")
    plt.ylabel("Frequency (1/week)")
    plt.xlabel("Time (weeks)")
    plt.title("Time-Frequency (STFT)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predictions(
    dates: np.ndarray,
    actual: np.ndarray,
    preds: Dict[str, np.ndarray],
    out_path: str,
    pi90: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", color="black")
    for name, pred in preds.items():
        plt.plot(dates, pred, label=name)
    if pi95 is not None:
        plt.fill_between(dates, pi95[0], pi95[1], color="#8A2BE2", alpha=0.15, label="PI95")
    if pi90 is not None:
        plt.fill_between(dates, pi90[0], pi90[1], color="#8A2BE2", alpha=0.30, label="PI90")
    plt.legend()
    plt.title("Test Predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stacking(
    dates: np.ndarray,
    actual: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    pi90: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", color="#FF6347")
    plt.plot(dates, pred, label="Stacking", color="#8A2BE2")
    if pi95 is not None:
        plt.fill_between(dates, pi95[0], pi95[1], color="#8A2BE2", alpha=0.15, label="PI95")
    if pi90 is not None:
        plt.fill_between(dates, pi90[0], pi90[1], color="#8A2BE2", alpha=0.30, label="PI90")
    plt.title("Residual Stacking Prediction")
    plt.xlabel("Date")
    plt.ylabel("GWL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_hist(residual: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(residual, bins=50, color="#FF4500", alpha=0.7)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_future_forecast(
    history_dates: np.ndarray,
    history_values: np.ndarray,
    future_dates: np.ndarray,
    future_values: np.ndarray,
    out_path: str,
    pi90: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(history_dates, history_values, label="History", color="#FF6347")
    plt.plot(future_dates, future_values, label="Future", color="#00CED1")
    if pi95 is not None:
        plt.fill_between(future_dates, pi95[0], pi95[1], color="#00CED1", alpha=0.15, label="Future PI95")
    if pi90 is not None:
        plt.fill_between(future_dates, pi90[0], pi90[1], color="#00CED1", alpha=0.30, label="Future PI90")
    plt.title("Rolling Forecast")
    plt.xlabel("Date")
    plt.ylabel("GWL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_well(
    file_path: str,
    aquifer: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    calib_ratio: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    out_dir: str,
    future_steps: int,
    gamma: float,
    lambda_t: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    # 单口井完整流程：训练、集成、共形区间与可视化输出。
    df = load_data(file_path)
    features = ["GWL", "TASMAX", "TAS", "Precipitation"]
    target = "GWL"

    split = prepare_splits(
        df,
        features=features,
        target=target,
        lookback=lookback,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        calib_ratio=calib_ratio,
    )

    train_loader = DataLoader(
        SequenceDataset(split.X_train, split.y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(split.X_val, split.y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    lstm = LSTMRegressor(n_features=len(features), hidden=64, layers=2, dropout=0.2)
    transformer = TransformerRegressor(n_features=len(features), d_model=64, heads=4, layers=2, dropout=0.2)
    tcn = TCNRegressor(n_features=len(features), channels=[32, 32, 32], kernel=3, dropout=0.2)

    lstm = train_model(lstm, train_loader, val_loader, device, epochs, lr, patience)
    transformer = train_model(transformer, train_loader, val_loader, device, epochs, lr, patience)
    tcn = train_model(tcn, train_loader, val_loader, device, epochs, lr, patience)

    pred_val_lstm = predict(lstm, split.X_val, device)
    pred_val_trans = predict(transformer, split.X_val, device)
    pred_val_tcn = predict(tcn, split.X_val, device)

    # Stacking 元模型学习对基模型预测的残差修正。
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    stack_val_X = np.column_stack([pred_val_lstm, pred_val_trans, pred_val_tcn])
    stack_val_res = split.y_val - stack_val_X.mean(axis=1)
    xgb.fit(stack_val_X, stack_val_res)

    pred_calib_lstm = predict(lstm, split.X_calib, device)
    pred_calib_trans = predict(transformer, split.X_calib, device)
    pred_calib_tcn = predict(tcn, split.X_calib, device)

    stack_calib_X = np.column_stack([pred_calib_lstm, pred_calib_trans, pred_calib_tcn])
    stack_calib_res_pred = xgb.predict(stack_calib_X).reshape(-1)
    pred_calib_stack = stack_calib_X.mean(axis=1) + stack_calib_res_pred

    pred_test_lstm = predict(lstm, split.X_test, device)
    pred_test_trans = predict(transformer, split.X_test, device)
    pred_test_tcn = predict(tcn, split.X_test, device)

    stack_test_X = np.column_stack([pred_test_lstm, pred_test_trans, pred_test_tcn])
    stack_res_pred = xgb.predict(stack_test_X).reshape(-1)
    pred_test_stack = stack_test_X.mean(axis=1) + stack_res_pred

    n_features = len(features)
    y_calib_orig = inverse_target(split.scaler, split.y_calib, n_features)
    y_test_orig = inverse_target(split.scaler, split.y_test, n_features)

    pred_calib_lstm_orig = inverse_target(split.scaler, pred_calib_lstm, n_features)
    pred_calib_trans_orig = inverse_target(split.scaler, pred_calib_trans, n_features)
    pred_calib_tcn_orig = inverse_target(split.scaler, pred_calib_tcn, n_features)
    pred_calib_stack_orig = inverse_target(split.scaler, pred_calib_stack, n_features)

    pred_test_lstm_orig = inverse_target(split.scaler, pred_test_lstm, n_features)
    pred_test_trans_orig = inverse_target(split.scaler, pred_test_trans, n_features)
    pred_test_tcn_orig = inverse_target(split.scaler, pred_test_tcn, n_features)
    pred_test_stack_orig = inverse_target(split.scaler, pred_test_stack, n_features)

    calib_scores = np.abs(y_calib_orig - pred_calib_stack_orig)

    z_calib_raw = np.column_stack([pred_calib_lstm_orig, pred_calib_trans_orig, pred_calib_tcn_orig])
    z_mu = z_calib_raw.mean(axis=0)
    z_std = z_calib_raw.std(axis=0)
    z_std = np.where(z_std <= 1e-8, 1.0, z_std)
    z_calib_std = (z_calib_raw - z_mu) / z_std

    tau_init = init_tau_from_calib_z(z_calib_std)

    z_test_raw = np.column_stack([pred_test_lstm_orig, pred_test_trans_orig, pred_test_tcn_orig])
    z_test_std = (z_test_raw - z_mu) / z_std

    pi90_l_test, pi90_u_test, pi95_l_test, pi95_u_test = compute_weighted_conformal_intervals(
        yhat=pred_test_stack_orig,
        z_target_std=z_test_std,
        target_dates=split.dates_test,
        calib_scores=calib_scores,
        z_calib_std=z_calib_std,
        calib_dates=split.dates_calib,
        gamma=gamma,
        lambda_t=lambda_t,
        tau=tau_init,
    )

    picp90, mpiw90 = interval_metrics(y_test_orig, pi90_l_test, pi90_u_test)
    picp95, mpiw95 = interval_metrics(y_test_orig, pi95_l_test, pi95_u_test)

    metrics_map = {
        "LSTM": metrics(y_test_orig, pred_test_lstm_orig),
        "Transformer": metrics(y_test_orig, pred_test_trans_orig),
        "TCN": metrics(y_test_orig, pred_test_tcn_orig),
        "Stacking": {
            **metrics(y_test_orig, pred_test_stack_orig),
            "PICP90": picp90,
            "PICP95": picp95,
            "MPIW90": mpiw90,
            "MPIW95": mpiw95,
            "TAU_INIT": tau_init,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "Date": split.dates_test,
            "Actual": y_test_orig,
            "LSTM": pred_test_lstm_orig,
            "Transformer": pred_test_trans_orig,
            "TCN": pred_test_tcn_orig,
            "Stacking": pred_test_stack_orig,
            "PI90_Lower": pi90_l_test,
            "PI90_Upper": pi90_u_test,
            "PI95_Lower": pi95_l_test,
            "PI95_Upper": pi95_u_test,
            "Aquifer": aquifer,
        }
    )
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    plot_predictions(
        split.dates_test,
        y_test_orig,
        {
            "LSTM": pred_test_lstm_orig,
            "Transformer": pred_test_trans_orig,
            "TCN": pred_test_tcn_orig,
            "Stacking": pred_test_stack_orig,
        },
        os.path.join(out_dir, "test_predictions.png"),
        pi90=(pi90_l_test, pi90_u_test),
        pi95=(pi95_l_test, pi95_u_test),
    )

    plot_stacking(
        split.dates_test,
        y_test_orig,
        pred_test_stack_orig,
        os.path.join(out_dir, "stacking_predictions.png"),
        pi90=(pi90_l_test, pi90_u_test),
        pi95=(pi95_l_test, pi95_u_test),
    )

    plot_residual_hist(
        y_test_orig - pred_test_stack_orig,
        os.path.join(out_dir, "stacking_residuals.png"),
    )

    make_time_frequency_plot(df["GWL"], os.path.join(out_dir, "time_frequency.png"))

    last_seq = split.X_test[-1].copy()
    future_preds_orig = []
    future_lstm_orig = []
    future_trans_orig = []
    future_tcn_orig = []

    # 从最后一个可用窗口开始进行滚动未来预测。
    for _ in range(future_steps):
        seq_t = torch.tensor(last_seq[np.newaxis, :, :], dtype=torch.float32).to(device)
        lstm_p = lstm(seq_t).detach().cpu().numpy().reshape(-1)
        trans_p = transformer(seq_t).detach().cpu().numpy().reshape(-1)
        tcn_p = tcn(seq_t).detach().cpu().numpy().reshape(-1)
        stacked_X = np.column_stack([lstm_p, trans_p, tcn_p])
        res_p = xgb.predict(stacked_X).reshape(-1)
        pred = stacked_X.mean(axis=1) + res_p

        pred_orig = inverse_target(split.scaler, np.array([pred[0]]), len(features))[0]
        future_preds_orig.append(pred_orig)
        future_lstm_orig.append(inverse_target(split.scaler, np.array([lstm_p[0]]), len(features))[0])
        future_trans_orig.append(inverse_target(split.scaler, np.array([trans_p[0]]), len(features))[0])
        future_tcn_orig.append(inverse_target(split.scaler, np.array([tcn_p[0]]), len(features))[0])

        new_row = np.zeros(last_seq.shape[1])
        new_row[0] = pred[0]
        last_seq = np.vstack([last_seq[1:], new_row])

    future_preds_orig = np.array(future_preds_orig)
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=future_steps, freq="7D")

    z_future_raw = np.column_stack([future_lstm_orig, future_trans_orig, future_tcn_orig])
    z_future_std = (z_future_raw - z_mu) / z_std

    pi90_l_future, pi90_u_future, pi95_l_future, pi95_u_future = compute_weighted_conformal_intervals(
        yhat=future_preds_orig,
        z_target_std=z_future_std,
        target_dates=future_dates.values,
        calib_scores=calib_scores,
        z_calib_std=z_calib_std,
        calib_dates=split.dates_calib,
        gamma=gamma,
        lambda_t=lambda_t,
        tau=tau_init,
    )

    future_df = pd.DataFrame(
        {
            "Date": future_dates.values,
            "Future_Stacking": future_preds_orig,
            "PI90_Lower": pi90_l_future,
            "PI90_Upper": pi90_u_future,
            "PI95_Lower": pi95_l_future,
            "PI95_Upper": pi95_u_future,
            "Aquifer": aquifer,
        }
    )
    future_df.to_csv(os.path.join(out_dir, "future_predictions.csv"), index=False)

    plot_future_forecast(
        df["Date"].values,
        df["GWL"].values,
        future_dates.values,
        future_preds_orig,
        os.path.join(out_dir, "future_forecast.png"),
        pi90=(pi90_l_future, pi90_u_future),
        pi95=(pi95_l_future, pi95_u_future),
    )

    return pred_df, metrics_map


def summarize_metrics(all_metrics: Dict[str, Dict[str, Dict[str, float]]], out_dir: str) -> None:
    # 汇总各井各模型指标并导出对比结果。
    rows = []
    for well_id, model_metrics in all_metrics.items():
        for model_name, metric_values in model_metrics.items():
            row = {"well": well_id, "model": model_name}
            row.update(metric_values)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # 快速生成跨井对比图。
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="model", y="RMSE", hue="well")
    plt.title("RMSE by Model and Well")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_comparison.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="model", y="NSE", hue="well")
    plt.title("NSE by Model and Well")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nse_comparison.png"), dpi=150)
    plt.close()


def main() -> None:
    # 程序入口：解析参数并逐井运行。
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--calib_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--future_steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda_t", type=float, default=0.02)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.calib_ratio
    if ratio_sum >= 1.0:
        raise ValueError("train_ratio + val_ratio + calib_ratio must be less than 1.0")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for file_path, aquifer in WELLS:
        well_id = aquifer
        out_dir = os.path.join(args.out_dir, aquifer)
        _, metrics_map = run_well(
            file_path=file_path,
            aquifer=aquifer,
            lookback=args.lookback,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            calib_ratio=args.calib_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=device,
            out_dir=out_dir,
            future_steps=args.future_steps,
            gamma=args.gamma,
            lambda_t=args.lambda_t,
        )
        all_metrics[well_id] = metrics_map

    summarize_metrics(all_metrics, args.out_dir)
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
