# -*- coding: utf-8 -*-
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import vendor_bootstrap  # noqa: F401

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import font_manager as fm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

try:
    from scipy.signal import find_peaks, stft
except Exception:  # pragma: no cover - optional dependency
    find_peaks = None
    stft = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


# 优先选择系统已安装的中文字体，避免图表中文显示异常。
_FONT_CANDIDATES = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
_INSTALLED_FONTS = {f.name for f in fm.fontManager.ttflist}
_CHOSEN_FONT = next((name for name in _FONT_CANDIDATES if name in _INSTALLED_FONTS), None)
if _CHOSEN_FONT is not None:
    mpl.rcParams["font.family"] = [_CHOSEN_FONT]
    mpl.rcParams["font.sans-serif"] = [_CHOSEN_FONT]
else:
    mpl.rcParams["font.sans-serif"] = _FONT_CANDIDATES
mpl.rcParams["axes.unicode_minus"] = False

def _load_default_wells() -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    data_dir = Path(__file__).resolve().parent / "selected_weekly_data_9wells_common"
    summary_path = data_dir / "nine_wells_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        wells: List[Tuple[str, str]] = []
        well_types: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        for _, row in summary.iterrows():
            aquifer_type = str(row["chinese_type"])
            counters[aquifer_type] = counters.get(aquifer_type, 0) + 1
            label = f"{aquifer_type}{counters[aquifer_type]}"
            wells.append((str(data_dir / str(row["output_file"])), label))
            well_types[label] = aquifer_type
        return wells, well_types

    wells = []
    well_types = {}
    for name in sorted(os.listdir(".")):
        if (not name.lower().endswith(".csv")) or (not os.path.isfile(name)):
            continue
        base = os.path.splitext(name)[0]
        if any(k in base.lower() for k in ["metrics_summary", "predictions", "future_predictions", "peak_metrics"]):
            continue
        label = base.rstrip("_- ")
        wells.append((name, label))
        well_types[label] = label.rstrip("1234567890") or label
    return wells, well_types


WELLS, WELL_TYPES = _load_default_wells()

PEAK_MODEL_COLUMNS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "tcn": "TCN",
    "stacking": "Stacking",
}
@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_selection: np.ndarray
    y_selection: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_future_seed: np.ndarray
    future_known_features_scaled: np.ndarray
    y_future_holdout: np.ndarray
    dates_selection: np.ndarray
    dates_calib: np.ndarray
    dates_test: np.ndarray
    dates_future_holdout: np.ndarray
    idx_selection: np.ndarray
    idx_calib: np.ndarray
    idx_test: np.ndarray
    idx_future_holdout: np.ndarray
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
        # 多层 LSTM 编码时序信号，并使用最后时间步做回归。
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
        # Transformer 使用固定正弦位置编码。
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ExplainableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src, attn_weights if need_weights else None


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int, heads: int, layers: int, dropout: float):
        super().__init__()
        # 先映射到 d_model，再经多层自注意力编码，最后用末时刻表示回归。
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [ExplainableTransformerEncoderLayer(d_model=d_model, heads=heads, dropout=dropout) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = self.input_proj(x)
        x = self.pos(x)
        attn_last = None
        for layer in self.layers:
            x, attn = layer(x, need_weights=return_attention)
            if return_attention:
                attn_last = attn
        out = self.head(x[:, -1, :]).squeeze(-1)
        if return_attention:
            return out, attn_last
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        # TCN 空洞卷积用于扩大时间感受野。
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
    selection_ratio: float,
    calib_ratio: float,
    holdout_steps: int,
) -> SplitData:
    # 按时间顺序切分 train/val/calib/test。
    data = df[features].values
    if target != features[0]:
        raise ValueError("target must be the first feature.")
    if horizon != 1:
        raise ValueError("Strict future_holdout recursion currently expects horizon=1.")
    if holdout_steps <= 0:
        raise ValueError("holdout_steps must be positive.")
    if holdout_steps >= len(data) - lookback:
        raise ValueError("holdout_steps leaves no pre-holdout sequences.")
    if train_ratio + val_ratio + selection_ratio + calib_ratio >= 1:
        raise ValueError("train_ratio + val_ratio + selection_ratio + calib_ratio must be less than 1.0")

    holdout_start_idx = len(data) - holdout_steps
    n_sequences = holdout_start_idx - lookback - horizon + 1
    if n_sequences <= 0:
        raise ValueError("Not enough samples for the given lookback and horizon.")

    train_size = int(train_ratio * n_sequences)
    val_size = int(val_ratio * n_sequences)
    selection_size = int(selection_ratio * n_sequences)
    calib_size = int(calib_ratio * n_sequences)
    test_size = n_sequences - train_size - val_size - selection_size - calib_size
    if min(train_size, val_size, selection_size, calib_size, test_size) <= 0:
        raise ValueError("train/val/selection/calib/test split has empty part. Adjust ratios or data size.")

    # 仅在训练期拟合标准化器，避免信息泄漏。
    train_end_idx = lookback + train_size + horizon - 2
    scaler = StandardScaler()
    scaler.fit(data[: train_end_idx + 1])
    data_scaled = scaler.transform(data)

    X, y, idx = build_sequences(data_scaled, lookback, horizon, target_index=0)
    X, y, idx = X[:n_sequences], y[:n_sequences], idx[:n_sequences]

    X_train, y_train = X[:train_size], y[:train_size]
    val_start = train_size
    val_end = val_start + val_size
    X_val, y_val = X[val_start:val_end], y[val_start:val_end]

    selection_start = val_end
    selection_end = selection_start + selection_size
    X_selection, y_selection = X[selection_start:selection_end], y[selection_start:selection_end]

    calib_start = selection_end
    calib_end = calib_start + calib_size
    X_calib, y_calib = X[calib_start:calib_end], y[calib_start:calib_end]
    X_test, y_test = X[calib_end:], y[calib_end:]
    idx_selection = idx[selection_start:selection_end]
    idx_calib = idx[calib_start:calib_end]
    idx_test = idx[calib_end:]
    idx_future_holdout = np.arange(holdout_start_idx, len(data))
    dates_selection = df.loc[idx_selection, "Date"].values
    dates_calib = df.loc[idx_calib, "Date"].values
    dates_test = df.loc[idx_test, "Date"].values
    dates_future_holdout = df.loc[idx_future_holdout, "Date"].values
    X_future_seed = data_scaled[holdout_start_idx - lookback : holdout_start_idx]
    future_known_features_scaled = data_scaled[holdout_start_idx:]
    y_future_holdout = data_scaled[idx_future_holdout, 0]

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_selection=X_selection,
        y_selection=y_selection,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test,
        y_test=y_test,
        X_future_seed=X_future_seed,
        future_known_features_scaled=future_known_features_scaled,
        y_future_holdout=y_future_holdout,
        dates_selection=dates_selection,
        dates_calib=dates_calib,
        dates_test=dates_test,
        dates_future_holdout=dates_future_holdout,
        idx_selection=idx_selection,
        idx_calib=idx_calib,
        idx_test=idx_test,
        idx_future_holdout=idx_future_holdout,
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
    # 仅对目标列（第0列）做反标准化。
    zeros = np.zeros((len(y_scaled), n_features))
    zeros[:, 0] = y_scaled
    return scaler.inverse_transform(zeros)[:, 0]


def persistence_for_indices(df: pd.DataFrame, target_idx: np.ndarray, target: str = "GWL") -> np.ndarray:
    target_idx = np.asarray(target_idx, dtype=int)
    if len(target_idx) == 0:
        return np.array([], dtype=float)
    if np.any(target_idx <= 0):
        raise ValueError("Persistence baseline needs a previous observed target.")
    return df[target].iloc[target_idx - 1].to_numpy(dtype=float)


def recursive_persistence(last_actual: float, steps: int) -> np.ndarray:
    return np.full(int(steps), float(last_actual), dtype=float)


def stack_predict_scaled(
    lstm: nn.Module,
    transformer: nn.Module,
    tcn: nn.Module,
    xgb: XGBRegressor,
    X: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_lstm = predict(lstm, X, device)
    pred_trans = predict(transformer, X, device)
    pred_tcn = predict(tcn, X, device)
    stack_X = np.column_stack([pred_lstm, pred_trans, pred_tcn])
    pred_stack = stack_X.mean(axis=1) + xgb.predict(stack_X).reshape(-1)
    return pred_lstm, pred_trans, pred_tcn, pred_stack


def inverse_prediction_bundle(
    scaler: StandardScaler,
    n_features: int,
    pred_lstm: np.ndarray,
    pred_trans: np.ndarray,
    pred_tcn: np.ndarray,
    pred_stack: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "LSTM": inverse_target(scaler, pred_lstm, n_features),
        "Transformer": inverse_target(scaler, pred_trans, n_features),
        "TCN": inverse_target(scaler, pred_tcn, n_features),
        "Stacking": inverse_target(scaler, pred_stack, n_features),
    }


def recursive_future_holdout_predictions(
    lstm: nn.Module,
    transformer: nn.Module,
    tcn: nn.Module,
    xgb: XGBRegressor,
    seed_seq: np.ndarray,
    known_features_scaled: np.ndarray,
    scaler: StandardScaler,
    n_features: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    seqs = {
        "LSTM": seed_seq.copy(),
        "Transformer": seed_seq.copy(),
        "TCN": seed_seq.copy(),
        "Stacking": seed_seq.copy(),
    }
    scaled_preds = {name: [] for name in seqs}

    for known_row in known_features_scaled:
        with torch.no_grad():
            for name, model in [("LSTM", lstm), ("Transformer", transformer), ("TCN", tcn)]:
                seq_t = torch.tensor(seqs[name][np.newaxis, :, :], dtype=torch.float32).to(device)
                pred = float(model(seq_t).detach().cpu().numpy().reshape(-1)[0])
                scaled_preds[name].append(pred)
                new_row = known_row.copy()
                new_row[0] = pred
                seqs[name] = np.vstack([seqs[name][1:], new_row])

            stack_seq_t = torch.tensor(seqs["Stacking"][np.newaxis, :, :], dtype=torch.float32).to(device)
            lstm_p = float(lstm(stack_seq_t).detach().cpu().numpy().reshape(-1)[0])
            trans_p = float(transformer(stack_seq_t).detach().cpu().numpy().reshape(-1)[0])
            tcn_p = float(tcn(stack_seq_t).detach().cpu().numpy().reshape(-1)[0])
            stack_X = np.array([[lstm_p, trans_p, tcn_p]])
            stack_p = float(stack_X.mean(axis=1)[0] + xgb.predict(stack_X).reshape(-1)[0])
            scaled_preds["Stacking"].append(stack_p)
            new_row = known_row.copy()
            new_row[0] = stack_p
            seqs["Stacking"] = np.vstack([seqs["Stacking"][1:], new_row])

    scaled_arrays = {name: np.asarray(values, dtype=float) for name, values in scaled_preds.items()}
    orig_arrays = {name: inverse_target(scaler, values, n_features) for name, values in scaled_arrays.items()}
    return scaled_arrays, orig_arrays


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
) -> Tuple[np.ndarray, np.ndarray]:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    tau2 = max(float(tau) ** 2, 1e-12)
    calib_dates = pd.to_datetime(calib_dates)
    target_dates = pd.to_datetime(target_dates)

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

        q95 = weighted_quantile(calib_scores, 0.95, w_mix)

        pi95_l[j] = yhat[j] - q95
        pi95_u[j] = yhat[j] + q95

    return pi95_l, pi95_u


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    return picp, mpiw


def make_time_frequency_plot(series: pd.Series, out_path: str) -> None:
    if stft is None:
        return
    # 使用 STFT 可视化时间-频率特征。
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
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", color="black")
    for name, pred in preds.items():
        plt.plot(dates, pred, label=name)
    if pi95 is not None:
        plt.fill_between(dates, pi95[0], pi95[1], color="#8A2BE2", alpha=0.15, label="PI95")
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
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", color="#FF6347")
    plt.plot(dates, pred, label="Stacking", color="#8A2BE2")
    if pi95 is not None:
        plt.fill_between(dates, pi95[0], pi95[1], color="#8A2BE2", alpha=0.15, label="PI95")
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
    pi95: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(history_dates, history_values, label="History", color="#FF6347")
    plt.plot(future_dates, future_values, label="Future", color="#00CED1")
    if pi95 is not None:
        plt.fill_between(future_dates, pi95[0], pi95[1], color="#00CED1", alpha=0.15, label="Future PI95")
    plt.title("Rolling Forecast")
    plt.xlabel("Date")
    plt.ylabel("GWL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _build_sequence_feature_names(feature_names: List[str], lookback: int) -> List[str]:
    names = []
    for step in range(lookback):
        lag = lookback - 1 - step
        for feat in feature_names:
            names.append(f"t-{lag}_{feat}")
    return names


def _save_force_plot_html(
    shap_values,
    sample_idx: int,
    out_html_path: str,
    out_fallback_png: str,
    max_display: int,
    feature_names_override: Optional[List[str]] = None,
) -> None:
    # 优先输出稳定的静态 waterfall PNG，避免 force-HTML 在部分环境失败。
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[sample_idx], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_fallback_png, dpi=150)
    plt.close()


def _to_explanation(values, data: np.ndarray, feature_names: List[str], base_value: float):
    if shap is None:
        return None
    if isinstance(values, shap.Explanation):
        return values
    base_values = np.full(shape=(data.shape[0],), fill_value=float(base_value), dtype=float)
    return shap.Explanation(values=np.asarray(values), base_values=base_values, data=data, feature_names=feature_names)


def explain_transformer_attention(
    transformer: TransformerRegressor,
    X_explain: np.ndarray,
    out_dir: str,
    device: torch.device,
    explain_sample_index: int,
) -> None:
    if len(X_explain) == 0:
        return

    target_idx = int(np.clip(explain_sample_index, 0, len(X_explain) - 1))
    sample = X_explain[target_idx : target_idx + 1]
    lookback = sample.shape[1]

    x_one = torch.tensor(sample, dtype=torch.float32, device=device)
    transformer.eval()
    with torch.inference_mode():
        _, attn = transformer(x_one, return_attention=True)

    if attn is None:
        return

    attn_np = attn.detach().cpu().numpy()[0]
    attn_avg = attn_np.mean(axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        attn_avg,
        cmap="magma",
        xticklabels=[f"t-{lookback - 1 - i}" for i in range(lookback)],
        yticklabels=[f"t-{lookback - 1 - i}" for i in range(lookback)],
    )
    plt.title("Transformer Attention (last layer, heads avg)")
    plt.xlabel("Key time step")
    plt.ylabel("Query time step")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transformer_attention_heatmap_sample.png"), dpi=150)
    plt.close()


def explain_transformer_with_shap(
    transformer: TransformerRegressor,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    out_dir: str,
    device: torch.device,
    shap_bg_samples: int,
    shap_explain_samples: int,
    explain_sample_index: int,
    seed: int,
) -> None:
    if shap is None:
        return

    if len(X_background) == 0 or len(X_explain) == 0:
        return

    rng = np.random.default_rng(seed)
    bg_n = min(shap_bg_samples, len(X_background))
    ex_n = min(shap_explain_samples, len(X_explain))
    bg_idx = rng.choice(len(X_background), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_explain), size=ex_n, replace=False)

    X_bg_seq = X_background[bg_idx]
    X_ex_seq = X_explain[ex_idx]
    lookback = X_ex_seq.shape[1]
    n_features = X_ex_seq.shape[2]

    X_bg_flat = X_bg_seq.reshape(bg_n, lookback * n_features)
    X_ex_flat = X_ex_seq.reshape(ex_n, lookback * n_features)
    flat_feature_names = _build_sequence_feature_names(feature_names, lookback)

    transformer.eval()

    def predict_flat(x_flat: np.ndarray) -> np.ndarray:
        x_seq = x_flat.reshape(-1, lookback, n_features).astype(np.float32)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device)
        with torch.inference_mode():
            pred = transformer(x_tensor)
        return pred.detach().cpu().numpy().reshape(-1)

    masker = shap.maskers.Independent(X_bg_flat, max_samples=min(100, len(X_bg_flat)))
    explainer = shap.Explainer(predict_flat, masker, feature_names=flat_feature_names)
    shap_values_raw = explainer(X_ex_flat)

    if isinstance(shap_values_raw, shap.Explanation):
        shap_values = shap_values_raw
    else:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected = np.asarray(expected).reshape(-1)[0]
        shap_values = _to_explanation(shap_values_raw, X_ex_flat, flat_feature_names, float(expected))

    plt.figure(figsize=(10, 4))
    shap.plots.beeswarm(shap_values, max_display=min(20, len(flat_feature_names)), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transformer_shap_beeswarm.png"), dpi=150)
    plt.close()

    target_idx = int(np.clip(explain_sample_index, 0, ex_n - 1))
    sv_one = shap_values.values[target_idx].reshape(lookback, n_features)

    plt.figure(figsize=(7, 4))
    sns.heatmap(
        sv_one,
        cmap="coolwarm",
        center=0.0,
        xticklabels=feature_names,
        yticklabels=[f"t-{lookback - 1 - i}" for i in range(lookback)],
    )
    plt.title("Transformer SHAP (time x feature)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transformer_shap_heatmap_sample.png"), dpi=150)
    plt.close()

    _save_force_plot_html(
        shap_values=shap_values,
        sample_idx=target_idx,
        out_html_path=os.path.join(out_dir, "transformer_shap_force_sample.html"),
        out_fallback_png=os.path.join(out_dir, "transformer_shap_waterfall_sample.png"),
        max_display=min(20, len(flat_feature_names)),
        feature_names_override=flat_feature_names,
    )



def explain_stacking_with_shap(
    xgb: XGBRegressor,
    stack_features: np.ndarray,
    out_dir: str,
) -> None:
    if shap is None or len(stack_features) == 0:
        return

    names = ["Pred_LSTM", "Pred_Transformer", "Pred_TCN"]
    explainer = shap.TreeExplainer(xgb)
    shap_values_raw = explainer(stack_features)

    if isinstance(shap_values_raw, shap.Explanation):
        shap_values = shap_values_raw
    else:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected = np.asarray(expected).reshape(-1)[0]
        shap_values = _to_explanation(shap_values_raw, stack_features, names, float(expected))

    plt.figure(figsize=(7, 4))
    shap.plots.beeswarm(shap_values, max_display=3, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stacking_xgb_shap_beeswarm.png"), dpi=150)
    plt.close()

    _save_force_plot_html(
        shap_values=shap_values,
        sample_idx=0,
        out_html_path=os.path.join(out_dir, "stacking_xgb_shap_force_sample.html"),
        out_fallback_png=os.path.join(out_dir, "stacking_xgb_shap_waterfall_sample.png"),
        max_display=3,
        feature_names_override=names,
    )



def _resolve_peak_tolerance(length: int, peak_tolerance: Optional[int]) -> int:
    if peak_tolerance is not None and peak_tolerance > 0:
        return int(peak_tolerance)
    return int(np.clip(max(1, length // 30), 3, 5))


def detect_peaks_adaptive(
    series: np.ndarray,
    prominence_scale: Optional[float] = None,
    distance_min: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    values = np.asarray(series, dtype=float)
    if find_peaks is None or len(values) < 3:
        return np.array([], dtype=int), {}, {"prominence": 0.0, "distance": 1.0}

    std_val = float(np.nanstd(values))
    q75, q25 = np.nanpercentile(values, [75, 25])
    iqr_val = float(q75 - q25)
    span = float(np.nanmax(values) - np.nanmin(values))

    base_scale = max(std_val, iqr_val / 1.349, 1e-8)
    scale = float(prominence_scale) if prominence_scale is not None else 0.8
    lower_bound = max(0.05 * span, 1e-6)
    prominence = max(scale * base_scale, lower_bound)

    auto_distance = max(1, len(values) // 20)
    min_distance = int(distance_min) if distance_min is not None else 1
    distance = max(min_distance, auto_distance)

    peaks, properties = find_peaks(values, prominence=prominence, distance=distance)
    return peaks.astype(int), properties, {"prominence": float(prominence), "distance": float(distance)}


def match_peaks_with_tolerance(
    true_peaks: np.ndarray,
    pred_peaks: np.ndarray,
    tolerance: int,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    true_arr = np.asarray(true_peaks, dtype=int)
    pred_arr = np.asarray(pred_peaks, dtype=int)
    used_pred: set[int] = set()
    matched_pairs: List[Tuple[int, int]] = []

    for t_idx in true_arr:
        candidates = []
        for p_idx in pred_arr:
            if p_idx in used_pred:
                continue
            gap = abs(int(t_idx) - int(p_idx))
            if gap <= tolerance:
                candidates.append((gap, int(p_idx)))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        best_pred = candidates[0][1]
        used_pred.add(best_pred)
        matched_pairs.append((int(t_idx), int(best_pred)))

    tp = len(matched_pairs)
    fp = int(len(pred_arr) - tp)
    fn = int(len(true_arr) - tp)
    return matched_pairs, tp, fp, fn


def calc_peak_metrics(
    true_series: np.ndarray,
    pred_series: np.ndarray,
    true_peaks: np.ndarray,
    pred_peaks: np.ndarray,
    matched_pairs: List[Tuple[int, int]],
    tp: int,
    fp: int,
    fn: int,
) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    detection_rate = tp / len(true_peaks) if len(true_peaks) > 0 else 0.0

    amp_errors = []
    amp_relative_errors = []
    timing_errors = []
    for t_idx, p_idx in matched_pairs:
        true_amp = float(true_series[t_idx])
        pred_amp = float(pred_series[p_idx])
        abs_err = abs(true_amp - pred_amp)
        amp_errors.append(abs_err)
        if abs(true_amp) > 1e-8:
            amp_relative_errors.append(abs_err / abs(true_amp))
        timing_errors.append(abs(t_idx - p_idx))

    amplitude_mae = float(np.mean(amp_errors)) if amp_errors else np.nan
    amplitude_rmse = float(np.sqrt(np.mean(np.square(amp_errors)))) if amp_errors else np.nan
    amplitude_mape = float(np.mean(amp_relative_errors) * 100) if amp_relative_errors else np.nan
    max_amplitude_error = float(np.max(amp_errors)) if amp_errors else np.nan

    mean_timing_error = float(np.mean(timing_errors)) if timing_errors else np.nan
    std_timing_error = float(np.std(timing_errors)) if timing_errors else np.nan
    max_timing_error = float(np.max(timing_errors)) if timing_errors else np.nan

    amplitude_score = 1.0 / (1.0 + amplitude_mape / 100.0) if np.isfinite(amplitude_mape) else 0.0
    timing_score = 1.0 / (1.0 + mean_timing_error) if np.isfinite(mean_timing_error) else 0.0
    composite_score = 0.5 * f1 + 0.3 * amplitude_score + 0.2 * timing_score

    return {
        "true_peak_count": int(len(true_peaks)),
        "pred_peak_count": int(len(pred_peaks)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "detection_rate": float(detection_rate),
        "amplitude_mae": amplitude_mae,
        "amplitude_rmse": amplitude_rmse,
        "amplitude_mape": amplitude_mape,
        "max_amplitude_error": max_amplitude_error,
        "mean_timing_error": mean_timing_error,
        "std_timing_error": std_timing_error,
        "max_timing_error": max_timing_error,
        "composite_score": float(composite_score),
    }


def evaluate_peak_for_models(
    pred_df: pd.DataFrame,
    well: str,
    model_columns: List[str],
    peak_tolerance: Optional[int],
    peak_prominence_scale: Optional[float],
    peak_distance_min: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    true_series = pred_df["Actual"].to_numpy(dtype=float)
    true_peaks, _, true_params = detect_peaks_adaptive(
        true_series,
        prominence_scale=peak_prominence_scale,
        distance_min=peak_distance_min,
    )
    tolerance = _resolve_peak_tolerance(len(true_series), peak_tolerance)

    rows = []
    details: Dict[str, Dict[str, Any]] = {}
    for model in model_columns:
        pred_series = pred_df[model].to_numpy(dtype=float)
        pred_peaks, _, pred_params = detect_peaks_adaptive(
            pred_series,
            prominence_scale=peak_prominence_scale,
            distance_min=peak_distance_min,
        )
        matched_pairs, tp, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=tolerance)
        metric_map = calc_peak_metrics(
            true_series=true_series,
            pred_series=pred_series,
            true_peaks=true_peaks,
            pred_peaks=pred_peaks,
            matched_pairs=matched_pairs,
            tp=tp,
            fp=fp,
            fn=fn,
        )
        row = {"well": well, "model": model}
        row.update(metric_map)
        rows.append(row)

        amplitude_errors = [abs(float(true_series[t]) - float(pred_series[p])) for t, p in matched_pairs]
        details[model] = {
            "true_peaks": true_peaks,
            "pred_peaks": pred_peaks,
            "matched_pairs": matched_pairs,
            "amplitude_errors": amplitude_errors,
            "true_params": true_params,
            "pred_params": pred_params,
            "tolerance": tolerance,
        }

    columns = [
        "well",
        "model",
        "true_peak_count",
        "pred_peak_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "detection_rate",
        "amplitude_mae",
        "amplitude_rmse",
        "amplitude_mape",
        "max_amplitude_error",
        "mean_timing_error",
        "std_timing_error",
        "max_timing_error",
        "composite_score",
    ]
    result_df = pd.DataFrame(rows)
    if len(result_df) == 0:
        result_df = pd.DataFrame(columns=columns)
    else:
        result_df = result_df[columns]
    return result_df, details


def plot_peak_detection(
    dates: np.ndarray,
    actual: np.ndarray,
    pred: np.ndarray,
    true_peaks: np.ndarray,
    pred_peaks: np.ndarray,
    matched_pairs: List[Tuple[int, int]],
    model_name: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", color="black", linewidth=1.8)
    plt.plot(dates, pred, label=model_name, color="#1f77b4", linewidth=1.4, alpha=0.9)

    if len(true_peaks) > 0:
        plt.scatter(dates[true_peaks], actual[true_peaks], color="#2ca02c", marker="o", s=40, label="Actual Peaks")
    if len(pred_peaks) > 0:
        plt.scatter(dates[pred_peaks], pred[pred_peaks], color="#d62728", marker="x", s=45, label="Pred Peaks")

    for t_idx, p_idx in matched_pairs:
        plt.plot([dates[t_idx], dates[p_idx]], [actual[t_idx], pred[p_idx]], color="gray", alpha=0.35, linestyle="--")

    plt.title(f"Peak Detection - {model_name}")
    plt.xlabel("Date")
    plt.ylabel("GWL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_peak_radar_compare(peak_df: pd.DataFrame, out_path: str) -> None:
    labels = ["Precision", "Recall", "F1", "Detection Rate", "Amplitude Accuracy", "Timing Accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for _, row in peak_df.iterrows():
        amp_score = 1.0 / (1.0 + row["amplitude_mape"] / 100.0) if np.isfinite(row["amplitude_mape"]) else 0.0
        timing_score = 1.0 / (1.0 + row["mean_timing_error"]) if np.isfinite(row["mean_timing_error"]) else 0.0
        vals = [
            float(row["precision"]),
            float(row["recall"]),
            float(row["f1"]),
            float(row["detection_rate"]),
            float(amp_score),
            float(timing_score),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=row["model"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("Peak Capability Radar")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_peak_amplitude_error_boxplot(amplitude_errors: Dict[str, List[float]], out_path: str) -> None:
    rows = []
    for model_name, errs in amplitude_errors.items():
        for err in errs:
            rows.append({"model": model_name, "amplitude_error": float(err)})

    plt.figure(figsize=(9, 4))
    if rows:
        err_df = pd.DataFrame(rows)
        sns.boxplot(data=err_df, x="model", y="amplitude_error", color="#87CEEB")
        sns.stripplot(data=err_df, x="model", y="amplitude_error", color="gray", alpha=0.4, jitter=0.25, size=3)
        plt.ylabel("Absolute Amplitude Error")
    else:
        plt.text(0.5, 0.5, "No matched peak pairs for amplitude errors", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title("Peak Amplitude Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_future_peak_display(
    history_dates: np.ndarray,
    history_values: np.ndarray,
    future_dates: np.ndarray,
    future_values: np.ndarray,
    out_path: str,
    peak_prominence_scale: Optional[float],
    peak_distance_min: Optional[int],
    history_tail: int = 52,
) -> None:
    future_values = np.asarray(future_values, dtype=float)
    future_peaks, _, _ = detect_peaks_adaptive(
        future_values,
        prominence_scale=peak_prominence_scale,
        distance_min=peak_distance_min,
    )

    tail_n = min(history_tail, len(history_values))
    plt.figure(figsize=(12, 4))
    plt.plot(history_dates[-tail_n:], history_values[-tail_n:], label="History (tail)", color="#FF6347")
    plt.plot(future_dates, future_values, label="Future Stacking", color="#00CED1")
    if len(future_peaks) > 0:
        plt.scatter(
            future_dates[future_peaks],
            future_values[future_peaks],
            marker="x",
            color="#d62728",
            s=48,
            label="Future Peaks",
        )
    plt.title("Future Peak Display (No Evaluation)")
    plt.xlabel("Date")
    plt.ylabel("GWL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize_peak_metrics(out_dir: str) -> None:
    peak_frames: List[pd.DataFrame] = []
    for item in sorted(os.listdir(out_dir)):
        well_dir = os.path.join(out_dir, item)
        if not os.path.isdir(well_dir):
            continue
        peak_path = os.path.join(well_dir, "peak", "peak_metrics.csv")
        if os.path.exists(peak_path):
            peak_frames.append(pd.read_csv(peak_path))

    if not peak_frames:
        return

    merged = pd.concat(peak_frames, ignore_index=True)
    merged.to_csv(os.path.join(out_dir, "peak_metrics_summary.csv"), index=False)

def run_well(
    file_path: str,
    aquifer: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    selection_ratio: float,
    calib_ratio: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    out_dir: str,
    holdout_steps: int,
    gamma: float,
    lambda_t: float,
    enable_intervals: bool,
    enable_explain: bool,
    shap_bg_samples: int,
    shap_explain_samples: int,
    explain_sample_index: int,
    seed: int,
    enable_peak_analysis: bool,
    peak_tolerance: Optional[int],
    peak_prominence_scale: Optional[float],
    peak_distance_min: Optional[int],
    peak_plot_models: List[str],
    dropout: float = 0.4,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    # 单井完整流程：数据切分、模型训练、集成预测、区间估计与结果落盘。
    df = load_data(file_path)
    features = ["GWL", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation"]
    target = "GWL"

    split = prepare_splits(
        df,
        features=features,
        target=target,
        lookback=lookback,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        selection_ratio=selection_ratio,
        calib_ratio=calib_ratio,
        holdout_steps=holdout_steps,
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

    lstm = LSTMRegressor(n_features=len(features), hidden=64, layers=2, dropout=dropout)
    transformer = TransformerRegressor(n_features=len(features), d_model=64, heads=4, layers=2, dropout=dropout)
    tcn = TCNRegressor(n_features=len(features), channels=[32, 32, 32], kernel=3, dropout=dropout)

    lstm = train_model(lstm, train_loader, val_loader, device, epochs, lr, patience)
    transformer = train_model(transformer, train_loader, val_loader, device, epochs, lr, patience)
    tcn = train_model(tcn, train_loader, val_loader, device, epochs, lr, patience)

    pred_val_lstm = predict(lstm, split.X_val, device)
    pred_val_trans = predict(transformer, split.X_val, device)
    pred_val_tcn = predict(tcn, split.X_val, device)

    # 使用 XGBoost 对基模型残差做二次学习（stacking）。
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

    n_features = len(features)
    pred_selection = stack_predict_scaled(lstm, transformer, tcn, xgb, split.X_selection, device)
    pred_calib = stack_predict_scaled(lstm, transformer, tcn, xgb, split.X_calib, device)
    pred_test = stack_predict_scaled(lstm, transformer, tcn, xgb, split.X_test, device)
    stack_test_X = np.column_stack(pred_test[:3])

    pred_selection_orig = inverse_prediction_bundle(split.scaler, n_features, *pred_selection)
    pred_calib_orig = inverse_prediction_bundle(split.scaler, n_features, *pred_calib)
    pred_test_orig = inverse_prediction_bundle(split.scaler, n_features, *pred_test)

    y_selection_orig = inverse_target(split.scaler, split.y_selection, n_features)
    y_calib_orig = inverse_target(split.scaler, split.y_calib, n_features)
    y_test_orig = inverse_target(split.scaler, split.y_test, n_features)
    y_future_orig = inverse_target(split.scaler, split.y_future_holdout, n_features)

    _, pred_future_orig = recursive_future_holdout_predictions(
        lstm=lstm,
        transformer=transformer,
        tcn=tcn,
        xgb=xgb,
        seed_seq=split.X_future_seed,
        known_features_scaled=split.future_known_features_scaled,
        scaler=split.scaler,
        n_features=n_features,
        device=device,
    )
    test_persistence = persistence_for_indices(df, split.idx_test, target=target)
    future_persistence = recursive_persistence(df[target].iloc[split.idx_future_holdout[0] - 1], len(split.idx_future_holdout))

    metrics_map = {
        "selection": {
            "Persistence": metrics(y_selection_orig, persistence_for_indices(df, split.idx_selection, target=target)),
            "LSTM": metrics(y_selection_orig, pred_selection_orig["LSTM"]),
            "Transformer": metrics(y_selection_orig, pred_selection_orig["Transformer"]),
            "TCN": metrics(y_selection_orig, pred_selection_orig["TCN"]),
            "Stacking": metrics(y_selection_orig, pred_selection_orig["Stacking"]),
        },
        "test": {
            "Persistence": metrics(y_test_orig, test_persistence),
            "LSTM": metrics(y_test_orig, pred_test_orig["LSTM"]),
            "Transformer": metrics(y_test_orig, pred_test_orig["Transformer"]),
            "TCN": metrics(y_test_orig, pred_test_orig["TCN"]),
            "Stacking": metrics(y_test_orig, pred_test_orig["Stacking"]),
        },
        "future_holdout": {
            "Persistence": metrics(y_future_orig, future_persistence),
            "LSTM": metrics(y_future_orig, pred_future_orig["LSTM"]),
            "Transformer": metrics(y_future_orig, pred_future_orig["Transformer"]),
            "TCN": metrics(y_future_orig, pred_future_orig["TCN"]),
            "Stacking": metrics(y_future_orig, pred_future_orig["Stacking"]),
        },
    }

    pi95_l_test = pi95_u_test = None
    pi95_l_future_holdout = pi95_u_future_holdout = None
    z_mu = z_std = z_calib_std = calib_scores = tau_init = None

    if enable_intervals:
        calib_scores = np.abs(y_calib_orig - pred_calib_orig["Stacking"])
        z_calib_raw = np.column_stack([pred_calib_orig["LSTM"], pred_calib_orig["Transformer"], pred_calib_orig["TCN"]])
        z_mu = z_calib_raw.mean(axis=0)
        z_std = z_calib_raw.std(axis=0)
        z_std = np.where(z_std <= 1e-8, 1.0, z_std)
        z_calib_std = (z_calib_raw - z_mu) / z_std

        tau_init = init_tau_from_calib_z(z_calib_std)

        z_test_raw = np.column_stack([pred_test_orig["LSTM"], pred_test_orig["Transformer"], pred_test_orig["TCN"]])
        z_test_std = (z_test_raw - z_mu) / z_std

        pi95_l_test, pi95_u_test = compute_weighted_conformal_intervals(
            yhat=pred_test_orig["Stacking"],
            z_target_std=z_test_std,
            target_dates=split.dates_test,
            calib_scores=calib_scores,
            z_calib_std=z_calib_std,
            calib_dates=split.dates_calib,
            gamma=gamma,
            lambda_t=lambda_t,
            tau=tau_init,
        )
        picp95, mpiw95 = interval_metrics(y_test_orig, pi95_l_test, pi95_u_test)
        metrics_map["test"]["Stacking"].update({"PICP95": picp95, "MPIW95": mpiw95, "TAU_INIT": tau_init})

        z_future_raw = np.column_stack([pred_future_orig["LSTM"], pred_future_orig["Transformer"], pred_future_orig["TCN"]])
        z_future_std = (z_future_raw - z_mu) / z_std
        pi95_l_future_holdout, pi95_u_future_holdout = compute_weighted_conformal_intervals(
            yhat=pred_future_orig["Stacking"],
            z_target_std=z_future_std,
            target_dates=split.dates_future_holdout,
            calib_scores=calib_scores,
            z_calib_std=z_calib_std,
            calib_dates=split.dates_calib,
            gamma=gamma,
            lambda_t=lambda_t,
            tau=tau_init,
        )
        picp95, mpiw95 = interval_metrics(y_future_orig, pi95_l_future_holdout, pi95_u_future_holdout)
        metrics_map["future_holdout"]["Stacking"].update({"PICP95": picp95, "MPIW95": mpiw95, "TAU_INIT": tau_init})

    os.makedirs(out_dir, exist_ok=True)
    peak_dir = os.path.join(out_dir, "peak")
    pred_data = {
        "Date": split.dates_test,
        "Actual": y_test_orig,
        "Persistence": test_persistence,
        "LSTM": pred_test_orig["LSTM"],
        "Transformer": pred_test_orig["Transformer"],
        "TCN": pred_test_orig["TCN"],
        "Stacking": pred_test_orig["Stacking"],
        "Aquifer": aquifer,
    }
    if enable_intervals:
        pred_data["PI95_Lower"] = pi95_l_test
        pred_data["PI95_Upper"] = pi95_u_test
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    future_data = {
        "Date": split.dates_future_holdout,
        "Actual": y_future_orig,
        "Persistence": future_persistence,
        "LSTM": pred_future_orig["LSTM"],
        "Transformer": pred_future_orig["Transformer"],
        "TCN": pred_future_orig["TCN"],
        "Stacking": pred_future_orig["Stacking"],
        "Aquifer": aquifer,
    }
    if enable_intervals:
        future_data["PI95_Lower"] = pi95_l_future_holdout
        future_data["PI95_Upper"] = pi95_u_future_holdout
    future_df = pd.DataFrame(future_data)
    future_df.to_csv(os.path.join(out_dir, "future_holdout_predictions.csv"), index=False)

    if enable_peak_analysis and find_peaks is not None:
        os.makedirs(peak_dir, exist_ok=True)
        eval_model_columns = ["LSTM", "Transformer", "TCN", "Stacking"]
        peak_df, peak_details = evaluate_peak_for_models(
            pred_df=pred_df,
            well=aquifer,
            model_columns=eval_model_columns,
            peak_tolerance=peak_tolerance,
            peak_prominence_scale=peak_prominence_scale,
            peak_distance_min=peak_distance_min,
        )
        peak_df.to_csv(os.path.join(peak_dir, "peak_metrics.csv"), index=False)

        selected_keys = [k.strip().lower() for k in peak_plot_models if k.strip()]
        if not selected_keys:
            selected_keys = list(PEAK_MODEL_COLUMNS.keys())
        selected_columns = [PEAK_MODEL_COLUMNS[k] for k in selected_keys if k in PEAK_MODEL_COLUMNS]
        if not selected_columns:
            selected_columns = eval_model_columns

        for model_name in selected_columns:
            details = peak_details.get(model_name)
            if details is None:
                continue
            plot_peak_detection(
                dates=pred_df["Date"].to_numpy(),
                actual=pred_df["Actual"].to_numpy(dtype=float),
                pred=pred_df[model_name].to_numpy(dtype=float),
                true_peaks=np.asarray(details["true_peaks"], dtype=int),
                pred_peaks=np.asarray(details["pred_peaks"], dtype=int),
                matched_pairs=details["matched_pairs"],
                model_name=model_name,
                out_path=os.path.join(peak_dir, f"peak_detection_{model_name.lower()}.png"),
            )

        plot_peak_radar_compare(peak_df=peak_df, out_path=os.path.join(peak_dir, "peak_radar_compare.png"))
        amp_map = {model: peak_details[model]["amplitude_errors"] for model in selected_columns if model in peak_details}
        plot_peak_amplitude_error_boxplot(
            amplitude_errors=amp_map,
            out_path=os.path.join(peak_dir, "peak_amplitude_error_boxplot.png"),
        )
    elif enable_peak_analysis:
        print("[Peak] scipy.signal.find_peaks is unavailable; peak analysis skipped.")

    plot_predictions(
        split.dates_test,
        y_test_orig,
        {
            "Persistence": test_persistence,
            "LSTM": pred_test_orig["LSTM"],
            "Transformer": pred_test_orig["Transformer"],
            "TCN": pred_test_orig["TCN"],
            "Stacking": pred_test_orig["Stacking"],
        },
        os.path.join(out_dir, "test_predictions.png"),
        pi95=(pi95_l_test, pi95_u_test) if enable_intervals else None,
    )

    plot_stacking(
        split.dates_test,
        y_test_orig,
        pred_test_orig["Stacking"],
        os.path.join(out_dir, "stacking_predictions.png"),
        pi95=(pi95_l_test, pi95_u_test) if enable_intervals else None,
    )

    plot_residual_hist(
        y_test_orig - pred_test_orig["Stacking"],
        os.path.join(out_dir, "stacking_residuals.png"),
    )

    make_time_frequency_plot(df["GWL"], os.path.join(out_dir, "time_frequency.png"))

    plot_predictions(
        split.dates_future_holdout,
        y_future_orig,
        {
            "Persistence": future_persistence,
            "LSTM": pred_future_orig["LSTM"],
            "Transformer": pred_future_orig["Transformer"],
            "TCN": pred_future_orig["TCN"],
            "Stacking": pred_future_orig["Stacking"],
        },
        os.path.join(out_dir, "future_holdout_predictions.png"),
        pi95=(pi95_l_future_holdout, pi95_u_future_holdout) if enable_intervals else None,
    )



    if enable_peak_analysis and find_peaks is not None:
        plot_future_peak_display(
            history_dates=df["Date"].values,
            history_values=df["GWL"].values,
            future_dates=split.dates_future_holdout,
            future_values=pred_future_orig["Stacking"],
            out_path=os.path.join(peak_dir, "future_peak_display.png"),
            peak_prominence_scale=peak_prominence_scale,
            peak_distance_min=peak_distance_min,
        )
    if enable_explain:
        explain_dir = os.path.join(out_dir, "explain")
        os.makedirs(explain_dir, exist_ok=True)

        explain_transformer_attention(
            transformer=transformer,
            X_explain=split.X_test,
            out_dir=explain_dir,
            device=device,
            explain_sample_index=explain_sample_index,
        )

        if shap is not None:
            explain_transformer_with_shap(
                transformer=transformer,
                X_background=split.X_train,
                X_explain=split.X_test,
                feature_names=features,
                out_dir=explain_dir,
                device=device,
                shap_bg_samples=shap_bg_samples,
                shap_explain_samples=shap_explain_samples,
                explain_sample_index=explain_sample_index,
                seed=seed,
            )
            explain_stacking_with_shap(
                xgb=xgb,
                stack_features=stack_test_X,
                out_dir=explain_dir,
            )

    return pred_df, metrics_map


def summarize_metrics(
    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: str,
    include_splits: Tuple[str, ...] = ("test", "future_holdout"),
) -> None:
    rows = []
    for well_id, split_metrics in all_metrics.items():
        for split_name, model_metrics in split_metrics.items():
            if split_name not in include_splits:
                continue
            for model_name, metric_values in model_metrics.items():
                row = {
                    "split": split_name,
                    "aquifer_type": WELL_TYPES.get(well_id, well_id.rstrip("1234567890") or well_id),
                    "well": well_id,
                    "model": model_name,
                }
                row.update(metric_values)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    if df.empty:
        return

    metric_cols = [col for col in ["RMSE", "MAE", "R2", "NSE", "PICP95", "MPIW95", "TAU_INIT"] if col in df.columns]
    type_df = df.groupby(["split", "aquifer_type", "model"], as_index=False)[metric_cols].mean(numeric_only=True)
    type_df.to_csv(os.path.join(out_dir, "metrics_by_type_summary.csv"), index=False)

    g = sns.catplot(data=type_df, x="model", y="RMSE", hue="aquifer_type", col="split", kind="bar", height=4, aspect=1.3)
    g.fig.suptitle("Mean RMSE by Split, Model, and Aquifer Type")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(out_dir, "rmse_comparison.png"), dpi=150)
    plt.close(g.fig)

    g = sns.catplot(data=type_df, x="model", y="NSE", hue="aquifer_type", col="split", kind="bar", height=4, aspect=1.3)
    g.fig.suptitle("Mean NSE by Split, Model, and Aquifer Type")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(out_dir, "nse_comparison.png"), dpi=150)
    plt.close(g.fig)


def main() -> None:
    # 绋嬪簭鍏ュ彛锛氳В鏋愬弬鏁板苟閫愪簳杩愯銆?
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--selection_ratio", type=float, default=0.1)
    parser.add_argument("--calib_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--holdout_steps", type=int, default=30)
    parser.add_argument("--future_steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda_t", type=float, default=0.02)
    parser.add_argument("--disable_intervals", action="store_true")
    parser.add_argument("--disable_explain", action="store_true")
    parser.set_defaults(enable_peak_analysis=True)
    parser.add_argument("--enable_peak_analysis", dest="enable_peak_analysis", action="store_true")
    parser.add_argument("--disable_peak_analysis", dest="enable_peak_analysis", action="store_false")
    parser.add_argument("--peak_tolerance", type=int, default=None)
    parser.add_argument("--peak_prominence_scale", type=float, default=None)
    parser.add_argument("--peak_distance_min", type=int, default=1)
    parser.add_argument("--peak_plot_models", type=str, default="lstm,transformer,tcn,stacking")
    parser.add_argument("--shap_bg_samples", type=int, default=80)
    parser.add_argument("--shap_explain_samples", type=int, default=40)
    parser.add_argument("--explain_sample_index", type=int, default=0)
    args = parser.parse_args()

    if args.future_steps is not None:
        args.holdout_steps = args.future_steps

    ratio_sum = args.train_ratio + args.val_ratio + args.selection_ratio + args.calib_ratio
    if ratio_sum >= 1.0:
        raise ValueError("train_ratio + val_ratio + selection_ratio + calib_ratio must be less than 1.0")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    peak_plot_models = [m.strip().lower() for m in args.peak_plot_models.split(",") if m.strip()]
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
            selection_ratio=args.selection_ratio,
            calib_ratio=args.calib_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=device,
            out_dir=out_dir,
            holdout_steps=args.holdout_steps,
            gamma=args.gamma,
            lambda_t=args.lambda_t,
            enable_intervals=not args.disable_intervals,
            enable_explain=not args.disable_explain,
            shap_bg_samples=args.shap_bg_samples,
            shap_explain_samples=args.shap_explain_samples,
            explain_sample_index=args.explain_sample_index,
            seed=args.seed,
            enable_peak_analysis=args.enable_peak_analysis,
            peak_tolerance=args.peak_tolerance,
            peak_prominence_scale=args.peak_prominence_scale,
            peak_distance_min=args.peak_distance_min,
            peak_plot_models=peak_plot_models,
            dropout=args.dropout,
        )
        all_metrics[well_id] = metrics_map

    summarize_metrics(all_metrics, args.out_dir)
    summarize_peak_metrics(args.out_dir)
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()












