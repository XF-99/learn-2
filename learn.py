import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    from scipy.signal import stft
except Exception:  # pragma: no cover - optional dependency
    stft = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


# Prefer an installed CJK font to avoid "Glyph missing" warnings on Chinese labels.
_FONT_CANDIDATES = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
_INSTALLED_FONTS = {f.name for f in fm.fontManager.ttflist}
_CHOSEN_FONT = next((name for name in _FONT_CANDIDATES if name in _INSTALLED_FONTS), None)
if _CHOSEN_FONT is not None:
    mpl.rcParams["font.family"] = [_CHOSEN_FONT]
    mpl.rcParams["font.sans-serif"] = [_CHOSEN_FONT]
else:
    mpl.rcParams["font.sans-serif"] = _FONT_CANDIDATES
mpl.rcParams["axes.unicode_minus"] = False


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
        # ???????????????????????????????????
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
    # In this project we prefer a stable static artifact over fragile force-HTML export.
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
    enable_explain: bool,
    shap_bg_samples: int,
    shap_explain_samples: int,
    explain_sample_index: int,
    seed: int,
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

    pi95_l_test, pi95_u_test = compute_weighted_conformal_intervals(
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

    picp95, mpiw95 = interval_metrics(y_test_orig, pi95_l_test, pi95_u_test)

    metrics_map = {
        "LSTM": metrics(y_test_orig, pred_test_lstm_orig),
        "Transformer": metrics(y_test_orig, pred_test_trans_orig),
        "TCN": metrics(y_test_orig, pred_test_tcn_orig),
        "Stacking": {
            **metrics(y_test_orig, pred_test_stack_orig),
            "PICP95": picp95,
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
        pi95=(pi95_l_test, pi95_u_test),
    )

    plot_stacking(
        split.dates_test,
        y_test_orig,
        pred_test_stack_orig,
        os.path.join(out_dir, "stacking_predictions.png"),
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

    pi95_l_future, pi95_u_future = compute_weighted_conformal_intervals(
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
        pi95=(pi95_l_future, pi95_u_future),
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
    parser.add_argument("--disable_explain", action="store_true")
    parser.add_argument("--shap_bg_samples", type=int, default=80)
    parser.add_argument("--shap_explain_samples", type=int, default=40)
    parser.add_argument("--explain_sample_index", type=int, default=0)
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
            enable_explain=not args.disable_explain,
            shap_bg_samples=args.shap_bg_samples,
            shap_explain_samples=args.shap_explain_samples,
            explain_sample_index=args.explain_sample_index,
            seed=args.seed,
        )
        all_metrics[well_id] = metrics_map

    summarize_metrics(all_metrics, args.out_dir)
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
