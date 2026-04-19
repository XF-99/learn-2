# -*- coding: utf-8 -*-
"""Run dropout sweep experiments against a separate learn.py.

The script does not write to or modify the source training script. It loads
learn.py in memory, overrides the dropout used by LSTM/Transformer/TCN, and
writes experiment outputs to the requested output directory.
"""
import argparse
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent


def validate_dropout(value: float) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")


def parse_dropout_values(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        validate_dropout(value)
        values.append(value)
    if not values:
        raise ValueError("dropout_values must contain at least one value")
    return values


def dropout_dir_name(value: float) -> str:
    label = f"{value:.6g}"
    if "." not in label:
        label = f"{label}.0"
    return f"dropout_{label}"


def metrics_to_rows(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    rows = []
    for well_id, model_metrics in all_metrics.items():
        for model_name, metric_values in model_metrics.items():
            row = {"well": well_id, "model": model_name}
            row.update(metric_values)
            rows.append(row)
    return rows


def read_compatible_learn_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    # Tolerate an interrupted earlier edit by restoring the original fixed-dropout shape in memory.
    text = text.replace(
        "    dropout: float,`r`n) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:",
        ") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:",
    )
    text = text.replace("dropout=dropout", "dropout=0.2")
    text = text.replace(
        '    parser.add_argument("--dropout", type=float, default=0.2)`r`n'
        '    parser.add_argument("--dropout_values", type=str, default=None)`r`n'
        '    parser.add_argument("--seed", type=int, default=42)',
        '    parser.add_argument("--seed", type=int, default=42)',
    )
    partial_main = text.find("\n    dropout_values = parse_dropout_values(args.dropout_values)")
    if partial_main != -1:
        marker = text.find('\nif __name__ == "__main__":', partial_main)
        if marker == -1:
            raise RuntimeError("Found partial dropout edit in learn.py, but could not locate main guard.")
        original_main_tail = '''
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
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
            enable_peak_analysis=args.enable_peak_analysis,
            peak_tolerance=args.peak_tolerance,
            peak_prominence_scale=args.peak_prominence_scale,
            peak_distance_min=args.peak_distance_min,
            peak_plot_models=peak_plot_models,
        )
        all_metrics[well_id] = metrics_map

    summarize_metrics(all_metrics, args.out_dir)
    summarize_peak_metrics(args.out_dir)
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
'''
        text = text[:partial_main] + "\n" + original_main_tail + text[marker:]
    return text


def load_learn_module(source_dir: Path) -> types.ModuleType:
    learn_path = source_dir / "learn.py"
    if not learn_path.exists():
        raise FileNotFoundError(f"learn.py not found: {learn_path}")
    old_cwd = Path.cwd()
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    try:
        module = types.ModuleType("learn_dropout_source")
        module.__file__ = str(learn_path)
        exec(compile(read_compatible_learn_source(learn_path), str(learn_path), "exec"), module.__dict__)
        return module
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass


def apply_dropout_override(learn: types.ModuleType, originals: Dict[str, Any], value: float) -> None:
    validate_dropout(value)

    class LSTMRegressorWithDropout(originals["lstm"]):
        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
            super().__init__(n_features=n_features, hidden=hidden, layers=layers, dropout=value)

    class TransformerRegressorWithDropout(originals["transformer"]):
        def __init__(self, n_features: int, d_model: int, heads: int, layers: int, dropout: float):
            super().__init__(n_features=n_features, d_model=d_model, heads=heads, layers=layers, dropout=value)

    class TCNRegressorWithDropout(originals["tcn"]):
        def __init__(self, n_features: int, channels: List[int], kernel: int, dropout: float):
            super().__init__(n_features=n_features, channels=channels, kernel=kernel, dropout=value)

    learn.LSTMRegressor = LSTMRegressorWithDropout
    learn.TransformerRegressor = TransformerRegressorWithDropout
    learn.TCNRegressor = TCNRegressorWithDropout


def run_experiment(
    learn: types.ModuleType,
    originals: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    dropout: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    learn.set_seed(args.seed)
    apply_dropout_override(learn, originals, dropout)

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    peak_plot_models = [m.strip().lower() for m in args.peak_plot_models.split(",") if m.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(args.source_dir)
    try:
        for file_path, aquifer in learn.WELLS:
            well_out_dir = out_dir / aquifer
            run_kwargs = {
                "file_path": file_path,
                "aquifer": aquifer,
                "lookback": args.lookback,
                "horizon": args.horizon,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "calib_ratio": args.calib_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "device": device,
                "out_dir": str(well_out_dir),
                "future_steps": args.future_steps,
                "gamma": args.gamma,
                "lambda_t": args.lambda_t,
                "enable_explain": not args.disable_explain,
                "shap_bg_samples": args.shap_bg_samples,
                "shap_explain_samples": args.shap_explain_samples,
                "explain_sample_index": args.explain_sample_index,
                "seed": args.seed,
                "enable_peak_analysis": args.enable_peak_analysis,
                "peak_tolerance": args.peak_tolerance,
                "peak_prominence_scale": args.peak_prominence_scale,
                "peak_distance_min": args.peak_distance_min,
                "peak_plot_models": peak_plot_models,
            }
            supported = inspect.signature(learn.run_well).parameters
            run_kwargs = {key: value for key, value in run_kwargs.items() if key in supported}
            _, metrics_map = learn.run_well(**run_kwargs)
            all_metrics[aquifer] = metrics_map
    finally:
        os.chdir(old_cwd)

    learn.summarize_metrics(all_metrics, str(out_dir))
    if hasattr(learn, "summarize_peak_metrics"):
        learn.summarize_peak_metrics(str(out_dir))
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    return all_metrics


def summarize_dropout_sweep(sweep_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(sweep_rows)
    df.to_csv(out_dir / "dropout_sweep_metrics.csv", index=False)

    metric_cols = [col for col in ["MAE", "RMSE", "R2", "NSE", "PICP95", "MPIW95", "TAU_INIT"] if col in df.columns]
    summary = df.groupby(["dropout", "model"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join([part for part in col if part]) for col in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "dropout_sweep_summary.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="dropout", y="RMSE", hue="model", marker="o", ci="sd")
    plt.title("RMSE by Dropout and Model")
    plt.tight_layout()
    plt.savefig(out_dir / "dropout_rmse_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="dropout", y="NSE", hue="model", marker="o", ci="sd")
    plt.title("NSE by Dropout and Model")
    plt.tight_layout()
    plt.savefig(out_dir / "dropout_nse_comparison.png", dpi=150)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep dropout values for a separate learn.py.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dropout_values", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
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
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_dropout"))
    parser.add_argument("--future_steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda_t", type=float, default=0.02)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_dir = args.source_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.calib_ratio
    if ratio_sum >= 1.0:
        raise ValueError("train_ratio + val_ratio + calib_ratio must be less than 1.0")

    learn = load_learn_module(args.source_dir)
    originals = {
        "lstm": learn.LSTMRegressor,
        "transformer": learn.TransformerRegressor,
        "tcn": learn.TCNRegressor,
    }

    dropout_values = parse_dropout_values(args.dropout_values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: List[Dict[str, Any]] = []
    for dropout in dropout_values:
        run_dir = args.out_dir / dropout_dir_name(dropout)
        metrics_map = run_experiment(learn, originals, args, device, run_dir, dropout)
        for row in metrics_to_rows(metrics_map):
            sweep_rows.append({"dropout": dropout, **row})

    summarize_dropout_sweep(sweep_rows, args.out_dir)


if __name__ == "__main__":
    main()
