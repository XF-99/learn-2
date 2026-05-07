# -*- coding: utf-8 -*-
"""Run lookback sweep experiments against learn.py.

The script keeps learn.py unchanged, runs the same training pipeline for each
lookback value, and writes both per-run outputs and cross-lookback summaries.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_LOOKBACK_VALUES = "12,18,26,32,38,44,50"
MODEL_FOR_BEST = "Stacking"


def parse_lookback_values(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("lookback values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("lookback_values must contain at least one value")
    return values


def lookback_dir_name(value: int) -> str:
    return f"lookback_{value}"


def metrics_to_rows(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    rows = []
    for well_id, model_metrics in all_metrics.items():
        for model_name, metric_values in model_metrics.items():
            row = {"well": well_id, "model": model_name}
            row.update(metric_values)
            rows.append(row)
    return rows


def load_learn_module(source_dir: Path):
    learn_path = source_dir / "learn.py"
    if not learn_path.exists():
        raise FileNotFoundError(f"learn.py not found: {learn_path}")

    module_name = "learn_lookback_source"
    spec = importlib.util.spec_from_file_location(module_name, learn_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load learn.py from {learn_path}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass


def run_experiment(
    learn,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    lookback: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    learn.set_seed(args.seed)
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(args.source_dir)
    try:
        for file_path, aquifer in learn.WELLS:
            well_out_dir = out_dir / aquifer
            run_kwargs = {
                "file_path": file_path,
                "aquifer": aquifer,
                "lookback": lookback,
                "horizon": args.horizon,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "device": device,
                "out_dir": str(well_out_dir),
                "future_steps": args.future_steps,
            }
            supported = inspect.signature(learn.run_well).parameters
            run_kwargs = {key: value for key, value in run_kwargs.items() if key in supported}
            _, metrics_map = learn.run_well(**run_kwargs)
            all_metrics[aquifer] = metrics_map
    finally:
        os.chdir(old_cwd)

    learn.summarize_metrics(all_metrics, str(out_dir))
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    return all_metrics


def flatten_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary.columns = ["_".join([part for part in col if part]) for col in summary.columns.to_flat_index()]
    return summary


def choose_best_lookback(summary: pd.DataFrame) -> Dict[str, Any]:
    stacking = summary[summary["model"] == MODEL_FOR_BEST].copy()
    if stacking.empty:
        raise ValueError(f"No {MODEL_FOR_BEST} rows found in lookback summary")

    stacking = stacking.sort_values(
        by=["NSE_mean", "RMSE_mean", "lookback"],
        ascending=[False, True, True],
    )
    best = stacking.iloc[0].to_dict()
    return {
        "selection_model": MODEL_FOR_BEST,
        "primary_metric": "NSE_mean",
        "primary_rule": "maximize",
        "tie_break_metric": "RMSE_mean",
        "tie_break_rule": "minimize",
        "best_lookback": int(best["lookback"]),
        "best_metrics": {
            key: float(value)
            for key, value in best.items()
            if key not in {"lookback", "model"} and pd.notna(value)
        },
    }


def summarize_lookback_sweep(sweep_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(sweep_rows)
    df.to_csv(out_dir / "lookback_sweep_metrics.csv", index=False)

    metric_cols = [col for col in ["MAE", "RMSE", "R2", "NSE"] if col in df.columns]
    summary = df.groupby(["lookback", "model"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary = flatten_summary_columns(summary)
    summary.to_csv(out_dir / "lookback_sweep_summary.csv", index=False)

    best = choose_best_lookback(summary)
    with (out_dir / "best_lookback.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="lookback", y="NSE", hue="model", marker="o", errorbar="sd")
    plt.title("NSE by Lookback and Model")
    plt.tight_layout()
    plt.savefig(out_dir / "lookback_nse_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="lookback", y="RMSE", hue="model", marker="o", errorbar="sd")
    plt.title("RMSE by Lookback and Model")
    plt.tight_layout()
    plt.savefig(out_dir / "lookback_rmse_comparison.png", dpi=150)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep lookback values for learn.py.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--lookback_values", type=str, default=DEFAULT_LOOKBACK_VALUES)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_lookback_sweep"))
    parser.add_argument("--future_steps", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_dir = args.source_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    learn = load_learn_module(args.source_dir)
    lookback_values = parse_lookback_values(args.lookback_values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: List[Dict[str, Any]] = []
    for lookback in lookback_values:
        run_dir = args.out_dir / lookback_dir_name(lookback)
        metrics_map = run_experiment(learn, args, device, run_dir, lookback)
        for row in metrics_to_rows(metrics_map):
            sweep_rows.append({"lookback": lookback, **row})

    summarize_lookback_sweep(sweep_rows, args.out_dir)


if __name__ == "__main__":
    main()
